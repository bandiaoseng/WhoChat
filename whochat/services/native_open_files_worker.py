"""Fresh per-process handle scans, contained by the owning broker's timeout.

No file contents are read. Blocking native name queries must run in this child
process, not in WhoChat's UI or capture thread. The parent owns its lifetime
through a Windows Job and accepts no partial evidence. There is no account or
file-list cache, and no call closes a handle in the inspected process.

API structure references: psutil PR 2932, merged commit
078f2f08d2ccea916a42f9571ab8bd40d958e7a1; Microsoft DuplicateHandle docs.
This Python prototype is independent of psutil's C implementation.
"""
import ctypes as C
from ctypes import wintypes as W
import json
import msvcrt
import os
import sys
import time
import math
from pathlib import PureWindowsPath
import psutil
from psutil._pswindows import convert_dos_path, isfile_strict

if os.name != 'nt' or C.sizeof(C.c_void_p) != 8:
    raise RuntimeError('Requires Windows x64')

K = C.WinDLL('kernel32', use_last_error=True, winmode=0x800)
N = C.WinDLL('ntdll', winmode=0x800)
HANDLE = C.c_void_p
ULONG = C.c_uint32
SIZE = C.c_size_t
STATUS = C.c_int32

class Entry(C.Structure):
    _fields_ = [('handle', HANDLE), ('handle_count', SIZE), ('pointer_count', SIZE),
                ('access', ULONG), ('type_index', ULONG), ('attributes', ULONG), ('reserved', ULONG)]

class Header(C.Structure):
    _fields_ = [('count', SIZE), ('reserved', SIZE)]

class Unicode(C.Structure):
    _fields_ = [('length', C.c_uint16), ('maximum', C.c_uint16), ('pointer', C.c_void_p)]

assert (C.sizeof(Entry), C.sizeof(Header), C.sizeof(Unicode)) == (40,16,16)
K.OpenProcess.argtypes = [ULONG, W.BOOL, ULONG]
K.OpenProcess.restype = HANDLE
K.GetCurrentProcess.argtypes = []
K.GetCurrentProcess.restype = HANDLE
K.CloseHandle.argtypes = [HANDLE]
K.CloseHandle.restype = W.BOOL
K.DuplicateHandle.argtypes = [HANDLE,HANDLE,HANDLE,C.POINTER(HANDLE),ULONG,W.BOOL,ULONG]
K.DuplicateHandle.restype = W.BOOL
K.GetFileType.argtypes = [HANDLE]
K.GetFileType.restype = ULONG
N.NtQueryInformationProcess.argtypes = [HANDLE,ULONG,C.c_void_p,ULONG,C.POINTER(ULONG)]
N.NtQueryInformationProcess.restype = STATUS
N.NtQueryObject.argtypes = [HANDLE,ULONG,C.c_void_p,ULONG,C.POINTER(ULONG)]
N.NtQueryObject.restype = STATUS
RESIZE = {0xC0000004,0xC0000023,0x80000005}

def query_variable(function, handle, info_class, *, cap):
    size = 65536 if info_class == 51 else 4096
    for _ in range(10):
        if size > cap:
            raise RuntimeError('native_buffer_limit')
        buffer = C.create_string_buffer(size)
        returned = ULONG()
        status = int(function(handle, info_class, buffer, size, C.byref(returned)))
        if status == 0:
            if not 0 < returned.value <= size:
                raise RuntimeError('native_buffer_length_invalid')
            return buffer, returned.value
        if status & 0xffffffff not in RESIZE:
            raise RuntimeError('native_query_failed')
        size = max(size*2, int(returned.value)+4096)
    raise RuntimeError('native_query_unstable')

def handles(process):
    buffer, valid = query_variable(N.NtQueryInformationProcess, process, 51, cap=16*1024*1024)
    if valid < C.sizeof(Header):
        raise RuntimeError('native_header_truncated')
    count = Header.from_buffer(buffer).count
    if not 0 <= count <= 100000 or C.sizeof(Header)+count*C.sizeof(Entry) > valid:
        raise RuntimeError('native_entries_truncated')
    # Copy numbers; no pointers into the result buffer survive this function.
    return [(int(item.handle or 0),int(item.type_index)) for item in
            (Entry*count).from_buffer(buffer,C.sizeof(Header))]

def file_type_index():
    # Identify the kernel File object type using a known local script handle.
    # Only this script's metadata is queried; its contents are not read here.
    with open(__file__,'rb') as stream:
        handle = msvcrt.get_osfhandle(stream.fileno())
        if K.GetFileType(handle) != 1:
            raise RuntimeError('probe_handle_not_disk')
        matches = [kind for value,kind in handles(K.GetCurrentProcess()) if value == handle]
        if len(matches) != 1 or not 0 < matches[0] <= 65535:
            raise RuntimeError('file_type_unavailable')
        return matches[0]

def file_name(handle):
    buffer, valid = query_variable(N.NtQueryObject,handle,1,cap=1024*1024)
    if valid < C.sizeof(Unicode):
        raise RuntimeError('filename_header_truncated')
    value = Unicode.from_buffer(buffer)
    if value.length == 0:
        return ''
    address = int(value.pointer or 0)
    if value.length % 2 or value.length > value.maximum or address < C.addressof(buffer)+C.sizeof(Unicode) or address+value.length > C.addressof(buffer)+valid:
        raise RuntimeError('filename_pointer_invalid')
    return C.wstring_at(address,value.length//2)

def _scan_once(pid, created):
    before = psutil.Process(pid)
    if before.create_time() != created:
        raise RuntimeError('process_incarnation_changed')
    executable = before.exe()
    index = file_type_index()
    process = K.OpenProcess(0x0400 | 0x0040,False,pid)
    if not process:
        raise C.WinError(C.get_last_error())
    result = set()
    counts = {'all_handles':0,'file_handles':0,'disk_handles':0,'excluded_by_isfile_predicate':0}
    try:
        entries = handles(process)
        counts['all_handles'] = len(entries)
        for value, kind in entries:
            if kind != index:
                continue
            counts['file_handles'] += 1
            duplicate = HANDLE()
            # Options 2 = SAME_ACCESS; CLOSE_SOURCE is never set. No inheritable handles.
            if not K.DuplicateHandle(process,value,K.GetCurrentProcess(),C.byref(duplicate),0,False,2):
                raise RuntimeError('file_handle_disappeared')
            try:
                C.set_last_error(0)
                filetype = K.GetFileType(duplicate)
                if filetype == 0 and C.get_last_error():
                    raise RuntimeError('file_type_unavailable')
                if filetype != 1:
                    continue
                counts['disk_handles'] += 1
                native = file_name(duplicate)
                if native:
                    path = convert_dos_path(native)
                    if not path or path.startswith('\\Device\\'):
                        raise RuntimeError('file_path_unmapped')
                    # Match the installed psutil 7.2.2 file-vs-directory predicate.
                    # Some valid handles cannot query FileBasicInformation; do
                    # not silently omit them as a workaround. This metadata-only
                    # path query remains contained by the parent process timeout.
                    if isfile_strict(path):
                        result.add(path)
                    else:
                        counts['excluded_by_isfile_predicate'] += 1
            finally:
                K.CloseHandle(duplicate)
        # A previously non-File handle may be reused for a new account file.
        # Refuse changed File membership instead of silently skipping it.
        file_handles = {(value,kind) for value,kind in entries if kind == index}
        final_file_handles = {(value,kind) for value,kind in handles(process) if kind == index}
        if file_handles != final_file_handles:
            raise RuntimeError('file_handle_set_changed')
    finally:
        K.CloseHandle(process)
    after = psutil.Process(pid)
    if after.create_time() != created or after.exe().casefold() != executable.casefold():
        raise RuntimeError('process_incarnation_changed')
    return sorted(result), counts

def _account_paths(names):
    result=set()
    for name in names:
        item=PureWindowsPath(name)
        if not item.is_absolute() or '..' in item.parts:
            raise RuntimeError('file_path_unmapped')
        parts=tuple(part.casefold() for part in item.parts)
        if ('db_storage' in parts or (len(parts)>=3 and parts[-3]=='login'
                and parts[-1] in {'key_info.db','key_info.db-wal','key_info.db-shm'})):
            result.add(parts)
    return result

def scan(pid, created):
    # Handle values can be reused without changing the final handle set. Two
    # fresh complete scans must agree on ALL account paths, including unknown
    # roots; a cached prior scan or a known-root-only comparison grants nothing.
    before,first_counts=_scan_once(pid,created)
    after,second_counts=_scan_once(pid,created)
    if _account_paths(before) != _account_paths(after):
        raise RuntimeError('account_file_set_changed')
    return after, {'first':first_counts,'second':second_counts}

def response_for(pid, created):
    start = time.perf_counter()
    try:
        if psutil.Process(pid).name().casefold() not in {'wechat.exe','weixin.exe'}:
            raise RuntimeError('not_wechat')
        names, counts = scan(pid,created)
        response = dict(ok=True,names=names,counts=counts,scan_ms=round((time.perf_counter()-start)*1000,3))
    except Exception as exc:
        # No partial evidence or exception text/path leaves the worker.
        known = {'native_buffer_limit','native_buffer_length_invalid','native_query_failed','native_query_unstable',
            'native_header_truncated','native_entries_truncated','probe_handle_not_disk','file_type_unavailable',
            'file_metadata_unavailable','filename_header_truncated','filename_pointer_invalid',
            'process_incarnation_changed','file_handle_disappeared','file_path_unmapped','not_wechat',
            'file_handle_set_changed','account_file_set_changed'}
        response = dict(ok=False,error_type=type(exc).__name__,error_code=str(exc) if str(exc) in known else 'unclassified')
    return response

def serve():
    # Requests are bounded; one complete response is emitted only after a full
    # fresh scan. No file evidence from an earlier request is kept or reused.
    while True:
        line=sys.stdin.readline(1025)
        if not line:
            return
        if len(line)>1024 or not line.endswith('\n'):
            return
        value=json.loads(line)
        if (not isinstance(value,dict) or set(value) != {'sequence','pid','created','nonce'}
                or type(value['sequence']) is not int or value['sequence'] <= 0
                or type(value['pid']) is not int or not 0 < value['pid'] < 2**32
                or type(value['created']) not in {float,int} or not math.isfinite(value['created']) or value['created']<=0
                or not isinstance(value['nonce'],str) or len(value['nonce'])!=32
                or any(char not in '0123456789abcdef' for char in value['nonce'])):
            return
        response=response_for(value['pid'],float(value['created']))
        response['sequence']=value['sequence']
        response['nonce']=value['nonce']
        response['pid']=value['pid']
        response['created']=value['created']
        print(json.dumps(response,ensure_ascii=True,separators=(',',':')),flush=True)

if __name__ == '__main__':
    if sys.argv[1:] == ['--serve']:
        serve()
    else:
        print(json.dumps(response_for(int(sys.argv[1]),float(sys.argv[2])),ensure_ascii=True),flush=True)
