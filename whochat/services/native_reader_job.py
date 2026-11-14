"""An unnamed, non-inherited Windows Job owns only the file-reader child.

The OS terminates that child if the parent exits, even while the child is stuck
in a native file-name query. No WeChat process is ever assigned to this job.
"""
from __future__ import annotations
import ctypes as C
from ctypes import wintypes as W
import os

class OwnedReaderJob:
    def __init__(self):
        if os.name != 'nt' or C.sizeof(C.c_void_p) != 8:
            raise RuntimeError('native_reader_platform_unsupported')

        class Basic(C.Structure):
            _fields_ = [('process_time', C.c_int64), ('job_time', C.c_int64),
                        ('flags', W.DWORD), ('min_working_set', C.c_size_t),
                        ('max_working_set', C.c_size_t), ('active_processes', W.DWORD),
                        ('affinity', C.c_size_t), ('priority', W.DWORD), ('scheduling', W.DWORD)]

        class Extended(C.Structure):
            _fields_ = [('basic', Basic), ('io', C.c_uint64 * 6),
                        ('process_memory', C.c_size_t), ('job_memory', C.c_size_t),
                        ('peak_process_memory', C.c_size_t), ('peak_job_memory', C.c_size_t)]

        if C.sizeof(Extended) != 144:
            raise RuntimeError('native_reader_job_layout_invalid')
        self._api = C.WinDLL('kernel32', use_last_error=True, winmode=0x800)
        self._api.CreateJobObjectW.argtypes = [C.c_void_p, W.LPCWSTR]
        self._api.CreateJobObjectW.restype = W.HANDLE
        self._api.SetInformationJobObject.argtypes = [W.HANDLE, W.INT, C.c_void_p, W.DWORD]
        self._api.SetInformationJobObject.restype = W.BOOL
        self._api.AssignProcessToJobObject.argtypes = [W.HANDLE, W.HANDLE]
        self._api.AssignProcessToJobObject.restype = W.BOOL
        self._api.CloseHandle.argtypes = [W.HANDLE]
        self._api.CloseHandle.restype = W.BOOL
        self._handle = self._api.CreateJobObjectW(None, None)
        if not self._handle:
            raise RuntimeError('native_reader_job_create_failed')
        limits = Extended()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self._api.SetInformationJobObject(self._handle, 9, C.byref(limits), C.sizeof(limits)):
            self.close()
            raise RuntimeError('native_reader_job_limit_failed')

    def assign(self, process):
        # Popen's owned process HANDLE avoids reopening a possibly recycled PID.
        if not self._handle or process.poll() is not None:
            raise RuntimeError('native_reader_child_not_live')
        handle = getattr(process, '_handle', None)
        if handle is None or not self._api.AssignProcessToJobObject(self._handle, int(handle)):
            raise RuntimeError('native_reader_job_assign_failed')

    def close(self):
        if self._handle:
            if not self._api.CloseHandle(self._handle):
                raise RuntimeError('native_reader_job_close_failed')
            self._handle = None
