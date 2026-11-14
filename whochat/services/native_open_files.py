"""Bounded, fresh account-file evidence from an owned Windows child process.

No identity/file-list result is cached. A timed-out worker and its private queue
are retired before another request can start. Job ownership also covers abrupt
parent exit. Errors cannot return a partial list or fall back to old evidence.
"""
from __future__ import annotations
from pathlib import Path, PureWindowsPath
import json
import math
import queue
import secrets
import subprocess
import sys
import threading
import time
from whochat.services.native_reader_job import OwnedReaderJob

MAX_RESPONSE = 4 * 1024 * 1024
MAX_FRESH_SCAN_ATTEMPTS = 4
RETRYABLE_ERRORS = frozenset({'file_handle_disappeared', 'file_handle_set_changed', 'account_file_set_changed'})
WORKER_ERROR_CODES = RETRYABLE_ERRORS | frozenset({
    'native_buffer_limit', 'native_buffer_length_invalid', 'native_query_failed', 'native_query_unstable',
    'native_header_truncated', 'native_entries_truncated', 'probe_handle_not_disk', 'file_type_unavailable',
    'file_metadata_unavailable', 'filename_header_truncated', 'filename_pointer_invalid',
    'process_incarnation_changed', 'file_path_unmapped', 'not_wechat', 'unclassified',
})
BROKER_ERROR_CODES = frozenset({
    'native_reader_invalid_process', 'native_reader_closed_or_busy', 'native_reader_closed',
    'native_reader_timeout', 'native_reader_start_failed', 'native_reader_worker_ended',
    'native_reader_evidence_unavailable', 'native_reader_negative_payload_invalid',
    'native_reader_file_list_invalid', 'native_reader_file_path_invalid',
    'native_reader_closed_or_expired', 'native_reader_response_invalid',
    'native_reader_io_failed', 'native_reader_stop_unconfirmed', 'native_reader_retry_complete',
})

class ReaderUnavailable(RuntimeError):
    pass

class NativeFileReader:
    def __init__(self, *, worker=None, timeout=1.5, diagnostics=None):
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 5:
            raise ValueError('native_reader_timeout_invalid')
        self.worker = Path(worker or Path(__file__).with_name('native_open_files_worker.py')).resolve()
        self.timeout = float(timeout)
        self._lock = threading.RLock()
        self._process = self._job = self._queue = self._thread = None
        self._sequence = 0
        self._closed = False
        self._broken = False
        self._diagnostics = diagnostics

    def _failure_diagnostic(self, code, attempt, started, result, *, winerror=None):
        """Numeric, allowlisted observation; diagnostic failure grants nothing."""
        try:
            code = code if code in WORKER_ERROR_CODES | BROKER_ERROR_CODES else 'unclassified'
            event = dict(code=code, attempt=attempt if type(attempt) is int and 0 <= attempt <= MAX_FRESH_SCAN_ATTEMPTS else 0,
                         elapsed_ms=max(0, min(2**31-1, round((time.monotonic()-started)*1000))),
                         result=result if result in {'retrying', 'failed', 'retry_succeeded', 'stop_failed'} else 'failed')
            if type(winerror) is int and 0 <= winerror <= 2**32-1:
                event['winerror'] = winerror
            if self._diagnostics is not None:
                self._diagnostics(event)
            else:
                from whochat.diagnostics import append_diagnostics_log
                append_diagnostics_log('native_account', json.dumps(event, separators=(',', ':')))
        except Exception:
            pass

    def _start(self):
        if self._closed or self._broken:
            raise ReaderUnavailable('native_reader_closed')
        if self._process is not None and self._process.poll() is None:
            return
        self._stop()
        try:
            self._job = OwnedReaderJob()
            self._process = subprocess.Popen(
                [sys.executable, '-I', '-X', 'utf8', '-B', str(self.worker), '--serve'],
                cwd=self.worker.parent, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW,
                close_fds=True, bufsize=-1,
            )
            self._job.assign(self._process)
            process, messages = self._process, queue.Queue(maxsize=1)

            def receive():
                try:
                    while True:
                        line = process.stdout.readline(MAX_RESPONSE + 1)
                        if not line or len(line) > MAX_RESPONSE or not line.endswith(b'\n'):
                            messages.put_nowait(None)
                            return
                        messages.put_nowait(line)
                except Exception:
                    try:
                        messages.put_nowait(None)
                    except queue.Full:
                        pass

            self._queue = messages
            self._thread = threading.Thread(target=receive, daemon=True, name='whochat-native-file-response')
            self._thread.start()
        except Exception as exc:
            self._stop()
            raise ReaderUnavailable('native_reader_start_failed') from exc

    def _stop(self):
        process, thread, job = self._process, self._thread, self._job
        # Retain every reference until stopping is confirmed. Failed stop is
        # terminal for this reader, so a new helper can never leak alongside it.
        try:
            if job is not None:
                job.close()
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=1)
                if thread is not None:
                    thread.join(timeout=1)
                    if thread.is_alive():
                        raise ReaderUnavailable('native_reader_thread_stop_unconfirmed')
                if process.stdin is not None:
                    process.stdin.close()
                if process.stdout is not None:
                    process.stdout.close()
        except Exception as exc:
            self._broken = True
            raise ReaderUnavailable('native_reader_stop_unconfirmed') from exc
        self._process = self._job = self._queue = self._thread = None

    def read(self, pid, created):
        started = time.monotonic()
        if (type(pid) is not int or not 0 < pid < 2**32
                or type(created) not in {int, float} or not math.isfinite(created) or created <= 0):
            self._failure_diagnostic('native_reader_invalid_process', 0, started, 'failed')
            raise ReaderUnavailable('native_reader_invalid_process')
        deadline = started + self.timeout
        if self._closed or self._broken or not self._lock.acquire(timeout=self.timeout):
            self._failure_diagnostic('native_reader_closed_or_busy', 0, started, 'failed')
            raise ReaderUnavailable('native_reader_closed_or_busy')
        try:
            if self._closed or self._broken:
                self._failure_diagnostic('native_reader_closed', 0, started, 'failed')
                raise ReaderUnavailable('native_reader_closed')
            if time.monotonic() >= deadline:
                self._failure_diagnostic('native_reader_timeout', 0, started, 'failed')
                raise ReaderUnavailable('native_reader_timeout')
            try:
                self._start()
            except Exception as exc:
                self._failure_diagnostic('native_reader_start_failed', 1, started, 'failed',
                                         winerror=getattr(exc, 'winerror', getattr(exc.__cause__, 'winerror', None)))
                if isinstance(exc, ReaderUnavailable) and exc.args == ('native_reader_stop_unconfirmed',):
                    self._failure_diagnostic('native_reader_stop_unconfirmed', 1, started, 'stop_failed',
                                             winerror=getattr(exc.__cause__, 'winerror', None))
                raise
            attempt, failure_code, failure_winerror = 1, None, None
            try:
                while True:
                    # Only the explicit, bound negative below can reach another
                    # RPC. It rereads everything; no first-attempt names survive.
                    if self._closed or time.monotonic() >= deadline:
                        raise ReaderUnavailable('native_reader_closed_or_expired')
                    self._sequence += 1
                    request = dict(sequence=self._sequence, pid=pid, created=created, nonce=secrets.token_hex(16))
                    payload = (json.dumps(request, separators=(',', ':')) + '\n').encode('utf-8')
                    self._process.stdin.write(payload)
                    self._process.stdin.flush()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ReaderUnavailable('native_reader_timeout')
                    line = self._queue.get(timeout=remaining)
                    if line is None:
                        raise ReaderUnavailable('native_reader_worker_ended')
                    result = json.loads(line)
                    if (not isinstance(result, dict) or type(result.get('sequence')) is not int
                            or result['sequence'] != self._sequence
                            or type(result.get('nonce')) is not str or result['nonce'] != request['nonce']
                            or type(result.get('pid')) is not int or result['pid'] != pid
                            or type(result.get('created')) not in {int,float} or result['created'] != created
                            or type(result.get('ok')) is not bool):
                        raise ReaderUnavailable('native_reader_evidence_unavailable')
                    if not result['ok']:
                        if (set(result) - {'sequence', 'nonce', 'pid', 'created', 'ok', 'error_type', 'error_code', 'winerror'}
                                or type(result.get('error_type')) is not str or type(result.get('error_code')) is not str
                                or ('winerror' in result and (type(result['winerror']) is not int or not 0 <= result['winerror'] <= 2**32-1))):
                            raise ReaderUnavailable('native_reader_negative_payload_invalid')
                        failure_code = result['error_code'] if result['error_code'] in WORKER_ERROR_CODES else 'unclassified'
                        failure_winerror = result.get('winerror')
                        if (attempt < MAX_FRESH_SCAN_ATTEMPTS and failure_code in RETRYABLE_ERRORS and result['error_type'] == 'RuntimeError'
                                and not self._closed and time.monotonic() < deadline):
                            self._failure_diagnostic(failure_code, attempt, started, 'retrying', winerror=failure_winerror)
                            # Transient handle churn can outlast two scans. A
                            # retry gets a new nonce and two complete native
                            # scans, but never a fresh deadline or old names.
                            attempt, failure_code, failure_winerror = attempt + 1, None, None
                            continue
                        raise ReaderUnavailable('native_reader_evidence_unavailable')
                    names = result.get('names')
                    if not isinstance(names, list) or len(names) > 50000:
                        raise ReaderUnavailable('native_reader_file_list_invalid')
                    for name in names:
                        if not isinstance(name, str) or not 0 < len(name) <= 32768 or '\x00' in name:
                            raise ReaderUnavailable('native_reader_file_path_invalid')
                        path = PureWindowsPath(name)
                        if not path.is_absolute() or '..' in path.parts or path.drive.startswith('\\\\?'):
                            raise ReaderUnavailable('native_reader_file_path_invalid')
                    if self._closed or time.monotonic() >= deadline:
                        raise ReaderUnavailable('native_reader_closed_or_expired')
                    if attempt > 1:
                        self._failure_diagnostic('native_reader_retry_complete', attempt, started, 'retry_succeeded')
                        # Diagnostics are observational; even a slow callback
                        # cannot let expired or closed evidence escape.
                        if self._closed or time.monotonic() >= deadline:
                            raise ReaderUnavailable('native_reader_closed_or_expired')
                    return tuple(names)
            except Exception as exc:
                if failure_code is None:
                    code = exc.args[0] if isinstance(exc, ReaderUnavailable) and exc.args else None
                    failure_code = (code if type(code) is str and code in BROKER_ERROR_CODES else
                                    'native_reader_timeout' if isinstance(exc, queue.Empty) else
                                    'native_reader_response_invalid' if isinstance(exc, (ValueError, TypeError)) else
                                    'native_reader_io_failed' if isinstance(exc, OSError) else 'unclassified')
                    failure_winerror = getattr(exc, 'winerror', getattr(exc.__cause__, 'winerror', None))
                self._failure_diagnostic(failure_code, attempt, started, 'failed', winerror=failure_winerror)
                try:
                    self._stop()
                except Exception as stop_exc:
                    self._failure_diagnostic('native_reader_stop_unconfirmed', attempt, started, 'stop_failed',
                                             winerror=getattr(stop_exc, 'winerror', getattr(stop_exc.__cause__, 'winerror', None)))
                    raise
                raise ReaderUnavailable('native_reader_fresh_scan_failed') from exc
        finally:
            self._lock.release()

    def close(self):
        self._closed = True
        with self._lock:
            self._stop()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
