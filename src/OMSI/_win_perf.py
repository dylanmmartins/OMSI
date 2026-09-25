# -*- coding: utf-8 -*-
"""
src/OMSI/_win_perf.py

Keep Windows from throttling OMSI and its worker processes while inference runs.

DMM, September 2026
"""

import contextlib
import os
import sys
import threading
import warnings

_IS_WINDOWS = sys.platform == 'win32'
_POLL_S = 2.0

_depth = 0
_depth_lock = threading.Lock()

if _IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

    _PROCESS_POWER_THROTTLING = 4
    _THROTTLING_VERSION = 1
    _THROTTLE_EXECUTION_SPEED = 0x1
    _THROTTLE_IGNORE_TIMER_RES = 0x4
    _PROCESS_SET_INFORMATION = 0x0200
    _TH32CS_SNAPPROCESS = 0x2
    _ES_CONTINUOUS = 0x80000000
    _ES_SYSTEM_REQUIRED = 0x00000001
    _INVALID_HANDLE = ctypes.c_void_p(-1).value

    class _ThrottlingState(ctypes.Structure):
        _fields_ = [('Version', wintypes.ULONG),
                    ('ControlMask', wintypes.ULONG),
                    ('StateMask', wintypes.ULONG)]

    class _ProcessEntry(ctypes.Structure):
        _fields_ = [('dwSize', wintypes.DWORD),
                    ('cntUsage', wintypes.DWORD),
                    ('th32ProcessID', wintypes.DWORD),
                    ('th32DefaultHeapID', ctypes.c_size_t),
                    ('th32ModuleID', wintypes.DWORD),
                    ('cntThreads', wintypes.DWORD),
                    ('th32ParentProcessID', wintypes.DWORD),
                    ('pcPriClassBase', wintypes.LONG),
                    ('dwFlags', wintypes.DWORD),
                    ('szExeFile', wintypes.WCHAR * 260)]

    _kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.SetProcessInformation.restype = wintypes.BOOL
    _kernel32.SetProcessInformation.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    _kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    _kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    _kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessEntry)]
    _kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessEntry)]
    _kernel32.SetThreadExecutionState.restype = wintypes.DWORD
    _kernel32.SetThreadExecutionState.argtypes = [wintypes.DWORD]


def _set_throttling(handle, disable):

    mask = _THROTTLE_EXECUTION_SPEED | _THROTTLE_IGNORE_TIMER_RES
    state = _ThrottlingState(_THROTTLING_VERSION, mask if disable else 0, 0)
    return bool(_kernel32.SetProcessInformation(
        handle, _PROCESS_POWER_THROTTLING, ctypes.byref(state), ctypes.sizeof(state)))


def _descendants(root_pid):

    snap = _kernel32.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
    if not snap or snap == _INVALID_HANDLE:
        return set()

    children = {}
    try:
        entry = _ProcessEntry()
        entry.dwSize = ctypes.sizeof(entry)
        ok = _kernel32.Process32FirstW(snap, ctypes.byref(entry))
        while ok:
            children.setdefault(entry.th32ParentProcessID, []).append(entry.th32ProcessID)
            ok = _kernel32.Process32NextW(snap, ctypes.byref(entry))
    finally:
        _kernel32.CloseHandle(snap)

    found, stack = set(), [root_pid]
    while stack:
        for pid in children.get(stack.pop(), []):
            if pid not in found and pid != root_pid:
                found.add(pid)
                stack.append(pid)
    return found


def _unthrottle_pid(pid):

    handle = _kernel32.OpenProcess(_PROCESS_SET_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        return _set_throttling(handle, disable=True)
    finally:
        _kernel32.CloseHandle(handle)


@contextlib.contextmanager
def no_power_throttling(verbose=False):

    global _depth

    if not _IS_WINDOWS:
        yield
        return

    with _depth_lock:
        _depth += 1
        outermost = _depth == 1

    if not outermost:
        try:
            yield
        finally:
            with _depth_lock:
                _depth -= 1
        return

    me = _kernel32.GetCurrentProcess()
    if _set_throttling(me, disable=True):
        if verbose:
            print('[OMSI] Windows power throttling disabled for this process and its children.')
    else:
        warnings.warn('Could not disable Windows power throttling (error {}); '
                      'timings may be slower.'.format(ctypes.get_last_error()))
    _kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)

    stop = threading.Event()
    done = set()

    def _watch():
        root = os.getpid()
        while True:
            for pid in _descendants(root) - done:
                _unthrottle_pid(pid)
                done.add(pid)
            if stop.wait(_POLL_S):
                return

    watcher = threading.Thread(target=_watch, name='omsi_win_perf', daemon=True)
    watcher.start()
    try:
        yield
    finally:
        stop.set()
        watcher.join(timeout=_POLL_S + 1.0)
        _set_throttling(me, disable=False)
        _kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
        with _depth_lock:
            _depth -= 1

