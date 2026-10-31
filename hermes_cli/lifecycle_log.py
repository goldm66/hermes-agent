"""CLI lifecycle witness — one JSONL line per start / signal / exit.

WHY this exists: a TUI has twice vanished back to the shell with no trace at
all — no traceback in agent.log, no CrashDumps entry, no ``Interrupt requested``
record, and the gateway PID unchanged. Every candidate (scheduled-task scripts,
hidden PowerShell launchers, console-focus theft) had to be ruled out by hand,
and the one thing that would have answered it instantly — "what did this
process see right before it died?" — was never written down.

So: record it before it happens again.

* ``start``  — pid, ppid, argv, cwd, tty-ness, console session, python version
* ``signal`` — SIGINT / SIGTERM / SIGBREAK, plus the Win32 console control
  events that Python never turns into a signal (CTRL_CLOSE / CTRL_LOGOFF /
  CTRL_SHUTDOWN): a window close and a ``taskkill`` look identical from the
  inside, and this is what tells them apart.
* ``exit``   — atexit path, uptime, foreground window owner at that moment

A ``start`` with no matching ``exit`` **is** the answer for a hard kill:
``taskkill /F`` runs no atexit handler. Also arms ``faulthandler`` so a native
crash (0xC0000005 in a C extension) leaves a traceback instead of silence.

Never raises into the caller: a diagnostic that can break startup is worse than
no diagnostic. Writes are append + line-buffered, so a killed process keeps the
``start`` line.
"""
from __future__ import annotations

import atexit
import json
import os
import sys
import time
import traceback

_INSTALLED = False
_START_TS = 0.0
_LOG_NAME = "cli-lifecycle.log"
_fault_file = None          # keep the handle alive for faulthandler


def _log_path() -> str:
    home = os.environ.get("HERMES_HOME") or os.path.join(os.path.expanduser("~"), ".hermes")
    logs = os.path.join(home, "logs")
    try:
        os.makedirs(logs, exist_ok=True)
    except Exception:
        logs = home
    return os.path.join(logs, _LOG_NAME)


def _write(record: dict) -> None:
    try:
        record.setdefault("ts", time.strftime("%Y-%m-%d %H:%M:%S"))
        with open(_log_path(), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _win_foreground() -> dict:
    """Foreground window + owning pid right now (Windows only).

    Tells "the user's terminal was in front when we died" from "something else
    had grabbed the foreground" — the two look the same in a log otherwise.
    """
    if os.name != "nt":
        return {}
    try:
        import ctypes
        u = ctypes.windll.user32
        hwnd = u.GetForegroundWindow()
        if not hwnd:
            return {}
        pid = ctypes.c_ulong(0)
        u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        title = ctypes.create_unicode_buffer(256)
        u.GetWindowTextW(hwnd, title, 256)
        cls = ctypes.create_unicode_buffer(256)
        u.GetClassNameW(hwnd, cls, 256)
        return {"fg_hwnd": int(hwnd), "fg_pid": int(pid.value),
                "fg_title": title.value, "fg_class": cls.value}
    except Exception:
        return {}


def _console_session() -> int | None:
    """Windows session id (0 = services, 1 = the interactive desktop)."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        return int(ctypes.windll.kernel32.WTSGetActiveConsoleSessionId())
    except Exception:
        return None


def _proc_session_id() -> int | None:
    """Our own session id — differs from the console session for Session-0 runs."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        pid = os.getpid()
        sid = ctypes.c_ulong(0)
        if ctypes.windll.kernel32.ProcessIdToSessionId(pid, ctypes.byref(sid)):
            return int(sid.value)
    except Exception:
        pass
    return None


def _install_console_ctrl_handler() -> None:
    """Catch CTRL_CLOSE/LOGOFF/SHUTDOWN, which CPython never maps to a signal.

    Return False so the event keeps propagating (default processing still runs,
    including Python's own SIGINT delivery). Handler order on Windows is LIFO,
    so this runs *before* CPython's — which is exactly what we want: the record
    is written while the process is still alive.
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        HANDLER = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
        names = {0: "CTRL_C", 1: "CTRL_BREAK", 2: "CTRL_CLOSE",
                 5: "CTRL_LOGOFF", 6: "CTRL_SHUTDOWN"}

        def _cb(event: int) -> bool:
            try:
                _write({"ev": "console_ctrl", "pid": os.getpid(),
                        "event": names.get(int(event), str(event)),
                        "uptime_s": round(time.time() - _START_TS, 2)})
            except Exception:
                pass
            return False        # do not swallow: let the normal path continue

        handler = HANDLER(_cb)
        ctypes.windll.kernel32.SetConsoleCtrlHandler(handler, True)
        globals()["_console_handler"] = handler   # keep the callback alive
    except Exception:
        pass


def _chain_signal(signum: int) -> None:
    """Record the signal, then hand control back to whoever handled it before."""
    try:
        _write({"ev": "signal", "pid": os.getpid(), "signo": int(signum),
                "signal": str(getattr(__import__("signal"), "Signals", lambda x: x)(signum)),
                "uptime_s": round(time.time() - _START_TS, 2),
                **_win_foreground()})
    except Exception:
        pass
    prev = _PREV_HANDLERS.get(signum)
    try:
        import signal as _sig
        if callable(prev) and prev not in (_sig.SIG_DFL, _sig.SIG_IGN, None):
            prev(signum, None)
            return
    except Exception:
        pass
    # Default disposition: SIGINT -> KeyboardInterrupt (so the TUI still honours
    # Ctrl+C exactly as before), SIGTERM/SIGBREAK -> terminate the process.
    if signum == getattr(__import__("signal"), "SIGINT", 2):
        raise KeyboardInterrupt
    os._exit(128 + int(signum))


_PREV_HANDLERS: dict[int, object] = {}


def _install_signals() -> None:
    try:
        import signal
    except Exception:
        return
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        signum = getattr(signal, name, None)
        if signum is None:
            continue
        try:
            _PREV_HANDLERS[signum] = signal.getsignal(signum)
            signal.signal(signum, lambda s, f: _chain_signal(s))
        except Exception:
            pass


def _on_exit() -> None:
    _write({"ev": "exit", "pid": os.getpid(),
            "uptime_s": round(time.time() - _START_TS, 2),
            **_win_foreground()})


def install() -> None:
    """Arm the witness. Idempotent; swallows every error."""
    global _INSTALLED, _START_TS, _fault_file
    if _INSTALLED:
        return
    _INSTALLED = True
    _START_TS = time.time()
    try:
        argv = list(sys.argv)
    except Exception:
        argv = []
    if os.environ.get("HERMES_LIFECYCLE_LOG", "1") == "0":
        return
    try:
        _write({"ev": "start", "pid": os.getpid(), "ppid": os.getppid(),
                "argv": argv[:12], "cwd": os.getcwd(),
                "stdin_tty": bool(getattr(sys.stdin, "isatty", lambda: False)()),
                "python": sys.version.split()[0],
                "console_session": _console_session(),
                "proc_session": _proc_session_id()})
    except Exception:
        pass
    try:
        import faulthandler
        crash = _log_path().replace(".log", "-faulthandler.log")
        _fault_file = open(crash, "a", encoding="utf-8", buffering=1)
        faulthandler.enable(file=_fault_file, all_threads=True)
    except Exception:
        pass
    atexit.register(_on_exit)
    _install_signals()
    _install_console_ctrl_handler()
    # A traceback escaping to the very top is still a lifecycle event worth
    # having next to the start line (agent.log may be a different process).
    try:
        sys.excepthook = _make_excepthook(sys.excepthook)
    except Exception:
        pass


def _make_excepthook(prev):
    def _hook(exc_type, exc, tb):
        try:
            _write({"ev": "uncaught", "pid": os.getpid(),
                    "type": getattr(exc_type, "__name__", str(exc_type)),
                    "msg": str(exc)[:500],
                    "where": "".join(traceback.format_tb(tb)[-3:])[-800:]})
        except Exception:
            pass
        return prev(exc_type, exc, tb)
    return _hook
