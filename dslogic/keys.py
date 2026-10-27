"""Watch the console for hot keys while a capture runs (Ctrl-X = stop and save).

Ctrl-C is not handled here: it arrives as SIGINT and DSLogic.capture() turns it into a
clean cancel.  Windows reads the console with msvcrt; POSIX puts the terminal in cbreak
mode (signals such as Ctrl-C still work) and restores it afterwards.
"""
from __future__ import annotations

import os
import sys
import threading
from collections.abc import Callable

CTRL_X = "\x18"


class KeyWatcher:
    """`with KeyWatcher({CTRL_X: la.stop}): la.capture()` -- calls the handler when the
    key is pressed. Does nothing when stdin is not an interactive console."""

    def __init__(self, handlers: dict[str, Callable[[], None]]):
        self.handlers = handlers
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        self._restore = None

    def feed(self, ch: str) -> None:
        fn = self.handlers.get(ch)
        if fn:
            fn()

    def __enter__(self):
        if not sys.stdin or not sys.stdin.isatty():
            return self
        if sys.platform == "win32":
            target = self._run_windows
        else:
            import termios
            import tty
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            self._restore = lambda: termios.tcsetattr(fd, termios.TCSADRAIN, old)
            target = self._run_posix
        self._thread = threading.Thread(target=target, name="dslogic-keys", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._done.set()
        if self._thread:
            self._thread.join(timeout=1)
        if self._restore:
            self._restore()
            self._restore = None
        return False

    def _run_windows(self):
        import msvcrt
        while not self._done.wait(0.05):
            while msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("\x00", "\xe0"):           # function/arrow key prefix: skip the code
                    msvcrt.getwch()
                    continue
                self.feed(ch)

    def _run_posix(self):
        import select
        fd = sys.stdin.fileno()
        while not self._done.is_set():
            r, _, _ = select.select([fd], [], [], 0.05)
            if r:
                data = os.read(fd, 32).decode("latin-1", "replace")
                for ch in data:
                    self.feed(ch)
