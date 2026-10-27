"""Where captured data goes while it arrives.

`ChannelSink` splits the device's interleaved 64-bit words into one array per channel
as the data arrives, so a capture never needs a second full-size copy.  Small captures
live in RAM; large ones (e.g. a 60 s stream at 250 MHz x 8 channels = 15 GB) go to
memory-mapped files in a spool folder.
"""
from __future__ import annotations

import atexit
import gc
import os
import shutil
import sys
import tempfile
import weakref
from pathlib import Path

import numpy as np

_spool_dirs: list[str] = []
_owners: list = []          # weakrefs (Capture is an unhashable dataclass)


def available_memory() -> int:
    """Free physical memory in bytes (best effort; 0 if unknown)."""
    try:
        if sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(st)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return int(st.ullAvailPhys)
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


class ChannelSink:
    """Receives the raw stream and keeps one uint64 array (64 samples/word) per channel."""

    def __init__(self, n_channels: int, words_per_channel: int, spool_dir=None):
        self.n = n_channels
        self.row_bytes = 8 * n_channels
        self.capacity = words_per_channel
        self.rows = 0
        self._carry = b""
        self.dir = None
        if spool_dir is not None:
            Path(spool_dir).mkdir(parents=True, exist_ok=True)
            self.dir = tempfile.mkdtemp(prefix="dslogic_spool_", dir=spool_dir)
            _spool_dirs.append(self.dir)
            self.cols = [np.memmap(os.path.join(self.dir, f"ch{k}.u64"), dtype="<u8", mode="w+",
                                   shape=(max(1, words_per_channel),))
                         for k in range(n_channels)]
        else:
            self.cols = [np.zeros(max(1, words_per_channel), dtype="<u8") for _ in range(n_channels)]

    @property
    def bytes_written(self) -> int:
        return self.rows * self.row_bytes

    def write(self, data) -> None:
        if self._carry:
            data = self._carry + bytes(data)
            self._carry = b""
        rows = min(len(data) // self.row_bytes, self.capacity - self.rows)
        if rows > 0:
            w = np.frombuffer(data, dtype="<u8", count=rows * self.n).reshape(rows, self.n)
            r = self.rows
            for k in range(self.n):
                self.cols[k][r:r + rows] = w[:, k]
            self.rows += rows
        rest = len(data) - rows * self.row_bytes
        if rest and self.rows < self.capacity:
            self._carry = bytes(data[rows * self.row_bytes:])

    def channel_bytes(self, k: int, nbytes: int) -> np.ndarray:
        """Packed bits of the k-th enabled channel (a view, no copy)."""
        return self.cols[k].view(np.uint8)[:nbytes]


def register_owner(obj) -> None:
    """Objects holding spooled memmaps; released at exit so the files can be deleted."""
    _owners.append(weakref.ref(obj))


@atexit.register
def _cleanup() -> None:
    for ref in _owners:
        o = ref()
        if o is None:
            continue
        try:
            o.release()
        except Exception:
            pass
    gc.collect()
    for d in _spool_dirs:
        shutil.rmtree(d, ignore_errors=True)
