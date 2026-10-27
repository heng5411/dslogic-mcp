"""Multi-threaded zip writer for .dsl files.

zipfile compresses entries one at a time on one core. Here each entry is deflated in a
worker thread (zlib and crc32 release the GIL, so threads run truly in parallel) and the
main thread writes the results in order. The output is a standard zip (deflate, zip64
when needed), which DSView (minizip unzOpen64), zipfile and unzip all read.
"""
from __future__ import annotations

import os
import struct
import time
import zlib
from collections import deque
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor

_U32 = 0xFFFFFFFF


def _deflate(data: bytes, level: int) -> tuple[bytes, int, int]:
    c = zlib.compressobj(level, zlib.DEFLATED, -15)
    comp = c.compress(data) + c.flush()
    return comp, zlib.crc32(data), len(data)


def _dos_time(t: float) -> tuple[int, int]:
    lt = time.localtime(t)
    dtime = (lt.tm_hour << 11) | (lt.tm_min << 5) | (lt.tm_sec // 2)
    ddate = ((max(lt.tm_year, 1980) - 1980) << 9) | (lt.tm_mon << 5) | lt.tm_mday
    return dtime, ddate


def write_zip(path, entries: Iterable[tuple[str, Callable[[], bytes]]], level: int = 6,
              workers: int | None = None, progress: Callable[[int], None] | None = None,
              _zip64_limit: int = _U32) -> None:
    """Write entries [(name, producer)] in order; producer() returns the entry's bytes.
    progress(n_done) is called after each entry is written."""
    workers = workers or min(32, os.cpu_count() or 4)
    dtime, ddate = _dos_time(time.time())
    central = []
    offset = 0
    done = 0

    def job(producer):
        return _deflate(producer(), level)

    with open(path, "wb") as f, ThreadPoolExecutor(max_workers=workers) as pool:
        pending: deque = deque()

        def flush_one():
            nonlocal offset, done
            name, fut = pending.popleft()
            comp, crc, size = fut.result()
            bname = name.encode("utf-8")
            hdr = struct.pack("<IHHHHHIIIHH", 0x04034B50, 20, 0, 8, dtime, ddate,
                              crc, len(comp), size, len(bname), 0)
            f.write(hdr)
            f.write(bname)
            f.write(comp)
            central.append((bname, crc, len(comp), size, offset))
            offset += len(hdr) + len(bname) + len(comp)
            done += 1
            if progress:
                progress(done)

        try:
            for name, producer in entries:
                pending.append((name, pool.submit(job, producer)))
                if len(pending) >= 2 * workers:      # bounded memory: ~2 blocks per worker
                    flush_one()
            while pending:
                flush_one()
        except BaseException:
            for _, fut in pending:
                fut.cancel()
            raise

        cd_start = offset
        for bname, crc, csize, usize, off in central:
            extra = b""
            off32 = off
            need = 20
            if off >= _zip64_limit:
                extra = struct.pack("<HHQ", 0x0001, 8, off)
                off32, need = _U32, 45
            rec = struct.pack("<IHHHHHHIIIHHHHHII", 0x02014B50, need, need, 0, 8, dtime, ddate,
                              crc, csize, usize, len(bname), len(extra), 0, 0, 0, 0, off32)
            f.write(rec)
            f.write(bname)
            f.write(extra)
            offset += len(rec) + len(bname) + len(extra)
        cd_size = offset - cd_start
        n = len(central)
        if n >= 0xFFFF or cd_start >= _zip64_limit or cd_size >= _zip64_limit:
            z64 = offset
            f.write(struct.pack("<IQHHIIQQQQ", 0x06064B50, 44, 45, 45, 0, 0, n, n, cd_size, cd_start))
            f.write(struct.pack("<IIQI", 0x07064B50, 0, z64, 1))
            f.write(struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, min(n, 0xFFFF), min(n, 0xFFFF),
                                min(cd_size, _U32), min(cd_start, _U32), 0))
        else:
            f.write(struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, n, n, cd_size, cd_start, 0))
