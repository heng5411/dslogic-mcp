"""Multi-level overview of a capture, for fast zoomed-out drawing.

For every channel it keeps min/max of the packed bytes at three resolutions:

    level 0: per 512 B   =  4 Ki samples
    level 1: per 32 KiB  = 256 Ki samples
    level 2: per 2 MiB   =  16 Mi samples  (one .dsl block)

A pixel that spans >= 4 Ki samples is drawn from the coarsest level that still has at least
one entry per pixel, so any zoom level reads at most ~2 entries per pixel per channel. Raw
samples are only read for views below 4 Ki samples per pixel (< 1 MB per channel).

The overview is built in the background on all cores, block-major across channels, so a
viewer can show it filling in from left to right (`valid_blocks` says what is ready). Built
overviews of .dsl files are cached on disk (keyed by path, size and mtime) so reopening a
file shows its overview immediately.
"""
from __future__ import annotations

import hashlib
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

UNITS = (512, 32 * 1024, 2 * 1024 * 1024)          # bytes per entry, per level
BLOCK = UNITS[-1]                                  # build granularity = one .dsl block
CACHE_VERSION = 3          # 3: edges on the first sample of an idle stretch were missed
DENSE_UNITS = 0.25          # blocks with more active 4 Ki-sample stretches than this: no edge index
MAX_EDGES = 64_000_000      # per channel; beyond that the channel is not indexed


def cache_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "dslogic" / "overview"


class Overview:
    def __init__(self, cap):
        self.cap = cap
        self.nbytes = (cap.num_samples + 7) // 8
        self.nblocks = max(1, -(-self.nbytes // BLOCK))
        self.mn: dict[int, list[np.ndarray]] = {}
        self.mx: dict[int, list[np.ndarray]] = {}
        self.valid: dict[int, np.ndarray] = {}      # per channel, per block
        # edge index: exact edge positions per block (None = too dense to index)
        self.blk_edges: dict[int, list] = {}
        self.blk_ends: dict[int, np.ndarray] = {}   # (first bit, last bit) of each block
        self._edges_all: dict[int, np.ndarray | None] = {}
        self.first_level: dict[int, int] = {}
        self._lock = threading.Lock()
        self._pool: ThreadPoolExecutor | None = None
        self.cancelled = False
        self.done_jobs = 0
        self.total_jobs = 0

    # --- storage -------------------------------------------------------------
    def _alloc(self, ch: int) -> None:
        if ch in self.mn:
            return
        self.mn[ch] = [np.full(-(-self.nbytes // u), 255, np.uint8) for u in UNITS]
        self.mx[ch] = [np.zeros(-(-self.nbytes // u), np.uint8) for u in UNITS]
        self.valid[ch] = np.zeros(self.nblocks, bool)
        self.blk_edges[ch] = [None] * self.nblocks
        self.blk_ends[ch] = np.zeros((self.nblocks, 2), np.uint8)

    def complete(self, ch: int) -> bool:
        return ch in self.valid and bool(self.valid[ch].all())

    @property
    def progress(self) -> float:
        return self.done_jobs / self.total_jobs if self.total_jobs else 1.0

    # --- building ------------------------------------------------------------
    def _read_block(self, ch: int, b: int) -> np.ndarray:
        packed = self.cap.data[ch]
        read = getattr(packed, "read_block", None)     # lazy .dsl: bypass its LRU cache
        if read is not None:
            return read(b)
        return np.asarray(packed[b * BLOCK:min(self.nbytes, (b + 1) * BLOCK)])

    def _build_block(self, ch: int, b: int) -> None:
        if self.cancelled or self.valid[ch][b]:
            return
        seg = self._read_block(ch, b)
        n = len(seg)
        u0 = UNITS[0]
        full = n // u0 * u0
        o0 = b * (BLOCK // u0)
        mn0, mx0 = self.mn[ch][0], self.mx[ch][0]
        if full:
            r = seg[:full].reshape(-1, u0)
            mn0[o0:o0 + full // u0] = r.min(axis=1)
            mx0[o0:o0 + full // u0] = r.max(axis=1)
        if n > full:                                    # partial last entry
            mn0[o0 + full // u0] = seg[full:].min()
            mx0[o0 + full // u0] = seg[full:].max()
        k0 = -(-n // u0)                                # level-0 entries in this block
        for lvl in (1, 2):                              # coarser levels from level 0
            f = UNITS[lvl] // u0
            o = b * (BLOCK // UNITS[lvl])
            m = -(-k0 // f)
            src_mn = mn0[o0:o0 + k0]
            src_mx = mx0[o0:o0 + k0]
            idx = np.arange(0, k0, f)
            self.mn[ch][lvl][o:o + m] = np.minimum.reduceat(src_mn, idx)
            self.mx[ch][lvl][o:o + m] = np.maximum.reduceat(src_mx, idx)
        self._index_edges(ch, b, seg, mn0[o0:o0 + k0], mx0[o0:o0 + k0])
        self.valid[ch][b] = True
        with self._lock:
            self.done_jobs += 1

    def _index_edges(self, ch, b, seg, mn, mx) -> None:
        """Record the exact edges inside block b (not across its start; see edges_of)."""
        from .capture import _edges_span
        n = len(seg)
        if n == 0:
            return
        self.blk_ends[ch][b] = (seg[0] & 1, (seg[-1] >> ((self._block_samples(b) - 1) & 7)) & 1)
        const = (mn == mx) & ((mn == 0) | (mn == 255))
        keep = ~const
        if len(keep) > 1:
            keep[1:] |= const[1:] & const[:-1] & (mn[1:] != mn[:-1])
        if len(keep) >= 64 and keep.mean() > DENSE_UNITS:   # (tiny tail blocks are always indexed)
            self.blk_edges[ch][b] = None                  # too busy to index cheaply
            return
        us = UNITS[0] * 8
        nsamp = self._block_samples(b)
        idx = np.flatnonzero(keep)
        parts = []
        if len(idx):
            breaks = np.flatnonzero(np.diff(idx) > 1)
            for a, e in zip(np.r_[idx[0], idx[breaks + 1]], np.r_[idx[breaks], idx[-1]] + 1):
                # +1: also the first sample after the run, where the line may settle into
                # the idle stretch that follows (that edge belongs to no active unit)
                s0, s1 = max(1, int(a) * us), min(nsamp, int(e) * us + 1)
                if s1 > s0:
                    parts.append(_edges_span(seg, s0, s1, "both"))
        base = b * BLOCK * 8
        e = np.concatenate(parts) + base if parts else np.zeros(0, np.int64)
        self.blk_edges[ch][b] = e

    def _block_samples(self, b: int) -> int:
        return min(BLOCK * 8, self.cap.num_samples - b * BLOCK * 8)

    def edges_of(self, ch: int) -> np.ndarray | None:
        """All edges of the channel (sorted), or None if not (yet) fully indexed."""
        if ch in self._edges_all:
            return self._edges_all[ch]
        if not self.complete(ch):
            return None
        blocks = self.blk_edges[ch]
        if any(e is None for e in blocks):
            self._edges_all[ch] = None
            return None
        ends = self.blk_ends[ch]
        parts = []
        for b, e in enumerate(blocks):
            if b and ends[b, 0] != ends[b - 1, 1]:        # level flips exactly at the block start
                parts.append(np.array([b * BLOCK * 8], np.int64))
            parts.append(e)
        out = np.concatenate(parts) if parts else np.zeros(0, np.int64)
        if len(out) > MAX_EDGES:
            out = None
        self.first_level[ch] = int(ends[0, 0])
        self._edges_all[ch] = out
        return out

    def build(self, channels, workers: int | None = None, background: bool = True,
              on_progress=None) -> None:
        """Build the given channels (block-major, so all channels fill in together)."""
        channels = [c for c in channels if not self.complete(c)]
        for c in channels:
            self._alloc(c)
        if not channels:
            return
        if self.load_cache(channels):
            channels = [c for c in channels if not self.complete(c)]
            if not channels:
                return
        jobs = [(c, b) for b in range(self.nblocks) for c in channels if not self.valid[c][b]]
        with self._lock:
            self.total_jobs += len(jobs)
        workers = workers or min(32, os.cpu_count() or 4)

        def run():
            with ThreadPoolExecutor(max_workers=workers) as pool:
                self._pool = pool
                step = max(1, len(jobs) // 50)
                futs = []
                for i, (c, b) in enumerate(jobs):
                    futs.append(pool.submit(self._build_block, c, b))
                for i, f in enumerate(futs):
                    f.result()
                    if on_progress and (i % step == 0):
                        on_progress(self.progress)
            if not self.cancelled:
                self.save_cache()
            if on_progress:
                on_progress(1.0)

        if background:
            threading.Thread(target=run, daemon=True, name="dslogic-overview").start()
        else:
            run()

    def cancel(self) -> None:
        self.cancelled = True

    # --- queries -------------------------------------------------------------------
    def level_for(self, spp: float) -> int | None:
        """Coarsest level with at least one entry per pixel, or None for raw samples."""
        best = None
        for lvl, u in enumerate(UNITS):
            if u * 8 <= spp:
                best = lvl
        return best

    def buckets(self, ch: int, lvl: int, s0: int, s1: int, width: int):
        """(x_starts, lo, hi) with lo/hi = NaN where the overview is not built yet."""
        us = UNITS[lvl] * 8
        mn, mx = self.mn[ch][lvl], self.mx[ch][lvl]
        u0, u1 = s0 // us, min(len(mn), -(-s1 // us))
        u1 = max(u1, u0 + 1)
        b = np.unique(np.linspace(u0, u1, width + 1).astype(np.int64)[:-1]) - u0
        smn = np.minimum.reduceat(mn[u0:u1], b)
        smx = np.maximum.reduceat(mx[u0:u1], b)
        units_per_block = BLOCK // UNITS[lvl]
        ok = self.valid[ch][(np.arange(u0, u1) // units_per_block)]
        okb = np.maximum.reduceat(ok.astype(np.uint8), b).astype(bool)
        lo = np.where(okb, (smn == 255), np.nan)
        hi = np.where(okb, (smx != 0), np.nan)
        return (b + u0) * us, lo, hi

    # --- disk cache ------------------------------------------------------------------
    def _cache_file(self) -> Path | None:
        path = self.cap.meta.get("path")
        if not path:
            return None
        try:
            st = os.stat(path)
        except OSError:
            return None
        key = f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}|{self.cap.num_samples}|{CACHE_VERSION}"
        return cache_dir() / (hashlib.sha1(key.encode()).hexdigest() + ".npz")

    def load_cache(self, channels) -> bool:
        f = self._cache_file()
        if f is None or not f.exists():
            return False
        try:
            z = np.load(f)
            for c in channels:
                k = f"ch{c}"
                if k + "_mn" not in z.files:
                    continue
                mn0, mx0 = z[k + "_mn"], z[k + "_mx"]
                if len(mn0) != len(self.mn[c][0]):
                    continue
                self.mn[c][0][:], self.mx[c][0][:] = mn0, mx0
                for lvl in (1, 2):
                    f_ = UNITS[lvl] // UNITS[0]
                    idx = np.arange(0, len(mn0), f_)
                    self.mn[c][lvl][:] = np.minimum.reduceat(mn0, idx)
                    self.mx[c][lvl][:] = np.maximum.reduceat(mx0, idx)
                self.valid[c][:] = True
                if k + "_edges" in z.files:
                    self._edges_all[c] = np.cumsum(z[k + "_edges"]).astype(np.int64)
                    self.first_level[c] = int(z[k + "_first"][0])
                else:
                    self._edges_all[c] = None
            return True
        except Exception:
            return False

    def save_cache(self) -> None:
        f = self._cache_file()
        if f is None:
            return
        done = [c for c in self.mn if self.complete(c)]
        if not done:
            return
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            arrays = {}
            for c in done:
                arrays[f"ch{c}_mn"] = self.mn[c][0]
                arrays[f"ch{c}_mx"] = self.mx[c][0]
                e = self.edges_of(c)
                if e is not None:                          # gaps compress far better than positions
                    arrays[f"ch{c}_edges"] = np.diff(e, prepend=0)
                    arrays[f"ch{c}_first"] = np.array([self.first_level[c]], np.uint8)
            tmp = f.with_name(f.stem + ".part.npz")
            np.savez_compressed(tmp, **arrays)
            os.replace(tmp, f)
        except OSError:
            pass


def get_overview(cap) -> Overview:
    ov = cap.meta.get("_overview")
    if ov is None:
        ov = Overview(cap)
        cap.meta["_overview"] = ov
    return ov
