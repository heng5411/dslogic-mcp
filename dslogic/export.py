"""Save / load captures.

.dsl  DSView session (zip): "header" ini, "decoders" + "session" JSON (channel names,
      colours, decoders), and per channel "L-<ch>/<block>" files of packed bits,
      2 MiB (16 Mi samples) per block -- the layout DSView 1.3.x writes (format v3).
.sr   sigrok/PulseView session (zip, version 2).
.vcd  Value Change Dump (GTKWave, etc.).
.csv  time + one column per channel (only rows where something changes by default).
.npz  numpy archive of the packed data plus metadata (fast, lossless).
"""
from __future__ import annotations

import configparser
import io
import json
import math
import os
import time
import zipfile
from pathlib import Path

import numpy as np

from .capture import Capture, Channel, format_rate

DSL_BLOCK_SAMPLES = 1 << 24          # samples per block, as in DSView
DSL_HEADER_VERSION = 3               # HEADER_FORMAT_VERSION
DSL_SESSION_VERSION = 3              # SESSION_FORMAT_VERSION
SR_CHANNEL_LOGIC = 10000
STRIG = {"X": 0, "R": 1, "1": 2, "F": 3, "0": 4, "C": 5}   # LogicSignal::LogicSetRegions


SAVE_TYPES = (".dsl", ".sr", ".vcd", ".csv", ".npz")


def check_save_path(path) -> Path:
    """Raise ValueError unless `path` has a supported extension (call before capturing)."""
    path = Path(path)
    if path.suffix.lower() not in SAVE_TYPES:
        raise ValueError(f"unknown file type {path.suffix!r} for {path} (use {', '.join(SAVE_TYPES)})")
    return path


def save(cap: Capture, path, **kw) -> Path:
    path = check_save_path(path)
    fn = {".dsl": save_dsl, ".sr": save_sr, ".vcd": save_vcd, ".csv": save_csv,
          ".npz": save_npz}[path.suffix.lower()]
    path.parent.mkdir(parents=True, exist_ok=True)
    fn(cap, path, **kw)
    return path


def load(path, lazy: bool | None = None) -> Capture:
    path = Path(path)
    if path.suffix.lower() == ".npz":
        return load_npz(path)
    if path.suffix.lower() == ".dsl" or zipfile.is_zipfile(path):
        return load_dsl(path, lazy=lazy)
    raise ValueError(f"cannot load {path} (supported: .dsl, .npz)")


# --- DSView .dsl ---------------------------------------------------------------
def _dsl_rate(hz: int) -> str:
    """sr_samplerate_string(): '250 MHz', '1 GHz', '100 kHz', '1.500000 MHz'-style."""
    for div, unit in ((10**9, "GHz"), (10**6, "MHz"), (10**3, "kHz")):
        if hz >= div:
            if hz % div == 0:
                return f"{hz // div} {unit}"
            return f"{hz // div}.{hz % div} {unit}" if unit != "kHz" else f"{hz // div}.{hz % div} KHz"
    return f"{hz} Hz"


def _parse_dsl_rate(s: str) -> int:
    num, _, unit = s.strip().partition(" ")
    mult = {"ghz": 10**9, "mhz": 10**6, "khz": 10**3, "hz": 1, "": 1}[unit.strip().lower()]
    if "." in num:
        whole, frac = num.split(".")
        return int(whole) * mult + int(frac)       # sr_si_string_u64 prints the remainder in Hz
    return int(num) * mult


def dsl_header(cap: Capture) -> str:
    lines = ["[version]", f"version = {DSL_HEADER_VERSION}", "[header]",
             "driver = DSLogic", "device mode = 0", "capturefile = data",
             f"total samples = {cap.num_samples}",
             f"total probes = {len(cap.data)}",
             f"total blocks = {max(1, math.ceil(cap.num_samples / DSL_BLOCK_SAMPLES))}",
             f"samplerate = {_dsl_rate(cap.samplerate)}",
             f"trigger time = {int(cap.start_time * 1000)}",
             f"trigger pos = {cap.trigger_pos}"]
    for i in cap.enabled:
        lines.append(f"probe{i} = {cap.channels[i].name}")
    return "\n".join(lines) + "\n"


def dsl_session(cap: Capture) -> dict:
    chans = []
    for view, i in enumerate(sorted(cap.channels)):
        c = cap.channels[i]
        chans.append({"index": i, "view_index": view, "type": SR_CHANNEL_LOGIC,
                      "enabled": i in cap.data, "name": c.name,
                      "colour": c.color or "default", "strigger": STRIG.get(c.trigger, 0)})
    return {"Version": DSL_SESSION_VERSION, "Device": "DSLogic", "DeviceMode": 0,
            "Title": "dslogic-tool", "channel": chans, "decoder": dsl_decoders(cap)}


def dsl_decoders(cap: Capture) -> list:
    base = len(cap.channels)
    return [d.dsview_json(cap, view_index=base + k) for k, d in enumerate(cap.decoders)]


def save_dsl(cap: Capture, path, compresslevel: int = 6, progress=None,
             workers: int | None = None) -> None:
    """Write a DSView .dsl. Blocks are compressed on all CPU cores (see fastzip).
    progress(fraction) is called after every 2 MiB block (useful for GB captures)."""
    from .fastzip import write_zip
    nblocks = max(1, math.ceil(cap.num_samples / DSL_BLOCK_SAMPLES))
    block_bytes = DSL_BLOCK_SAMPLES // 8
    total_bytes = (cap.num_samples + 7) // 8
    meta = [("header", dsl_header(cap)),
            ("decoders", json.dumps(dsl_decoders(cap), indent=4)),
            ("session", json.dumps(dsl_session(cap), indent=4))]

    def block(ch, b):
        packed = cap.data[ch]
        return lambda: packed[b * block_bytes: min((b + 1) * block_bytes, total_bytes)].tobytes()

    entries = [(name, (lambda t=text: t.encode("utf-8"))) for name, text in meta]
    entries += [(f"L-{ch}/{b}", block(ch, b)) for ch in cap.enabled for b in range(nblocks)]
    total = len(entries)
    path = Path(path)
    tmp = path.with_name(path.name + ".part")
    try:
        write_zip(tmp, entries, level=compresslevel, workers=workers,
                  progress=(lambda n: progress(n / total)) if progress else None)
        os.replace(tmp, path)               # never leave a half-written .dsl behind
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


_thread_files = __import__("threading").local()   # per-thread file handles for _raw_read


class DslChannel:
    """Packed bits of one channel of a .dsl, decompressed block by block on demand.

    Behaves like a 1-D uint8 array for what the library needs: len(), slicing,
    integer and index-array lookups, np.asarray(). Recently used blocks are cached.
    Several threads may read at once (zipfile's shared-file lock + zlib releasing the GIL).
    """
    dtype = np.dtype(np.uint8)
    ndim = 1

    def __init__(self, zf: zipfile.ZipFile, ch: int, nblocks: int, total_bytes: int,
                 cache_blocks: int = 48):
        import threading
        from collections import OrderedDict
        self.zf, self.ch, self.nblocks, self.total = zf, ch, nblocks, total_bytes
        self.block_bytes = DSL_BLOCK_SAMPLES // 8
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._cache_blocks = cache_blocks
        self._lock = threading.Lock()

    def __len__(self):
        return self.total

    @property
    def shape(self):
        return (self.total,)

    def block(self, b: int) -> np.ndarray:
        with self._lock:
            if b in self._cache:
                self._cache.move_to_end(b)
                return self._cache[b]
        name = f"L-{self.ch}/{b}"
        try:
            arr = np.frombuffer(self._raw_read(name), dtype=np.uint8)
        except KeyError:
            arr = np.zeros(0, np.uint8)
        want = min(self.block_bytes, self.total - b * self.block_bytes)
        if len(arr) < want:
            arr = np.concatenate([arr, np.zeros(want - len(arr), np.uint8)])
        arr = arr[:want]
        with self._lock:
            self._cache[b] = arr
            while len(self._cache) > self._cache_blocks:
                self._cache.popitem(last=False)
        return arr

    def _raw_read(self, name: str) -> bytes:
        """Decompress one zip entry with a single zlib call on a per-thread file handle.
        zipfile's reader inflates in small Python-level steps; this way the work runs
        outside the GIL, so many threads really decompress in parallel."""
        import struct
        import zlib
        info = self.zf.getinfo(name)
        tl = _thread_files.__dict__.setdefault("files", {})
        f = tl.get(self.zf.filename)
        if f is None:
            f = open(self.zf.filename, "rb")
            tl[self.zf.filename] = f
        f.seek(info.header_offset)
        hdr = f.read(30)
        if hdr[:4] != b"PK\x03\x04":
            return self.zf.read(name)
        n, e = struct.unpack("<HH", hdr[26:30])
        f.seek(info.header_offset + 30 + n + e)
        raw = f.read(info.compress_size)
        if info.compress_type == zipfile.ZIP_DEFLATED:
            return zlib.decompress(raw, -15)
        if info.compress_type == zipfile.ZIP_STORED:
            return raw
        return self.zf.read(name)

    def read_block(self, b: int) -> np.ndarray:
        """Block b without touching the LRU cache (for one-pass scans)."""
        with self._lock:
            if b in self._cache:
                return self._cache[b]
        try:
            arr = np.frombuffer(self._raw_read(f"L-{self.ch}/{b}"), dtype=np.uint8)
        except KeyError:
            arr = np.zeros(0, np.uint8)
        want = min(self.block_bytes, self.total - b * self.block_bytes)
        if len(arr) < want:
            arr = np.concatenate([arr, np.zeros(want - len(arr), np.uint8)])
        return arr[:want]

    def _range(self, start: int, stop: int) -> np.ndarray:
        start, stop = max(0, start), min(self.total, stop)
        if stop <= start:
            return np.zeros(0, np.uint8)
        bs = self.block_bytes
        b0, b1 = start // bs, (stop - 1) // bs
        if b0 == b1:
            return self.block(b0)[start - b0 * bs: stop - b0 * bs]
        parts = [self.block(b) for b in range(b0, b1 + 1)]
        return np.concatenate(parts)[start - b0 * bs: stop - b0 * bs]

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self.total)
            r = self._range(start, stop)
            return r if step == 1 else r[::step]
        if isinstance(key, (int, np.integer)):
            k = int(key) + (self.total if key < 0 else 0)
            return self.block(k // self.block_bytes)[k % self.block_bytes]
        idx = np.asarray(key, dtype=np.int64)
        out = np.empty(idx.shape, np.uint8)
        if idx.size == 0:
            return out
        blocks = idx // self.block_bytes
        order = np.argsort(blocks, kind="stable")
        sb = blocks[order]
        cuts = np.flatnonzero(np.diff(sb)) + 1
        starts = np.r_[0, cuts]
        uniq = sb[starts]
        if len(uniq) > 2:                       # many blocks: decompress them in parallel, once each
            from concurrent.futures import ThreadPoolExecutor
            import os
            with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 4)) as pool:
                data = list(pool.map(lambda b: self.read_block(int(b)), uniq))
        else:
            data = [self.block(int(b)) for b in uniq]
        ends = np.r_[cuts, len(sb)]
        for blk, arr, a, e in zip(uniq.tolist(), data, starts.tolist(), ends.tolist()):
            sel = order[a:e]
            out[sel] = arr[idx[sel] - blk * self.block_bytes]
        return out

    def __array__(self, dtype=None, copy=None):
        a = self._range(0, self.total)
        return a.astype(dtype) if dtype is not None else a


def load_dsl(path, lazy: bool | None = None) -> Capture:
    """Load a .dsl. lazy=None decides by size: captures above 256 MB uncompressed are
    read block by block on demand (fast to open, low memory) instead of all at once."""
    z = zipfile.ZipFile(path)
    try:
        cap = _load_dsl(z, lazy)
    except BaseException:
        z.close()
        raise
    if not any(isinstance(d, DslChannel) for d in cap.data.values()):
        z.close()
    else:
        cap.meta["zipfile"] = z
    cap.meta["path"] = os.path.abspath(path)
    return cap


def _load_dsl(z: zipfile.ZipFile, lazy: bool | None) -> Capture:
    names = set(z.namelist())
    ini = configparser.ConfigParser(interpolation=None, strict=False)
    ini.optionxform = str
    ini.read_string(z.read("header").decode("utf-8", "replace"))
    h = ini["header"]
    if int(h.get("device mode", "0")) != 0:
        raise ValueError("only logic-analyser (.dsl device mode 0) captures are supported")
    n = int(h["total samples"])
    nblocks = int(h["total blocks"])
    rate = _parse_dsl_rate(h["samplerate"])
    probes = [int(k[5:]) for k in h if k.startswith("probe") and k[5:].isdigit()]
    need = (n + 7) // 8
    if lazy is None:
        lazy = need * len(probes) > 256 << 20
    channels, data = {}, {}
    for i in probes:
        channels[i] = Channel(i, name=h[f"probe{i}"])
        data[i] = DslChannel(z, i, nblocks, need, cache_blocks=12 if lazy else 0)
    if not lazy:                                    # decompress all blocks on all cores
        from concurrent.futures import ThreadPoolExecutor
        full = {i: np.empty(need, np.uint8) for i in probes}

        def fill(job):
            i, b = job
            lc = data[i]
            blk = lc.block(b)
            full[i][b * lc.block_bytes: b * lc.block_bytes + len(blk)] = blk

        jobs = [(i, b) for i in probes for b in range(-(-need // (DSL_BLOCK_SAMPLES // 8)))]
        with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 4)) as pool:
            list(pool.map(fill, jobs))
        data = full
    session = {}
    if "session" in names:
        try:
            session = json.loads(z.read("session").decode("utf-8", "replace"))
        except json.JSONDecodeError:
            session = {}
    decoders = session.get("decoder")
    if decoders is None and "decoders" in names:
        try:
            decoders = json.loads(z.read("decoders").decode("utf-8", "replace"))
        except json.JSONDecodeError:
            decoders = []
    for c in session.get("channel", []):
        i = c.get("index")
        if i in channels:
            if c.get("name"):
                channels[i].name = c["name"]
            col = c.get("colour")
            if col and col != "default" and str(col).startswith("#") and len(col) == 7:
                channels[i].color = col.lower()
            channels[i].trigger = {v: k for k, v in STRIG.items()}.get(int(c.get("strigger", 0)), "X")
    cap = Capture(samplerate=rate, num_samples=n, data=data, channels=channels,
                  trigger_pos=int(h.get("trigger pos", "0")),
                  triggered=int(h.get("trigger pos", "0")) > 0,
                  start_time=int(h.get("trigger time", str(int(time.time() * 1000)))) / 1000.0,
                  device=h.get("driver", "DSLogic"))
    cap.decoders = _decoders_from_json(decoders or [])
    return cap


def _decoders_from_json(arr: list) -> list:
    from .decoders import DSViewDecoder, I2C, SPI, UART
    out = []
    for d in arr:
        chans = {k: v for o in d.get("channel", []) for k, v in o.items()}
        opts = d.get("options", {})
        label = d.get("label", "")
        did = d.get("id", "")
        try:
            if did.endswith(":spi") and "clk" in chans:
                dec = SPI(chans["clk"], chans.get("mosi"), chans.get("miso"), chans.get("cs"), label=label or "SPI")
            elif did.endswith(":i2c"):
                dec = I2C(chans["scl"], chans["sda"], label=label or "I2C")
            elif did.endswith(":uart"):
                dec = UART(chans["rxtx"], label=label or "UART")
            else:
                raise KeyError
            dec.id = did
            dec.options.update(opts)
        except KeyError:
            dec = DSViewDecoder(did, chans, opts, label, d.get("stacked decoders"))
        out.append(dec)
    return out


# --- sigrok .sr ---------------------------------------------------------------
def _interleave(cap: Capture, channels: list[int], start: int, stop: int) -> np.ndarray:
    """Sample-major bytes (unitsize = ceil(n/8)), bit k = k-th listed channel."""
    unit = (len(channels) + 7) // 8
    out = np.zeros((stop - start, unit), np.uint8)
    for k, ch in enumerate(channels):
        out[:, k // 8] |= cap.samples(ch, start, stop) << (k % 8)
    return out


def save_sr(cap: Capture, path, chunk_samples: int = 1 << 22) -> None:
    chans = cap.enabled
    unit = (len(chans) + 7) // 8
    meta = ["[global]", "sigrok version=0.5.2", "", "[device 1]", "capturefile=logic-1",
            f"total probes={len(chans)}", f"samplerate={format_rate(cap.samplerate)}",
            "total analog=0"]
    meta += [f"probe{k + 1}={cap.channels[ch].name}" for k, ch in enumerate(chans)]
    meta.append(f"unitsize={unit}")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("version", "2")
        z.writestr("metadata", "\n".join(meta) + "\n")
        for k, s in enumerate(range(0, cap.num_samples, chunk_samples)):
            e = min(cap.num_samples, s + chunk_samples)
            z.writestr(f"logic-1-{k + 1}", _interleave(cap, chans, s, e).tobytes())


# --- VCD -------------------------------------------------------------------------
def _vcd_timescale(samplerate: int, timescale_ps: int | None = None):
    """(timescale text, sample -> time function). VCD only allows 1, 10 or 100 of a unit,
    so the unit is the coarsest one in which the sample period is a whole number
    (250 MHz: '1 ns', steps of 4); otherwise 1 fs with each time rounded on its own
    (no drift accumulates over a long capture)."""
    from fractions import Fraction
    period_fs = Fraction(10**15, int(samplerate))
    units = [(m, u, m * f) for u, f in (("s", 10**15), ("ms", 10**12), ("us", 10**9), ("ns", 10**6),
                                        ("ps", 10**3), ("fs", 1)) for m in (100, 10, 1)]
    if timescale_ps is not None:                       # caller's choice, if it is a legal one
        units = [x for x in units if x[2] == timescale_ps * 1000] or units[-1:]
    for m, u, fs in units:
        step = period_fs / fs
        if step.denominator == 1:
            k = int(step)
            return f"{m} {u}", lambda i: i * k
    m, u, fs = units[-1]
    return f"{m} {u}", lambda i: int((Fraction(i) * period_fs / fs).__round__())


def save_vcd(cap: Capture, path, timescale_ps: int | None = None) -> None:
    chans = cap.enabled
    scale, at = _vcd_timescale(cap.samplerate, timescale_ps)
    ids = {ch: _vcd_id(k) for k, ch in enumerate(chans)}
    first, ev_idx, ev_ch, ev_val = {}, [], [], []
    for k, ch in enumerate(chans):
        first[ch] = int(cap.samples(ch, 0, 1)[0]) if cap.num_samples else 0
        idx = cap.edges(ch)                         # chunked: fine for huge captures
        ev_idx.append(idx)
        ev_ch.append(np.full(len(idx), k, np.int32))
        ev_val.append(((np.arange(len(idx)) + 1 + first[ch]) & 1).astype(np.uint8))
    idx = np.concatenate(ev_idx) if ev_idx else np.array([], np.int64)
    order = np.argsort(idx, kind="stable")
    idx = idx[order]
    chk = np.concatenate(ev_ch)[order] if ev_ch else idx
    val = np.concatenate(ev_val)[order] if ev_val else idx
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write(f"$date {time.ctime(cap.start_time)} $end\n$version dslogic-tool $end\n")
        f.write(f"$timescale {scale} $end\n$scope module dslogic $end\n")
        for ch in chans:
            name = cap.channels[ch].name.replace(" ", "_")
            f.write(f"$var wire 1 {ids[ch]} {name} $end\n")
        f.write("$upscope $end\n$enddefinitions $end\n#0\n$dumpvars\n")
        for ch in chans:
            f.write(f"{first[ch]}{ids[ch]}\n")
        f.write("$end\n")
        id_list = [ids[ch] for ch in chans]
        last = -1
        out = []
        for i, k, v in zip(idx.tolist(), chk.tolist(), val.tolist()):
            if i != last:
                out.append(f"#{at(i)}\n")
                last = i
            out.append(f"{v}{id_list[k]}\n")
            if len(out) > 100_000:
                f.write("".join(out))
                out.clear()
        f.write("".join(out))
        f.write(f"#{at(cap.num_samples)}\n")


def _vcd_id(k: int) -> str:
    chars = [chr(c) for c in range(33, 127)]
    s = ""
    k += 1
    while k:
        k, r = divmod(k - 1, len(chars))
        s = chars[r] + s
    return s


# --- CSV -------------------------------------------------------------------------
def save_csv(cap: Capture, path, changes_only: bool = True, chunk: int = 1 << 20) -> None:
    """One row per sample where any channel changes (changes_only, the default: built
    from the edges, so multi-GB captures are fine), or one row per sample."""
    chans = cap.enabled
    if changes_only:
        keep = np.unique(np.concatenate([np.zeros(1, np.int64)] + [cap.edges(ch) for ch in chans])) \
            if cap.num_samples else np.zeros(0, np.int64)
    else:
        keep = None
    total = len(keep) if keep is not None else cap.num_samples
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("sample,time_s," + ",".join(cap.channels[ch].name for ch in chans) + "\n")
        for a in range(0, total, chunk):
            idx = keep[a:a + chunk] if keep is not None else np.arange(a, min(total, a + chunk), dtype=np.int64)
            cols = np.stack([cap.levels_at(ch, idx) for ch in chans], axis=1)
            t = (idx - cap.trigger_pos) / cap.samplerate
            buf = io.StringIO()
            for i, ti, row in zip(idx.tolist(), t.tolist(), cols.tolist()):
                buf.write(f"{i},{ti:.12g}," + ",".join(map(str, row)) + "\n")
            f.write(buf.getvalue())


# --- NPZ -------------------------------------------------------------------------
def save_npz(cap: Capture, path) -> None:
    meta = {"samplerate": cap.samplerate, "num_samples": cap.num_samples,
            "trigger_pos": cap.trigger_pos, "triggered": cap.triggered,
            "start_time": cap.start_time, "device": cap.device,
            "channels": {i: {"name": c.name, "color": c.color, "trigger": c.trigger}
                         for i, c in cap.channels.items()},
            "decoders": dsl_decoders(cap)}
    arrays = {f"ch{i}": cap.data[i] for i in cap.enabled}
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)


def load_npz(path) -> Capture:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    data = {int(k[2:]): z[k] for k in z.files if k.startswith("ch")}
    chans = {int(i): Channel(int(i), c["name"], c["color"], int(i) in data, c.get("trigger", "X"))
             for i, c in meta["channels"].items()}
    cap = Capture(samplerate=meta["samplerate"], num_samples=meta["num_samples"], data=data,
                  channels=chans, trigger_pos=meta["trigger_pos"], triggered=meta["triggered"],
                  start_time=meta["start_time"], device=meta["device"])
    cap.decoders = _decoders_from_json(meta.get("decoders", []))
    return cap
