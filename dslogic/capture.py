"""Captured logic data.

Every channel is kept packed, 1 bit per sample, LSB first (bit k of byte j is sample
8*j+k) -- the same layout as DSView's in-memory blocks and its .dsl files, so saving and
loading needs no conversion.  The hardware sends 64-sample words per enabled channel in
turn ("LA_CROSS_DATA"); `from_cross()` splits that stream.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Channel:
    """Per-channel settings: shown name, colour, enable flag and simple trigger."""
    index: int
    name: str = ""
    color: str | None = None        # "#rrggbb" (DSView signal colour) or None = default
    enabled: bool = True
    trigger: str = "X"              # simple trigger condition, see dslogic.trigger

    def __post_init__(self):
        if not self.name:
            self.name = str(self.index)
        if self.color is not None:
            self.color = normalize_color(self.color)


_NAMED = {"red": "#ff0000", "green": "#00ff00", "blue": "#0000ff", "yellow": "#ffff00",
          "cyan": "#00ffff", "magenta": "#ff00ff", "white": "#ffffff", "black": "#000000",
          "orange": "#ffa500", "purple": "#800080", "gray": "#808080", "grey": "#808080",
          "pink": "#ffc0cb", "brown": "#a52a2a"}


def normalize_color(c) -> str:
    if isinstance(c, (tuple, list)) and len(c) == 3:
        return "#{:02x}{:02x}{:02x}".format(*(int(v) & 0xFF for v in c))
    s = str(c).strip().lower()
    s = _NAMED.get(s, s)
    if not s.startswith("#"):
        s = "#" + s
    if len(s) != 7 or any(ch not in "0123456789abcdef" for ch in s[1:]):
        raise ValueError(f"bad colour {c!r}: use '#rrggbb', (r, g, b) or a basic colour name")
    return s


@dataclass
class Capture:
    samplerate: int
    num_samples: int
    data: dict[int, np.ndarray]                 # channel index -> packed uint8 bits
    channels: dict[int, Channel] = field(default_factory=dict)
    trigger_pos: int = 0                        # sample index of the trigger
    triggered: bool = False
    start_time: float = field(default_factory=time.time)   # epoch seconds
    decoders: list = field(default_factory=list)            # decoder objects (dslogic.decoders)
    device: str = "DSLogic U3Pro32"
    meta: dict = field(default_factory=dict)    # extra info (settings used, overflow, ...)

    def __post_init__(self):
        for i in self.data:
            self.channels.setdefault(i, Channel(i))

    def release(self) -> None:
        """Drop the sample data (closes spool files of a large capture)."""
        self.data = {}

    # --- construction --------------------------------------------------
    @classmethod
    def from_cross(cls, raw: bytes | np.ndarray, enabled: list[int], samplerate: int,
                   num_samples: int | None = None, **kw) -> "Capture":
        """Split the device's interleaved stream of 64-bit words into channels."""
        n_en = len(enabled)
        words = np.frombuffer(raw, dtype="<u8")
        words = words[: (len(words) // n_en) * n_en].reshape(-1, n_en)
        total = words.shape[0] * 64
        num_samples = total if num_samples is None else min(num_samples, total)
        nbytes = (num_samples + 7) // 8
        data = {ch: np.ascontiguousarray(words[:, k]).view(np.uint8)[:nbytes].copy()
                for k, ch in enumerate(enabled)}
        return cls(samplerate=samplerate, num_samples=num_samples, data=data, **kw)

    @classmethod
    def from_arrays(cls, samplerate: int, arrays: dict[int, np.ndarray], **kw) -> "Capture":
        """Build from unpacked 0/1 arrays (e.g. for tests or synthetic data)."""
        n = min(len(a) for a in arrays.values())
        data = {ch: np.packbits(np.asarray(a[:n], dtype=np.uint8) & 1, bitorder="little")
                for ch, a in arrays.items()}
        return cls(samplerate=samplerate, num_samples=n, data=data, **kw)

    # --- access ----------------------------------------------------------
    @property
    def enabled(self) -> list[int]:
        return sorted(self.data)

    @property
    def duration(self) -> float:
        return self.num_samples / self.samplerate

    def index(self, ch: int | str) -> int:
        """Channel index from an index or a channel name."""
        if isinstance(ch, (int, np.integer)):
            if int(ch) not in self.data:
                raise KeyError(f"channel {ch} not in capture (have {self.enabled})")
            return int(ch)
        for i, c in self.channels.items():
            if c.name == ch and i in self.data:
                return i
        if str(ch).isdigit() and int(ch) in self.data:
            return int(ch)
        raise KeyError(f"no channel named {ch!r}")

    def __getitem__(self, ch: int | str) -> np.ndarray:
        return self.samples(ch)

    def samples(self, ch: int | str, start: int = 0, stop: int | None = None) -> np.ndarray:
        """Unpacked samples (uint8 0/1) of one channel, optionally a [start, stop) window."""
        stop = self.num_samples if stop is None else min(stop, self.num_samples)
        start = max(0, start)
        packed = self.data[self.index(ch)]
        b0, b1 = start // 8, (stop + 7) // 8
        bits = np.unpackbits(packed[b0:b1], bitorder="little")
        return bits[start - b0 * 8: stop - b0 * 8]

    def packed(self, ch: int | str) -> np.ndarray:
        return self.data[self.index(ch)]

    def edges(self, ch: int | str, kind: str = "both", start: int = 0, stop: int | None = None,
              chunk: int = 1 << 24) -> np.ndarray:
        """Sample indices i in [start, stop) where the channel changes (s[i] != s[i-1]);
        kind 'rising', 'falling' or 'both'.

        Fast on huge captures: when the overview (dslogic.overview) covers the range, only
        the stretches it marks as active are read; other ranges are scanned in parallel
        chunks (unpacking and diffing run outside the GIL)."""
        ch = self.index(ch)
        stop = self.num_samples if stop is None else min(stop, self.num_samples)
        start = max(0, start)
        if stop <= start:
            return np.array([], np.int64)
        ov = self.meta.get("_overview")
        allv = ov.edges_of(ch) if ov is not None and ch in ov.valid else None
        if allv is not None:                            # edge index: no decompression at all
            k0, k1 = np.searchsorted(allv, start), np.searchsorted(allv, stop)
            e = allv[k0:k1]
            if kind == "both":
                return e
            after = (ov.first_level[ch] ^ ((np.arange(k0, k1) + 1) & 1)).astype(bool)
            return e[after] if kind == "rising" else e[~after]
        spans = self._active_spans(ch, start, stop)
        if spans is None:
            spans = [(a, min(stop, a + chunk)) for a in range(start, stop, chunk)]
        packed = self.data[ch]
        if len(spans) > 1 and sum(b - a for a, b in spans) > (1 << 22):
            from concurrent.futures import ThreadPoolExecutor
            import os
            with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 4)) as pool:
                out = list(pool.map(lambda ab: _edges_span(packed, ab[0], ab[1], kind), spans))
        else:
            out = [_edges_span(packed, a, b, kind) for a, b in spans]
        out = [o for o in out if len(o)]
        return np.concatenate(out) if out else np.array([], np.int64)

    def levels_at(self, ch: int | str, idx) -> np.ndarray:
        """Channel level (0/1) at sample indices; from the edge index when available."""
        ch = self.index(ch)
        idx = np.asarray(idx, dtype=np.int64)
        ov = self.meta.get("_overview")
        allv = ov.edges_of(ch) if ov is not None and ch in ov.valid else None
        if allv is not None:
            return (ov.first_level[ch] ^ (np.searchsorted(allv, idx, side="right") & 1)).astype(np.uint8)
        packed = self.data[ch]
        return (packed[idx >> 3] >> (idx & 7)) & 1

    def _active_spans(self, ch: int, start: int, stop: int):
        """Sample ranges that can contain edges, from the overview's 4 Ki-sample min/max
        (a stretch is skipped when it is constant and equal to the stretch before it).
        None when no complete overview covers the range."""
        ov = self.meta.get("_overview")
        if ov is None or ch not in ov.valid:
            return None
        from .overview import BLOCK, UNITS
        us = UNITS[0] * 8
        u0, u1 = start // us, -(-stop // us)
        if not ov.valid[ch][u0 * UNITS[0] // BLOCK: -(-u1 * UNITS[0] // BLOCK)].all():
            return None
        mn, mx = ov.mn[ch][0][u0:u1], ov.mx[ch][0][u0:u1]
        const = (mn == mx) & ((mn == 0) | (mn == 255))
        keep = ~const
        if len(keep) > 1:
            keep[1:] |= const[1:] & const[:-1] & (mn[1:] != mn[:-1])   # level flips at a boundary
        keep[0] = True                                                 # may differ from s[start-1]
        idx = np.flatnonzero(keep)
        if len(idx) == 0:
            return []
        breaks = np.flatnonzero(np.diff(idx) > 1)
        run_start = np.r_[idx[0], idx[breaks + 1]]
        run_end = np.r_[idx[breaks], idx[-1]] + 1
        # each run reaches one sample past its end: an edge may sit on the first sample
        # of the constant stretch that follows
        return [(max(start, int(a + u0) * us), min(stop, int(b + u0) * us + 1))
                for a, b in zip(run_start, run_end)]

    def _edges_around(self, ch: int, s: int, before: int, after: int, max_span: int):
        """Up to `before` edges <= s and `after` edges > s, searching outward in growing
        windows (bounded by max_span samples each way, so idle lines stay fast)."""
        w = 4096
        while True:
            lo, hi = max(0, s - w), min(self.num_samples, s + w + 1)
            e = self.edges(ch, start=lo + 1, stop=hi)
            prev, nxt = e[e <= s], e[e > s]
            done_l = len(prev) >= before or lo == 0 or w >= max_span
            done_r = len(nxt) >= after or hi == self.num_samples or w >= max_span
            if done_l and done_r:
                return prev[-before:] if before else prev[:0], nxt[:after]
            w *= 4

    def next_edge(self, ch: int, s: int, limit: int | None = None) -> int | None:
        """First sample index > s whose level differs from sample s (None if none within
        `limit` samples). Scans packed bytes, 8 samples per compare, stopping early."""
        n = self.num_samples
        stop = n if limit is None else min(n, s + 1 + limit)
        if s + 1 >= stop:
            return None
        packed = self.data[ch]
        lv = int((packed[s >> 3] >> (s & 7)) & 1)
        same = 0xFF if lv else 0x00
        # rest of the first byte
        j = s >> 3
        for k in range(s + 1, min(stop, (j + 1) * 8)):
            if ((int(packed[k >> 3]) >> (k & 7)) & 1) != lv:
                return k
        b, bend = j + 1, (stop + 7) >> 3
        step = 1 << 16
        while b < bend:
            seg = np.asarray(packed[b:min(bend, b + step)])
            hit = np.flatnonzero(seg != same)
            if len(hit):
                bb = b + int(hit[0])
                byte = int(seg[hit[0]]) ^ same
                k = bb * 8 + ((byte & -byte).bit_length() - 1)
                return k if k < stop else None
            b += step
            step = min(step * 2, 1 << 22)
        return None

    def prev_edge(self, ch: int, s: int, limit: int | None = None) -> int | None:
        """Start of the run containing s: the largest index e <= s with level(e) !=
        level(e-1) (None if the run reaches back past `limit` samples or to sample 0)."""
        lo = 0 if limit is None else max(0, s - limit)
        if s <= lo:
            return None
        packed = self.data[ch]
        lv = int((packed[s >> 3] >> (s & 7)) & 1)
        same = 0xFF if lv else 0x00
        j = s >> 3
        for k in range(s - 1, max(lo, j * 8) - 1, -1):
            if ((int(packed[k >> 3]) >> (k & 7)) & 1) != lv:
                return k + 1
        b = j                                   # bytes [blo, b) remain to scan backwards
        blo = lo >> 3
        step = 1 << 16
        while b > blo:
            a0 = max(blo, b - step)
            seg = np.asarray(packed[a0:b])
            hit = np.flatnonzero(seg != same)
            if len(hit):
                bb = a0 + int(hit[-1])
                byte = int(seg[hit[-1]]) ^ same
                k = bb * 8 + byte.bit_length() - 1      # highest differing bit
                return k + 1 if k + 1 > lo else None
            b = a0
            step = min(step * 2, 1 << 22)
        return None

    def pulse_at(self, ch: int | str, sample: int, max_span: int = 1 << 24) -> dict:
        """Measure the pulse containing `sample` on one channel, like DSView's hover:
        level, start/end sample, width, and (when the edges exist) period, frequency
        and duty cycle. Missing values are None (e.g. no edge within max_span)."""
        ch = self.index(ch)
        sample = int(min(max(sample, 0), self.num_samples - 1))
        level = int(self.samples(ch, sample, sample + 1)[0])
        start = self.prev_edge(ch, sample, max_span)
        end = self.next_edge(ch, sample, max_span)
        nxt2 = self.next_edge(ch, end, max_span) if end is not None else None
        r = {"channel": ch, "name": self.channels[ch].name, "level": level,
             "start": start, "end": end,
             "width": None, "period": None, "frequency": None, "duty": None}
        if r["start"] is not None and r["end"] is not None:
            r["width"] = (r["end"] - r["start"]) / self.samplerate
            if nxt2 is not None:
                period = (nxt2 - r["start"]) / self.samplerate
                high = r["width"] if level else period - r["width"]
                r.update(period=period, frequency=1 / period, duty=high / period)
        return r

    def measure(self, ch: int | str, start: int, stop: int) -> dict:
        """Edges, complete periods, mean frequency and duty cycle of one channel between
        samples start and stop (edges exactly at either end count, so a range snapped
        onto two rising edges gives whole periods)."""
        ch = self.index(ch)
        start, stop = max(0, int(start)), min(self.num_samples, int(stop))
        e = self.edges(ch, start=max(start, 1), stop=min(self.num_samples, stop + 1))
        lv = (self.packed(ch)[e >> 3] >> (e & 7)) & 1 if len(e) else np.zeros(0, np.uint8)
        rising = e[lv == 1]
        r = {"channel": ch, "name": self.channels[ch].name, "start": start, "stop": stop,
             "duration": (stop - start) / self.samplerate, "edges": int(len(e)),
             "rising": int(len(rising)), "falling": int(len(e) - len(rising)),
             "periods": max(0, int(len(rising)) - 1),
             "frequency": None, "period": None, "duty": None}
        if len(rising) >= 2:
            period = float((rising[-1] - rising[0]) / (len(rising) - 1) / self.samplerate)
            r.update(period=period, frequency=1 / period)
            seg_s, seg_e = int(rising[0]), int(rising[-1])
            r["duty"] = float(self.samples(ch, seg_s, seg_e).mean()) if seg_e - seg_s <= 1 << 26 else None
        return r

    def time_of(self, sample: int | np.ndarray) -> float | np.ndarray:
        """Seconds relative to the trigger point."""
        return (np.asarray(sample) - self.trigger_pos) / self.samplerate

    def set_name(self, ch: int, name: str) -> None:
        self.channels[self.index(ch)].name = name

    def set_color(self, ch: int | str, color) -> None:
        self.channels[self.index(ch)].color = normalize_color(color)

    def add_decoder(self, decoder) -> None:
        self.decoders.append(decoder)

    def decode(self, decoder=None):
        """Run one decoder (or all attached ones) and return its annotations."""
        if decoder is not None:
            return decoder.decode(self)
        out = {}
        for d in self.decoders:
            key, n = d.label, 2
            while key in out:                      # e.g. two DSView decoders both "0:SPI"
                key, n = f"{d.label} #{n}", n + 1
            out[key] = d.decode(self)
        return out

    def slice(self, start: int, stop: int, channels=None) -> "Capture":
        """New in-memory capture holding samples [start, stop) (optionally a channel subset)."""
        stop = min(stop, self.num_samples)
        keep = [self.index(c) for c in channels] if channels else list(self.data)
        data = {ch: np.packbits(self.samples(ch, start, stop), bitorder="little") for ch in keep}
        return Capture(samplerate=self.samplerate, num_samples=stop - start, data=data,
                       channels={i: Channel(c.index, c.name, c.color, c.enabled, c.trigger)
                                 for i, c in self.channels.items() if i in data},
                       trigger_pos=max(0, self.trigger_pos - start),
                       triggered=self.triggered and start <= self.trigger_pos < stop,
                       start_time=self.start_time, decoders=list(self.decoders),
                       device=self.device, meta=dict(self.meta))

    def trigger_cause(self) -> str:
        """What fired, from the data at the trigger sample, e.g. 'CS falling edge'.
        Uses each channel's simple-trigger condition; empty if not triggered."""
        if not self.triggered:
            return ""
        from .trigger import describe_condition
        tp, parts = self.trigger_pos, []
        for i in self.enabled:
            code = self.channels[i].trigger
            if code in ("X", "", None):
                continue
            name = self.channels[i].name
            if code in ("R", "F", "C") and 0 < tp < self.num_samples:
                s = self.samples(i, max(0, tp - 2), min(self.num_samples, tp + 3))
                d = np.diff(s.astype(np.int8))
                if (d == 1).any():
                    parts.append(f"{name} rising edge")
                elif (d == -1).any():
                    parts.append(f"{name} falling edge")
                else:
                    parts.append(f"{name} {describe_condition(code)} (no edge at the trigger sample)")
            else:
                parts.append(f"{name} {describe_condition(code)}")
        return " AND ".join(parts) or "advanced/serial trigger"

    def summary(self) -> str:
        names = ", ".join(f"{i}:{self.channels[i].name}" for i in self.enabled)
        if self.triggered:
            trig = (f"trigger @ {self.trigger_pos / self.samplerate * 1e3:.6g} ms "
                    f"(sample {self.trigger_pos}): {self.trigger_cause()}")
        else:
            trig = "no trigger"
        return (f"{self.num_samples} samples @ {format_rate(self.samplerate)} "
                f"({format_time(self.duration)}), {len(self.data)} ch [{names}], {trig}")

    def timing_text(self) -> str:
        """Window and timing lines for reports (uses meta['timing'] from DSLogic.capture)."""
        lines = []
        if self.triggered:
            pre = self.trigger_pos / self.samplerate
            post = (self.num_samples - self.trigger_pos) / self.samplerate
            lines.append(f"window  {format_time(self.duration)}: {format_time(pre)} before trigger, "
                         f"{format_time(post)} after")
        else:
            lines.append(f"window  {format_time(self.duration)} from start")
        t = self.meta.get("timing")
        if t:
            if self.meta.get("operation_mode") == "stream":
                lines.append(f"timing  streamed in {t['total_s']:.2f} s")
            else:
                wait = "waited for trigger + captured" if self.triggered else "captured"
                lines.append(f"timing  {wait} in {t['armed_to_data_s']:.2f} s, "
                             f"uploaded in {t['transfer_s']:.2f} s (total {t['total_s']:.2f} s)")
        return "\n".join(lines)

    # --- files -------------------------------------------------------------
    def save(self, path: str | Path, **kw) -> Path:
        """Save by extension: .dsl (DSView), .vcd, .csv, .npz, .sr (sigrok)."""
        from . import export
        return export.save(self, path, **kw)

    @classmethod
    def load(cls, path: str | Path, lazy: bool | None = None) -> "Capture":
        """Load a .dsl (DSView or this library) or .npz file. Large .dsl files open
        lazily (blocks decompressed when first used); lazy=False forces a full load."""
        from . import export
        return export.load(path, lazy=lazy)


def _edges_span(packed, s0: int, s1: int, kind: str) -> np.ndarray:
    """Edges i in [s0, s1) (s[i] != s[i-1], looking one sample before s0)."""
    if s1 <= s0:
        return np.array([], np.int64)
    lo = s0 - 1 if s0 > 0 else 0
    b0, b1 = lo // 8, (s1 + 7) // 8
    seg = np.asarray(packed[b0:b1])
    if len(seg) and seg[0] in (0, 255) and not np.any(seg != seg[0]):
        return np.array([], np.int64)          # constant (incl. the overlap sample)
    bits = np.unpackbits(seg, bitorder="little")[lo - b0 * 8: s1 - b0 * 8]
    d = np.diff(bits.view(np.int8))
    if kind == "rising":
        idx = np.flatnonzero(d == 1)
    elif kind == "falling":
        idx = np.flatnonzero(d == -1)
    else:
        idx = np.flatnonzero(d)
    idx = idx.astype(np.int64) + lo + 1
    return idx[idx >= s0] if s0 == 0 else idx


def format_time(sec: float) -> str:
    """3e-7 -> '300.000 ns', 2e-5 -> '20.000 us', 0.05 -> '50.000 ms', 68.7 -> '68.700 s'."""
    a = abs(sec)
    if a >= 1:
        return f"{sec:.3f} s"
    if a >= 1e-3:
        return f"{sec * 1e3:.3f} ms"
    if a >= 1e-6 or a == 0:
        return f"{sec * 1e6:.3f} us"
    return f"{sec * 1e9:.3f} ns"


def format_rate(hz: float) -> str:
    for unit, div in (("GHz", 1e9), ("MHz", 1e6), ("kHz", 1e3)):
        if hz >= div:
            v = hz / div
            return f"{v:g} {unit}"
    return f"{hz:g} Hz"


def parse_rate(text: str | int | float) -> int:
    """'250M', '250MHz', '1G', '100k', 2e6 -> Hz."""
    if isinstance(text, (int, float)):
        return int(text)
    s = text.strip().lower().replace(" ", "").removesuffix("hz").removesuffix("s/s").removesuffix("sps")
    mult = 1
    for suffix, m in (("g", 1e9), ("m", 1e6), ("k", 1e3)):
        if s.endswith(suffix):
            s, mult = s[:-1], m
            break
    return int(round(float(s) * mult))


def parse_count(text: str | int | float) -> int:
    """'10M', '1k', '64Mi', 1e6 -> samples (decimal unless an 'i' suffix is used)."""
    if isinstance(text, (int, float)):
        return int(text)
    s = text.strip().lower().replace(" ", "").removesuffix("samples").removesuffix("sa")
    for suffix, m in (("gi", 1 << 30), ("mi", 1 << 20), ("ki", 1 << 10),
                      ("g", 10**9), ("m", 10**6), ("k", 10**3)):
        if s.endswith(suffix):
            return int(round(float(s[: -len(suffix)]) * m))
    return int(float(s))


def parse_time(text: str | float) -> float:
    """'10ms', '2s', '500us', '1.5' -> seconds."""
    if isinstance(text, (int, float)):
        return float(text)
    s = text.strip().lower().replace(" ", "")
    for suffix, m in (("ns", 1e-9), ("us", 1e-6), ("µs", 1e-6), ("ms", 1e-3), ("s", 1.0)):
        if s.endswith(suffix):
            return float(s[: -len(suffix)]) * m
    return float(s)
