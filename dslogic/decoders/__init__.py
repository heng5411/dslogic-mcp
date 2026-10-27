"""Protocol decoders.

Each decoder does two jobs:
  * `decode(capture)` decodes in Python and returns a list of `Annotation`s;
  * `dsview_json()` describes it for DSView, so a saved .dsl opens in DSView with the
    same decoder already added (DSView ids "0:spi", "1:i2c", "1:uart", ...).

`DSViewDecoder` adds any other DSView decoder (CAN, 1-Wire, I2S, ...) to a saved file;
it cannot decode in Python.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DECODER_CONFIG_VERSION = 2       # DEOCDER_CONFIG_VERSION in DSView storesession.cpp


@dataclass
class Annotation:
    start: int                  # first sample
    end: int                    # last sample
    kind: str                   # e.g. "word", "transfer", "address", "data", "start", "stop"
    data: dict = field(default_factory=dict)
    text: str = ""
    bits: list | None = None    # sample of each bit, in wire order (where it was read), if known

    def __str__(self):
        return f"[{self.start}..{self.end}] {self.kind}: {self.text}"


class Decoder:
    id: str = ""                # DSView decoder id
    label: str = ""
    channels: dict[str, int]    # DSView channel id -> capture channel (index or name)
    options: dict

    def bound(self, cap) -> dict[str, int]:
        return {k: cap.index(v) for k, v in self.channels.items() if v is not None}

    def dsview_json(self, cap=None, view_index: int = -1) -> dict:
        chans = []
        for k, v in self.channels.items():
            if v is None:
                continue
            chans.append({k: cap.index(v) if cap is not None else v})
        return {
            "id": self.id,
            "channel": chans,
            "options": dict(self.options),
            "version": DECODER_CONFIG_VERSION,
            "label": self.label or self.id.split(":")[-1].upper(),
            "stacked decoders": [],
            "view_index": view_index,
            "show": {},
        }

    def decode(self, cap, start: int = 0, stop: int | None = None) -> list[Annotation]:
        raise NotImplementedError(f"{type(self).__name__} only describes a DSView decoder")

    def settings_text(self) -> str:
        """The decoder's settings in words (shown by the viewer and `dslogic decode`)."""
        return ", ".join(f"{k} {v}" for k, v in self.options.items())


class DSViewDecoder(Decoder):
    """Any DSView decoder, saved into the .dsl only.  Example:
    DSViewDecoder("can", {"can_rx": 5}, {"bitrate": 500000}, label="CAN")"""

    def __init__(self, id: str, channels: dict[str, int], options: dict | None = None,
                 label: str = "", stacked: list[dict] | None = None):
        self.id, self.channels, self.options, self.label = id, channels, options or {}, label
        self.stacked = stacked or []

    def dsview_json(self, cap=None, view_index: int = -1) -> dict:
        j = super().dsview_json(cap, view_index)
        j["stacked decoders"] = [{"id": s["id"], "options": s.get("options", {})} for s in self.stacked]
        return j


def _bits_at(cap, ch, idx: np.ndarray) -> np.ndarray:
    """Channel values at sample indices (edge index or packed lookup; no full unpack)."""
    return cap.levels_at(ch, idx)


def _words(bits: np.ndarray, size: int, msb_first: bool) -> np.ndarray:
    n = len(bits) // size
    b = bits[: n * size].reshape(n, size).astype(np.uint64)
    w = (np.uint64(1) << np.arange(size - 1, -1, -1, dtype=np.uint64)) if msb_first \
        else (np.uint64(1) << np.arange(size, dtype=np.uint64))
    return (b * w).sum(axis=1).astype(np.uint64)


def _shift(anns: list[Annotation], offset: int) -> list[Annotation]:
    if offset:
        for a in anns:
            a.start += offset
            a.end += offset
    return anns


class SPI(Decoder):
    """SPI (DSView "0:spi"). Pass channel indices or names; miso/mosi/cs are optional."""
    id = "0:spi"

    def __init__(self, clk, mosi=None, miso=None, cs=None, cpol: int = 0, cpha: int = 0,
                 bitorder: str = "msb-first", wordsize: int = 8, cs_polarity: str = "active-low",
                 label: str = "SPI", cs_glitch: int = 8):
        """cs_glitch: CS gaps shorter than this many samples are merged (probe crosstalk
        glitches inside a frame), and clockless CS pulses this short are ignored."""
        self.cs_glitch = int(cs_glitch)
        self.channels = {"clk": clk, "miso": miso, "mosi": mosi, "cs": cs}
        self.options = {"cs_polarity": cs_polarity, "cpol": int(cpol), "cpha": int(cpha),
                        "bitorder": bitorder, "wordsize": int(wordsize)}
        self.label = label

    @property
    def mode(self) -> int:
        """SPI mode 0..3 (= CPOL * 2 + CPHA)."""
        return self.options["cpol"] * 2 + self.options["cpha"]

    def settings_text(self) -> str:
        o = self.options
        edge = "rising" if o["cpol"] == o["cpha"] else "falling"
        return (f"mode {self.mode}: CPOL {o['cpol']} (clock idles {'high' if o['cpol'] else 'low'}), "
                f"CPHA {o['cpha']} (data read on the {edge} edge), {o['bitorder']}, "
                f"{o['wordsize']}-bit words" + (f", CS {o['cs_polarity']}" if self.channels.get("cs") is not None else ""))

    def _windows(self, cap, b, start=0, stop=None):
        """(start, end) sample ranges where CS is active, built from CS edges (chunked, so
        it works on very long captures). Gaps shorter than cs_glitch are merged."""
        if "cs" not in b:
            return [(start, stop)]
        active = 0 if self.options["cs_polarity"] == "active-low" else 1
        level = int(_bits_at(cap, b["cs"], [start])[0])
        bounds = [start] if level == active else []
        bounds += cap.edges(b["cs"], start=start + 1, stop=stop).tolist()
        if len(bounds) % 2:
            bounds.append(stop)
        merged = []
        for s, e in zip(bounds[0::2], bounds[1::2]):
            if merged and s - merged[-1][1] < self.cs_glitch:
                merged[-1][1] = e                 # inactive gap is a glitch: same frame
            else:
                merged.append([s, e])
        return [(s, e) for s, e in merged]

    def decode(self, cap, start: int = 0, stop: int | None = None) -> list[Annotation]:
        stop = cap.num_samples if stop is None else min(stop, cap.num_samples)
        b = self.bound(cap)
        o = self.options
        size, msb = o["wordsize"], o["bitorder"] == "msb-first"
        sample_rising = o["cpol"] == o["cpha"]
        edges = cap.edges(b["clk"], "rising" if sample_rising else "falling", start, stop)
        out = []
        for ws, we in self._windows(cap, b, start, stop):
            e = edges[(edges >= ws) & (edges < we)]
            if len(e) == 0:
                if "cs" not in b or we - ws < self.cs_glitch:
                    continue                      # short clockless CS pulse: crosstalk
                out.append(Annotation(int(ws), int(we), "transfer", {"bits": 0},
                                      "CS active, no clock edges"))
                continue
            n = len(e) // size
            vals = {}
            for line in ("mosi", "miso"):
                if line in b:
                    vals[line] = _words(_bits_at(cap, b[line], e), size, msb)
            for k in range(n):
                d = {line: int(v[k]) for line, v in vals.items()}
                text = " ".join(f"{line.upper()}={v:0{(size + 3) // 4}X}" for line, v in d.items())
                bits = e[k * size:(k + 1) * size]
                out.append(Annotation(int(bits[0]), int(bits[-1]), "word", d, text, bits=bits.tolist()))
            d = {line: [int(x) for x in v] for line, v in vals.items()}
            text = " | ".join(f"{line.upper()}: " + " ".join(f"{x:02X}" for x in v) for line, v in d.items())
            if len(e) % size:
                text += f" (+{len(e) % size} bits)"
            out.append(Annotation(int(ws), int(we), "transfer", d | {"bits": len(e)}, text))
        return out

    def transfers(self, cap, start: int = 0, stop: int | None = None) -> list[Annotation]:
        return [a for a in self.decode(cap, start, stop) if a.kind == "transfer"]


class I2C(Decoder):
    """I2C (DSView "1:i2c")."""
    id = "1:i2c"

    def __init__(self, scl, sda, address_format: str = "unshifted", label: str = "I2C"):
        self.channels = {"scl": scl, "sda": sda}
        self.options = {"address_format": address_format}
        self.label = label

    def settings_text(self) -> str:
        return f"{self.options['address_format']} 7-bit addresses"

    def decode(self, cap, start: int = 0, stop: int | None = None) -> list[Annotation]:
        return self._decode(cap, start, stop)

    def _decode(self, cap, start, stop) -> list[Annotation]:
        b = self.bound(cap)
        # only the edges are visited: levels just before and at each edge
        stop = cap.num_samples if stop is None else min(stop, cap.num_samples)
        ev = np.union1d(cap.edges(b["scl"], start=start + 1, stop=stop),
                        cap.edges(b["sda"], start=start + 1, stop=stop))
        if len(ev) == 0:
            return []
        c_prev, c_now = _bits_at(cap, b["scl"], ev - 1), _bits_at(cap, b["scl"], ev)
        d_prev, d_now = _bits_at(cap, b["sda"], ev - 1), _bits_at(cap, b["sda"], ev)
        out: list[Annotation] = []
        state, bits, bit_start, nbyte, frame_start = "idle", [], 0, 0, 0
        at: list[int] = []
        for i, c0, c1, d0, d1 in zip(ev.tolist(), c_prev.tolist(), c_now.tolist(),
                                     d_prev.tolist(), d_now.tolist()):
            if c0 == 1 and c1 == 1 and d0 != d1:          # SDA moves while SCL high
                if d1 == 0:
                    out.append(Annotation(i, i, "start" if state == "idle" else "repeated start",
                                          {}, "S" if state == "idle" else "Sr"))
                    state, bits, nbyte, frame_start = "addr", [], 0, i
                else:
                    out.append(Annotation(i, i, "stop", {}, "P"))
                    state, bits = "idle", []
                continue
            if c0 == 0 and c1 == 1 and state != "idle":     # SCL rising: sample
                if not bits:
                    bit_start = i
                    at = []
                bits.append(int(d1))
                at.append(i)
                if len(bits) == 9:
                    byte = int("".join(map(str, bits[:8])), 2)
                    ack = bits[8] == 0
                    if state == "addr":
                        addr, rw = byte >> 1, "read" if byte & 1 else "write"
                        shown = byte if self.options["address_format"] == "shifted" else addr
                        out.append(Annotation(bit_start, i, "address",
                                              {"address": addr, "rw": rw, "ack": ack},
                                              f"{rw[0].upper()} 0x{shown:02X} {'ACK' if ack else 'NACK'}",
                                              bits=at))
                        state = "data"
                    else:
                        out.append(Annotation(bit_start, i, "data",
                                              {"value": byte, "ack": ack, "index": nbyte},
                                              f"0x{byte:02X} {'ACK' if ack else 'NACK'}", bits=at))
                        nbyte += 1
                    bits = []
        return out


class UART(Decoder):
    """UART (DSView "1:uart"). `rx` is the line to decode."""
    id = "1:uart"

    def __init__(self, rx, baudrate: int = 115200, num_data_bits: int = 8, parity_type: str = "none",
                 num_stop_bits: float = 1.0, bit_order: str = "lsb-first", invert: bool = False,
                 format: str = "hex", label: str = "UART"):
        self.channels = {"rxtx": rx}
        self.options = {"baudrate": int(baudrate), "num_data_bits": int(num_data_bits),
                        "parity_type": parity_type, "parity_check": "yes",
                        "num_stop_bits": float(num_stop_bits), "bit_order": bit_order,
                        "format": format, "invert": "yes" if invert else "no", "anno_startstop": "no"}
        self.label = label

    def decode(self, cap, start: int = 0, stop: int | None = None) -> list[Annotation]:
        return self._decode(cap, start, stop)

    def _decode(self, cap, start, stop) -> list[Annotation]:
        b = self.bound(cap)
        o = self.options
        stop = cap.num_samples if stop is None else min(stop, cap.num_samples)
        inv = o["invert"] == "yes"
        spb = cap.samplerate / o["baudrate"]
        if spb < 3:
            raise ValueError(f"sample rate too low for {o['baudrate']} baud ({spb:.1f} samples/bit)")
        nd = o["num_data_bits"]
        npar = 0 if o["parity_type"] == "none" else 1
        # start bits are the line's falling edges (rising on an inverted line)
        falls = cap.edges(b["rxtx"], "rising" if inv else "falling", start + 1, stop)
        n, out, pos = stop, [], start
        frame_len = (1 + nd + npar + o["num_stop_bits"]) * spb
        while True:
            k = np.searchsorted(falls, pos)
            if k >= len(falls):
                break
            s = int(falls[k])
            if s + frame_len > n:
                break
            centers = (s + (np.arange(1 + nd + npar + 1) + 0.5) * spb).astype(np.int64)
            v = _bits_at(cap, b["rxtx"], centers)
            if inv:
                v = 1 - v
            if v[0] != 0:                  # glitch, not a start bit
                pos = s + 1
                continue
            data_bits = v[1:1 + nd]
            if o["bit_order"] == "lsb-first":
                data_bits = data_bits[::-1]
            value = int("".join(map(str, data_bits)), 2)
            info = {"value": value, "framing_error": bool(v[-1] != 1)}
            if npar:
                p = int(v[1 + nd])
                ones = int(data_bits.sum()) + p
                ok = {"odd": ones % 2 == 1, "even": ones % 2 == 0, "zero": p == 0, "one": p == 1}[o["parity_type"]]
                info["parity_error"] = not ok
            text = f"0x{value:02X}" + (f" '{chr(value)}'" if 32 <= value < 127 else "")
            if info["framing_error"]:
                text += " (framing error)"
            if info.get("parity_error"):
                text += " (parity error)"
            end = int(s + frame_len) - 1
            out.append(Annotation(s, end, "data", info, text, bits=centers[1:1 + nd].tolist()))
            pos = int(s + (frame_len - 0.5 * spb))
        return out

    def settings_text(self) -> str:
        o = self.options
        par = {"none": "N", "odd": "O", "even": "E", "zero": "0", "one": "1"}.get(o["parity_type"], "?")
        return (f"{o['baudrate']} baud, {o['num_data_bits']}{par}{o['num_stop_bits']:g}, {o['bit_order']}"
                + (", inverted" if o["invert"] == "yes" else ""))

    def text(self, cap, start: int = 0, stop: int | None = None) -> str:
        return "".join(chr(a.data["value"]) for a in self.decode(cap, start, stop))


__all__ = ["Annotation", "Decoder", "DSViewDecoder", "SPI", "I2C", "UART"]
