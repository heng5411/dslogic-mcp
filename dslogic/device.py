"""DSLogic U3Pro32 / U3Pro16 driver (logic analyser mode).

Port of libsigrok4DSL's DSLogic driver (dsl.c, dslogic.c).  Typical use:

    from dslogic import DSLogic
    with DSLogic.open() as la:
        la.samplerate = "250M"
        la.sample_count = "10M"
        la.enable_channels([0, 1, 2, 3])
        la.set_names({0: "CS", 1: "SCLK", 2: "MISO", 3: "MOSI"})
        la.set_trigger("CS", "falling")
        cap = la.capture()
        cap.save("spi.dsl")
"""
from __future__ import annotations

import json
import logging
import math
import os
import struct
import sys
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import constants as C
from .capture import (Capture, Channel, format_rate, format_time, normalize_color, parse_count,
                      parse_rate, parse_time)
from .setting import ArmParams, build_arm_blocks
from .spool import ChannelSink, available_memory, register_owner
from .trigger import SerialTrigger, Stage, Trigger, cond_code, describe_condition
from .usbio import DeviceNotFound, DSLogicError, UsbTransport

log = logging.getLogger("dslogic")


class CaptureError(DSLogicError):
    pass


class CaptureCancelled(KeyboardInterrupt):
    """capture() was cancelled (Ctrl-C or DSLogic.cancel()); the data was discarded.
    A KeyboardInterrupt subclass, so scripts stop as they would on a plain Ctrl-C."""


class OverflowError_(CaptureError):
    """Stream mode: the USB link could not keep up (lower the rate or channel count)."""


@dataclass
class DeviceInfo:
    index: int
    model: str
    superspeed: bool
    bus_address: str
    pid: int

    def __str__(self):
        return (f"[{self.index}] {self.model} at bus.addr {self.bus_address} "
                f"({'USB 3 SuperSpeed' if self.superspeed else 'USB 2 High-Speed'})")


def _bitstream_search_dirs(extra: str | os.PathLike | None) -> list[Path]:
    dirs = []
    if extra:
        dirs.append(Path(extra))
    if os.environ.get("DSLOGIC_RES"):
        dirs.append(Path(os.environ["DSLOGIC_RES"]))
    dirs.append(Path(__file__).with_name("res"))
    if sys.platform == "win32":
        for base in (os.environ.get("ProgramFiles", r"C:\Program Files"),
                     os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")):
            dirs.append(Path(base) / "DSView" / "res")
    else:
        dirs += [Path("/usr/share/DSView/res"), Path("/usr/local/share/DSView/res"),
                 Path("/mnt/c/Program Files/DSView/res")]
    return dirs


def find_bitstream(name: str, extra_dir=None) -> Path:
    for d in _bitstream_search_dirs(extra_dir):
        p = d / name
        if p.is_file():
            return p
    raise DSLogicError(f"FPGA bitstream {name} not found; pass firmware_dir= or set DSLOGIC_RES "
                       "to DSView's 'res' folder")


class DSLogic:
    """One opened DSLogic analyser.  All settings can be changed while open; they are
    applied at the next `capture()` (the threshold is written to the hardware at once)."""

    def __init__(self, transport, profile: C.Profile, firmware_dir=None):
        self._t = transport
        self.profile = profile
        self._firmware_dir = firmware_dir
        self.fw_version = (0, 0)
        self.hdl_version = None
        n = profile.total_ch_num
        self.channels: list[Channel] = [Channel(i) for i in range(n)]
        self._samplerate = C.MHZ(100)
        self._sample_count = 1_000_000
        self._stream = False
        self._channel_mode: int | None = None      # None = choose automatically
        self._threshold = 1.0
        self.filter = False                       # "1 Sample Clock" glitch filter
        self.external_clock = False
        self.clock_negedge = False                # external clock: sample on falling edge
        self.stop_option = "upload"               # buffer mode: on stop "upload" captured data or "stop"
        self.trigger = Trigger()
        self.decoders: list = []
        # Large captures: "auto" spools to disk when the capture would take more than
        # half the free RAM, "always"/"never" force it. Files go to spool_dir (default:
        # the system temp folder) and are deleted when Python exits.
        self.spool = "auto"
        self.spool_dir: str | None = None
        self._stop = threading.Event()       # stop and keep the data
        self._cancel = threading.Event()     # stop and discard
        self._busy = False
        self.last_status: dict = {}
        # what capture() is doing, for progress displays:
        # "idle", "waiting" (for trigger), "capturing", "uploading" (buffer mode), "done"
        self.capture_phase = "idle"

    # =====================================================================
    # discovery / open / close
    # =====================================================================
    @staticmethod
    def scan() -> list[DeviceInfo]:
        out = []
        for ctx, dev, ss in UsbTransport.list_devices():
            prof = C.find_profile(dev.getProductID(), ss)
            if prof is None:
                continue
            out.append(DeviceInfo(len(out), prof.model, ss,
                                  f"{dev.getBusNumber()}.{dev.getDeviceAddress()}", prof.pid))
        return out

    @classmethod
    def open(cls, index: int = 0, firmware_dir=None, threshold: float | None = None) -> "DSLogic":
        """Open the `index`-th supported analyser, load the FPGA if needed, and set it up."""
        matches = []
        for ctx, dev, ss in UsbTransport.list_devices():
            prof = C.find_profile(dev.getProductID(), ss)
            if prof is not None:
                matches.append((ctx, dev, prof))
        if not matches:
            raise DeviceNotFound(
                "no DSLogic U3Pro32/U3Pro16 found. Check the cable (use a USB 3 port), close "
                "DSView, and on WSL attach it with 'usbipd attach --wsl --busid <id>'.")
        if index >= len(matches):
            raise DeviceNotFound(f"only {len(matches)} analyser(s) found, index {index} requested")
        ctx, dev, prof = matches[index]
        t = UsbTransport.open_device(ctx, dev)
        la = cls(t, prof, firmware_dir)
        if threshold is not None:
            la._threshold = float(threshold)
        try:
            la._bring_up()
        except BaseException:
            t.close()
            raise
        return la

    def _bring_up(self) -> None:
        """dsl_dev_open() + dslogic.c dev_open()."""
        t = self._t
        try:
            manuf, product = t.strings()
        except Exception:
            manuf, product = "DreamSourceLab", ""
        if not manuf.startswith("DreamSourceLab"):
            raise DSLogicError(f"unexpected USB manufacturer string {manuf!r}")
        if product and not product.startswith("USB-based DSL Instrument v2"):
            raise DSLogicError(f"device firmware not running (product string {product!r}); "
                               "start DSView once to initialise it")
        fw = t.ctl_read(C.DSL_CTL_FW_VERSION, 0, 2)
        self.fw_version = (fw[0], fw[1])
        if fw[0] != C.DSL_REQUIRED_VERSION_MAJOR:
            raise DSLogicError(f"firmware {fw[0]}.{fw[1]} not supported (need {C.DSL_REQUIRED_VERSION_MAJOR}.x)"
                               "; update it with a current DSView")
        t.claim()
        hw = t.ctl_read(C.DSL_CTL_HW_STATUS, 0, 1)[0]
        if not hw & C.bmFPGA_DONE:
            self.load_fpga()
        else:
            self.write_reg(C.CTR0_ADDR, C.bmNONE)
            self.hdl_version = self.read_hdl_version()
            if self.hdl_version != C.DSL_HDL_VERSION:
                log.info("FPGA image %#x != %#x, reloading", self.hdl_version, C.DSL_HDL_VERSION)
                self.load_fpga()
        self.write_reg(C.CTR0_ADDR, C.bmNONE)
        self._write_threshold()
        if self.profile.feature_caps & C.CAPS_FEATURE_ADF4360:
            for reg, data, delay in C.ADC_CLK_INIT_500M:
                if delay:
                    time.sleep(delay / 1000)
                for b in data:
                    self.write_reg(reg, b)
        log.info("opened %s, firmware %d.%d", self.profile.model, *self.fw_version)

    def close(self) -> None:
        if self._t is not None:
            self._t.close()
            self._t = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        return f"<DSLogic {self.profile.model} {'SS' if self.profile.superspeed else 'HS'}>"

    # =====================================================================
    # low level
    # =====================================================================
    def write_reg(self, addr: int, value: int) -> None:
        self._t.ctl_write(C.DSL_CTL_I2C_REG, addr, bytes([value & 0xFF]))

    def read_reg(self, addr: int) -> int:
        return self._t.ctl_read(C.DSL_CTL_I2C_STATUS, addr, 1)[0]

    def read_hdl_version(self) -> int:
        return self._t.ctl_read(C.DSL_CTL_I2C_STATUS, 0, C.HDL_VERSION_ADDR + 1)[C.HDL_VERSION_ADDR]

    def hw_status(self) -> int:
        return self._t.ctl_read(C.DSL_CTL_HW_STATUS, 0, 1)[0]

    def _wait_hw(self, bit: int, what: str, timeout: float = 5.0) -> None:
        end = time.monotonic() + timeout
        while not self.hw_status() & bit:
            if time.monotonic() > end:
                raise DSLogicError(f"timeout waiting for {what}")

    def load_fpga(self, path=None) -> None:
        """dsl_fpga_config(): program the FPGA with the bitstream."""
        path = Path(path) if path else find_bitstream(self.profile.fpga_bitstream, self._firmware_dir)
        data = path.read_bytes()
        t = self._t
        log.info("configuring FPGA from %s (%d bytes)", path, len(data))
        t.ctl_write(C.DSL_CTL_PROG_B, 0, bytes([~C.bmWR_PROG_B & 0xFF]))
        t.ctl_write(C.DSL_CTL_LED, 0, bytes([~C.bmLED_GREEN & ~C.bmLED_RED & 0xFF]))
        t.ctl_write(C.DSL_CTL_PROG_B, 0, bytes([C.bmWR_PROG_B]))
        self._wait_hw(C.bmFPGA_INIT_B, "FPGA INIT_B")
        t.ctl_write(C.DSL_CTL_INTRDY, 0, bytes([~C.bmWR_INTRDY & 0xFF]))
        n = len(data)
        t.ctl_write(C.DSL_CTL_BULK_WR, 0, bytes([n & 0xFF, (n >> 8) & 0xFF, (n >> 16) & 0xFF]))
        sent = t.bulk_write(data, timeout_ms=max(1000, n // 50))
        if sent != n:
            raise DSLogicError(f"FPGA configuration: sent {sent} of {n} bytes")
        t.ctl_write(C.DSL_CTL_INTRDY, 0, bytes([C.bmWR_INTRDY]))
        self._wait_hw(C.bmGPIF_DONE, "GPIF done")
        t.ctl_write(C.DSL_CTL_INTRDY, 0, bytes([~C.bmWR_INTRDY & 0xFF]))
        self._wait_hw(C.bmFPGA_DONE, "FPGA done")
        t.ctl_write(C.DSL_CTL_LED, 0, bytes([C.bmLED_GREEN]))
        t.ctl_write(C.DSL_CTL_WORDWIDE, 0, bytes([C.bmWR_WORDWIDE]))
        self.hdl_version = self.read_hdl_version()

    # =====================================================================
    # settings
    # =====================================================================
    @property
    def model(self) -> str:
        return self.profile.model

    @property
    def samplerate(self) -> int:
        return self._samplerate

    @samplerate.setter
    def samplerate(self, value) -> None:
        hz = parse_rate(value)
        if hz not in C.SAMPLERATES_1000:
            raise ValueError(f"{format_rate(hz)} is not a supported sample rate; choose from "
                             + ", ".join(format_rate(r) for r in self.samplerates()))
        self._samplerate = hz

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @sample_count.setter
    def sample_count(self, value) -> None:
        n = parse_count(value)
        if n < 1024:
            raise ValueError("sample count must be at least 1024")
        self._sample_count = n

    @property
    def duration(self) -> float:
        return self._sample_count / self._samplerate

    @duration.setter
    def duration(self, value) -> None:
        """Capture length in seconds (or '10ms'); sets sample_count from the sample rate."""
        self.sample_count = max(1024, math.ceil(parse_time(value) * self._samplerate))

    @property
    def operation_mode(self) -> str:
        return "stream" if self._stream else "buffer"

    @operation_mode.setter
    def operation_mode(self, mode: str) -> None:
        mode = mode.lower()
        if mode not in ("buffer", "stream"):
            raise ValueError("operation_mode must be 'buffer' or 'stream'")
        self._stream = mode == "stream"
        if self._channel_mode is not None and C.CHANNEL_MODES[self._channel_mode].stream != self._stream:
            self._channel_mode = None

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, volts: float) -> None:
        """Input threshold voltage (0.0 - 5.0 V; e.g. 1.65 for 3.3 V logic, 0.9 for 1.8 V)."""
        v = float(volts)
        if not 0.0 <= v <= 5.0:
            raise ValueError("threshold must be between 0.0 and 5.0 V")
        self._threshold = v
        if self._t is not None:
            self._write_threshold()

    def _write_threshold(self) -> None:
        if self.profile.feature_caps & C.CAPS_FEATURE_MAX25_VTH:
            code = int(self._threshold / 3.3 * (1.0 / 2.0) * 255)
        else:
            code = int(self._threshold / 3.3 * (1.5 / 2.5) * 255)
        self.write_reg(C.VTH_ADDR, min(255, code))

    @property
    def channel_mode(self) -> C.ChannelMode:
        """The DSView "channel mode" in use (explicit, or chosen to fit the settings)."""
        if self._channel_mode is not None:
            return C.CHANNEL_MODES[self._channel_mode]
        return self._auto_channel_mode()

    @channel_mode.setter
    def channel_mode(self, mode) -> None:
        if mode is None or mode == "auto":
            self._channel_mode = None
            return
        mid = mode.id if isinstance(mode, C.ChannelMode) else int(mode)
        if mid not in self.profile.channel_modes:
            raise ValueError(f"channel mode {mid} not available; see channel_modes()")
        self._channel_mode = mid
        self._stream = C.CHANNEL_MODES[mid].stream

    def channel_modes(self, stream: bool | None = None) -> list[C.ChannelMode]:
        modes = [C.CHANNEL_MODES[i] for i in self.profile.channel_modes]
        return [m for m in modes if stream is None or m.stream == stream]

    def _auto_channel_mode(self) -> C.ChannelMode:
        en = self.enabled_channels()
        top = max(en) if en else 0
        for m in self.channel_modes(self._stream):
            if m.min_samplerate <= self._samplerate <= m.max_samplerate and top < m.num \
                    and len(en) <= m.vld_num:
                return m
        raise ValueError(self._why_no_mode(en))

    def _why_no_mode(self, en) -> str:
        lines = [f"no {self.operation_mode}-mode channel mode supports {format_rate(self._samplerate)} "
                 f"with channels {en}. Available:"]
        for m in self.channel_modes(self._stream):
            lines.append(f"  {m.descr}: channels 0..{m.num - 1}, at most {m.vld_num} enabled, "
                         f"{format_rate(m.min_samplerate)}..{format_rate(m.max_samplerate)}")
        return "\n".join(lines)

    def samplerates(self, mode: C.ChannelMode | None = None) -> list[int]:
        """Sample rates allowed in `mode` (default: any mode of the current operation mode)."""
        modes = [mode] if mode else self.channel_modes(self._stream)
        lo = min(m.min_samplerate for m in modes)
        hi = max(m.max_samplerate for m in modes)
        return [r for r in C.SAMPLERATES_1000 if lo <= r <= hi]

    def max_samples(self) -> int | None:
        """Buffer mode: per-channel memory depth for the enabled channel count."""
        if self._stream:
            return None
        n = max(1, len(self.enabled_channels()))
        return (self.profile.hw_depth // n) & ~C.SAMPLES_ALIGN

    # --- channels ---------------------------------------------------------
    def channel(self, ch: int | str) -> Channel:
        if isinstance(ch, str) and not ch.isdigit():
            for c in self.channels:
                if c.name == ch:
                    return c
            raise KeyError(f"no channel named {ch!r}")
        i = int(ch)
        if not 0 <= i < len(self.channels):
            raise IndexError(f"channel {i} out of range 0..{len(self.channels) - 1}")
        return self.channels[i]

    def enabled_channels(self) -> list[int]:
        return [c.index for c in self.channels if c.enabled]

    def enable_channels(self, chans: Iterable[int | str], exclusive: bool = True) -> None:
        """Enable the given channels (and, by default, disable all others)."""
        wanted = {self.channel(c).index for c in chans}
        for c in self.channels:
            if c.index in wanted:
                c.enabled = True
            elif exclusive:
                c.enabled = False

    def set_enabled(self, ch: int | str, enabled: bool = True) -> None:
        self.channel(ch).enabled = bool(enabled)

    def disable_channels(self, chans: Iterable[int | str]) -> None:
        for ch in chans:
            self.channel(ch).enabled = False

    def set_name(self, ch: int | str, name: str) -> None:
        self.channel(ch).name = str(name)

    def set_names(self, names: dict[int, str] | list[str]) -> None:
        items = names.items() if isinstance(names, dict) else enumerate(names)
        for ch, name in items:
            self.set_name(ch, name)

    def set_color(self, ch: int | str, color) -> None:
        self.channel(ch).color = normalize_color(color)

    def set_colors(self, colors: dict[int | str, object]) -> None:
        for ch, col in colors.items():
            self.set_color(ch, col)

    # --- trigger ------------------------------------------------------------
    def set_trigger(self, ch: int | str, condition: str) -> None:
        """Simple trigger on one channel: 'rising', 'falling', 'high', 'low', 'edge' or 'X'
        (none).  Conditions on several channels must all hold at once."""
        c = self.channel(ch)
        code = cond_code(condition)
        c.trigger = code
        if code == "X":
            self.trigger.simple.pop(c.index, None)
        else:
            self.trigger.simple[c.index] = code

    def clear_trigger(self) -> None:
        """No trigger: capture starts immediately."""
        pos = self.trigger.position
        self.trigger = Trigger(position=pos)
        for c in self.channels:
            c.trigger = "X"

    @property
    def trigger_position(self) -> int:
        return self.trigger.position

    @trigger_position.setter
    def trigger_position(self, percent: int) -> None:
        """Share of the capture placed before the trigger, 0..100 %."""
        p = int(percent)
        if not 0 <= p <= 100:
            raise ValueError("trigger position is a percentage 0..100")
        self.trigger.position = p

    def set_advanced_trigger(self, stages: list[Stage]) -> None:
        self.trigger.stages = list(stages)
        self.trigger.serial = None

    def set_serial_trigger(self, serial: SerialTrigger) -> None:
        self.trigger.serial = serial

    # --- decoders ---------------------------------------------------------
    def add_decoder(self, decoder) -> None:
        """Attach a decoder (dslogic.decoders.SPI/I2C/UART/DSViewDecoder); it is copied to
        every capture, decoded with `capture.decode()` and saved into .dsl files."""
        self.decoders.append(decoder)

    def remove_decoder(self, decoder_or_label) -> None:
        self.decoders = [d for d in self.decoders
                         if d is not decoder_or_label and d.label != decoder_or_label]

    # --- settings profile -------------------------------------------------
    def settings(self) -> dict:
        from dataclasses import asdict
        d = {
            "samplerate": self._samplerate, "sample_count": self._sample_count,
            "operation_mode": self.operation_mode, "channel_mode": self._channel_mode,
            "threshold": self._threshold, "filter": self.filter,
            "external_clock": self.external_clock, "clock_negedge": self.clock_negedge,
            "stop_option": self.stop_option, "trigger_position": self.trigger.position,
            "channels": [{"index": c.index, "name": c.name, "color": c.color,
                          "enabled": c.enabled, "trigger": c.trigger} for c in self.channels],
            "decoders": [d.dsview_json() for d in self.decoders],
        }
        if self.trigger.stages:
            d["advanced_trigger"] = [asdict(st) for st in self.trigger.stages]
        if self.trigger.serial is not None:
            d["serial_trigger"] = asdict(self.trigger.serial)
        return d

    def apply_settings(self, s: dict) -> list[str]:
        """Apply a settings dict (from settings(), a JSON file or a DSView session).
        Returns warnings about settings that could not be applied as given."""
        from .export import _decoders_from_json
        warn = []
        if "operation_mode" in s:
            self.operation_mode = s["operation_mode"]
        for key in ("samplerate", "sample_count", "threshold", "filter", "external_clock",
                    "clock_negedge", "stop_option"):
            if key in s:
                setattr(self, key, s[key])
        if s.get("channel_mode") is not None:
            mid = int(s["channel_mode"])
            if mid in self.profile.channel_modes:
                self.channel_mode = mid
            else:
                self.channel_mode = None
                warn.append(f"channel mode {mid} is not available on this connection "
                            f"({'USB3' if self.profile.superspeed else 'USB2'}); choosing automatically")
        listed = set()
        if "channels" in s:
            self.clear_trigger()
        for c in s.get("channels", []):
            ch = self.channel(c["index"])
            listed.add(ch.index)
            ch.name, ch.enabled = c.get("name", ch.name), c.get("enabled", ch.enabled)
            ch.color = normalize_color(c["color"]) if c.get("color") else None
            self.set_trigger(ch.index, c.get("trigger", "X"))
        if s.get("channels_exclusive"):
            for ch in self.channels:
                if ch.index not in listed:
                    ch.enabled = False
        if "trigger_position" in s:
            self.trigger_position = s["trigger_position"]
        if s.get("advanced_trigger"):
            self.set_advanced_trigger([Stage(**st) for st in s["advanced_trigger"]])
        if s.get("serial_trigger"):
            self.set_serial_trigger(SerialTrigger(**s["serial_trigger"]))
        if "decoders" in s:
            self.decoders = _decoders_from_json(s["decoders"])
        for w in warn:
            log.warning(w)
        return warn

    def save_settings(self, path) -> None:
        Path(path).write_text(json.dumps(self.settings(), indent=2))

    def load_settings(self, path) -> list[str]:
        """Load settings from this library's JSON, a DSView .dsl capture or a DSView
        settings file -- e.g. to repeat a DSView capture with identical settings."""
        from .dsview_session import is_dsview_session, read_session, settings_from_dsview
        import zipfile
        p = Path(path)
        data = read_session(p) if zipfile.is_zipfile(p) else json.loads(p.read_text(encoding="utf-8"))
        warn = []
        if is_dsview_session(data):
            data, warn = settings_from_dsview(data)
            for w in warn:
                log.warning(w)
        return warn + self.apply_settings(data)

    # =====================================================================
    # capture
    # =====================================================================
    def stop(self) -> None:
        """End a running capture() early and keep what was captured (DSView's Stop).
        Safe to call from another thread or a key handler."""
        self._stop.set()

    def cancel(self) -> None:
        """Abort a running capture(); it raises CaptureCancelled and returns no data."""
        self._cancel.set()

    def read_progress(self) -> tuple[bool, int]:
        """(triggered, captured sample counter) from the FPGA status registers."""
        b = self._t.ctl_read(C.DSL_CTL_I2C_STATUS, 0, 4)
        trig_hit, cnt3, cnt2, cnt1 = b[0], b[1], b[2], b[3]
        captured = (cnt1 << 8) + (cnt2 << 16) + (cnt3 << 24) + ((trig_hit >> 2) << 32)
        return bool(trig_hit & 1), captured

    def _validate(self) -> tuple[C.ChannelMode, list[int]]:
        en = self.enabled_channels()
        if not en:
            raise ValueError("no channel enabled")
        mode = self.channel_mode
        if self._channel_mode is not None:
            bad = [i for i in en if i >= mode.num]
            if bad or len(en) > mode.vld_num:
                raise ValueError(f"{mode.descr}: channels {bad or en} cannot be used "
                                 f"(0..{mode.num - 1}, at most {mode.vld_num} enabled)")
            if not mode.min_samplerate <= self._samplerate <= mode.max_samplerate:
                raise ValueError(f"{mode.descr}: sample rate must be "
                                 f"{format_rate(mode.min_samplerate)}..{format_rate(mode.max_samplerate)}")
        for ch in self.trigger.simple:
            if ch not in en:
                raise ValueError(f"trigger on channel {ch}, which is disabled")
        depth = self.max_samples()
        if depth is not None and self._sample_count > depth:
            raise ValueError(f"buffer mode holds at most {depth} samples per channel with "
                             f"{len(en)} channels enabled (asked {self._sample_count}); "
                             "use stream mode or fewer channels")
        return mode, en

    def _transfer_plan(self, bytes_per_ms: int) -> tuple[int, int]:
        """(buffer size, number of transfers), after get_buffer_size()/get_number_of_transfers()."""
        ss = self.profile.superspeed
        align = 1024 if ss else 512
        if self._stream:
            size = (10 if ss else 20) * bytes_per_ms
            size = max(align, (size + align - 1) // align * align)
            # DSView queues ~40 ms; Python needs more slack, so queue ~250 ms.
            n = min(64, max(math.ceil((40 if ss else 100) * bytes_per_ms / size),
                            math.ceil(250 * bytes_per_ms / size)))
        else:
            size = 1024 * 1024
            n = 16 if ss else 4
        return size, max(1, n)

    def capture(self, timeout: float | None = None, progress: Callable | None = None) -> Capture:
        """Arm, wait for the trigger and return the capture.

        timeout:  seconds to wait in total (None = forever). On timeout a buffer-mode
                  capture is stopped and whatever was captured is uploaded (stop_option
                  "upload"); otherwise CaptureError is raised.
        progress: callback(triggered: bool, fraction: float) about 5x per second;
                  `self.capture_phase` says which phase the fraction belongs to.
        Ctrl-C or `stop()` from another thread ends the capture the same way.
        """
        if self._t is None:
            raise DSLogicError("device is closed")
        if self._busy:
            raise DSLogicError("capture already running")
        mode, en = self._validate()
        self.capture_phase = "waiting" if self.trigger.enabled else "capturing"
        self._busy = True
        self._stop.clear()
        self._cancel.clear()
        # Ctrl-C cancels: the analyser is stopped cleanly (no USB transfer torn down
        # mid-flight), then CaptureCancelled is raised. Use stop() to keep the data.
        import signal
        old_handler = None
        if threading.current_thread() is threading.main_thread():
            old_handler = signal.signal(signal.SIGINT, lambda *_: self._cancel.set())
        try:
            return self._run_capture(mode, en, timeout, progress)
        finally:
            self._busy = False
            if old_handler is not None:
                signal.signal(signal.SIGINT, old_handler)

    def _run_capture(self, mode, en, timeout, progress) -> Capture:
        t = self._t
        ch_num = len(en)
        limit = self._sample_count
        actual_samples = (limit + C.SAMPLES_ALIGN) & ~C.SAMPLES_ALIGN
        actual_bytes = actual_samples // C.ATOMIC_SAMPLES * ch_num * C.ATOMIC_SIZE
        trig_table = self.trigger.table()
        params = ArmParams(profile=self.profile, ch_mode=mode, samplerate=self._samplerate,
                           limit_samples=limit, actual_samples=actual_samples, enabled=en,
                           trigger=trig_table, stream=self._stream,
                           clock_external=self.external_clock, clock_negedge=self.clock_negedge,
                           filter_1t=self.filter)
        setting, ext32, fields = build_arm_blocks(params)
        log.debug("arm %s", fields)

        # dev_acquisition_start(): stop the previous run, arm the FPGA
        t.ctl_write(C.DSL_CTL_STOP)
        self._arm(setting, ext32)

        spool_dir = self._spool_target(actual_bytes)
        sink = ChannelSink(ch_num, actual_samples // C.ATOMIC_SAMPLES, spool_dir)
        if spool_dir:
            log.info("spooling %.2f GB to %s", actual_bytes / 1e9, sink.dir)
        st = {"num_bytes": 0, "header": None, "state": "start", "error": None,
              "actual_bytes": actual_bytes, "actual_samples": actual_samples, "abort": False}
        header_size = 1024 if self.profile.feature_caps & C.CAPS_FEATURE_USB30 else 512

        def on_header(data, status):
            if st["abort"] or status != "ok" or len(data) != header_size:
                if not st["abort"]:
                    st["error"] = f"trigger header transfer failed ({status}, {len(data)} bytes)"
                    st["state"] = "error"
                return False
            check_id, real_pos, _ram, rem_l, rem_h, hstatus = struct.unpack_from("<6I", data)
            if check_id != C.TRIG_CHECKID:
                st["error"] = f"bad trigger header id {check_id:#x}"
                st["state"] = "error"
                return False
            remain = (rem_h << 32) + rem_l
            st["header"] = dict(real_pos=real_pos, remain=remain, status=hstatus)
            # receive_header() measures against limit_samples; DSView's sample counts are
            # 1024-aligned so that equals the count armed into the FPGA. Use the armed
            # (aligned-up) count so arbitrary counts are not rounded down.
            if self._stream or remain < actual_samples:
                if not self._stream:
                    a = (actual_samples - remain) & ~C.SAMPLES_ALIGN
                    st["actual_bytes"] = a // C.ATOMIC_SAMPLES * ch_num * C.ATOMIC_SIZE
                    st["actual_samples"] = st["actual_bytes"] // ch_num * 8
                st["t_header"] = time.monotonic()
                st["state"] = "data"
            else:
                st["error"] = "device reported nothing captured"
                st["state"] = "error"
            return False

        def on_data(data, status):
            if status not in ("ok", "timeout"):
                if status != "cancelled" and st["state"] == "data":
                    st["error"] = f"bulk transfer {status}"
                    st["state"] = "error"
                return False
            if st["state"] != "data":
                return False
            n = len(data)
            if n:
                remain = st["actual_bytes"] - st["num_bytes"]
                take = min(n, remain)
                if take > 0:
                    sink.write(data[:take])
                    st["num_bytes"] += take
                if st["num_bytes"] >= st["actual_bytes"]:
                    st["state"] = "done"
                    return False
            return True

        size, ntr = self._transfer_plan(params.bytes_per_ms())
        reqs = [t.submit_read(header_size, on_header)]
        reqs += [t.submit_read(size, on_data) for _ in range(ntr)]
        t.ctl_write(C.DSL_CTL_START)
        started = time.monotonic()
        last_poll = started
        overflow = False
        stopped_early = False
        stop_sent_at = 0.0
        try:
            while st["state"] in ("start", "data"):
                try:
                    t.handle_events(0.05)
                except KeyboardInterrupt:
                    self._cancel.set()
                now = time.monotonic()
                if self._cancel.is_set():
                    st["abort"] = True
                    self.write_reg(C.CTR0_ADDR, C.bmFORCE_RDY)
                    break
                want_stop = self._stop.is_set() or (timeout is not None and now - started > timeout)
                if stopped_early and st["header"] is None and now - stop_sent_at > 5.0:
                    st["abort"] = True          # device ignored FORCE_STOP: give up
                    self.write_reg(C.CTR0_ADDR, C.bmFORCE_RDY)
                    break
                if want_stop and not st["abort"] and not stopped_early:
                    stopped_early = True
                    stop_sent_at = now
                    if not self._stream and self.stop_option == "upload" and st["header"] is None:
                        log.info("stopping: uploading what was captured")
                        self.write_reg(C.CTR0_ADDR, C.bmFORCE_STOP)
                    else:
                        st["abort"] = True
                        self.write_reg(C.CTR0_ADDR, C.bmFORCE_RDY)
                        break
                if now - last_poll >= 0.2:
                    last_poll = now
                    if st["state"] == "start":
                        trig, cnt = self.read_progress()
                        self.last_status = {"triggered": trig, "captured": cnt}
                        self.capture_phase = "capturing" if trig or not self.trigger.enabled else "waiting"
                        if progress:
                            frac = (limit - cnt) / limit if trig else cnt / limit
                            progress(trig, max(0.0, min(1.0, frac)))
                    else:
                        if self._stream and self.hw_status() & C.bmSYS_OVERFLOW:
                            overflow = True
                            st["abort"] = True
                            break
                        self.capture_phase = "capturing" if self._stream else "uploading"
                        if progress:
                            progress(True, st["num_bytes"] / max(1, st["actual_bytes"]))
        finally:
            for r in reqs:
                r.cancel()
            end = time.monotonic() + 2.0
            while any(not r.done for r in reqs) and time.monotonic() < end:
                t.handle_events(0.05)
            try:
                t.ctl_write(C.DSL_CTL_STOP)
                self.read_reg(C.HW_STATUS_ADDR)
            except DSLogicError as e:
                log.warning("stop command failed: %s", e)

        if self._cancel.is_set():
            self.capture_phase = "idle"
            raise CaptureCancelled("capture cancelled; data discarded")
        if overflow:
            raise OverflowError_(
                f"stream overflow after {st['num_bytes']} bytes: USB cannot sustain "
                f"{format_rate(self._samplerate)} x {ch_num} channels; lower one of them or use buffer mode")
        if st["state"] == "error":
            raise CaptureError(st["error"])
        got = st["num_bytes"] - st["num_bytes"] % (ch_num * C.ATOMIC_SIZE)
        if got == 0:
            raise CaptureError("capture stopped before any data arrived"
                               + (" (no trigger?)" if self.trigger.enabled else ""))
        self.capture_phase = "done"
        if progress:
            progress(True, 1.0)
        hdr = st["header"] or {}
        triggered = bool(hdr.get("status", 0) & 1) and self.trigger.enabled
        samples = min(got // ch_num * 8, st["actual_samples"], limit)
        nbytes = (samples + 7) // 8
        cap = Capture(samplerate=self._samplerate, num_samples=samples,
                                 data={ch: sink.channel_bytes(k, nbytes) for k, ch in enumerate(en)},
                                 channels={c.index: Channel(c.index, c.name, c.color, True, c.trigger)
                                           for c in self.channels if c.enabled},
                                 trigger_pos=int(hdr.get("real_pos", 0)) if triggered else 0,
                                 triggered=triggered, start_time=time.time(),
                                 decoders=list(self.decoders), device=self.profile.model)
        t_done = time.monotonic()
        t_hdr = st.get("t_header", t_done)
        # Buffer mode: the header arrives once the capture is complete, so armed->header is
        # trigger wait + post-trigger capture; header->done is the upload.
        timing = {"armed_to_data_s": t_hdr - started, "transfer_s": t_done - t_hdr,
                  "total_s": t_done - started}
        cap.meta = {"operation_mode": self.operation_mode, "channel_mode": mode.descr,
                    "threshold": self._threshold, "requested_samples": limit,
                    "stopped_early": stopped_early or samples < limit, "header": hdr,
                    "timing": timing, "spool_dir": sink.dir}
        if sink.dir:
            register_owner(cap)
        return cap

    def _spool_target(self, nbytes: int) -> str | None:
        """Folder to spool a capture of `nbytes` to, or None to keep it in RAM."""
        import shutil
        import tempfile
        mode = self.spool
        if mode == "never":
            return None
        if mode == "auto":
            free = available_memory()
            if (free and nbytes < free // 2) or (not free and nbytes < 2 << 30):
                return None
        target = self.spool_dir or tempfile.gettempdir()
        Path(target).mkdir(parents=True, exist_ok=True)
        disk_free = shutil.disk_usage(target).free
        if disk_free < nbytes * 1.05:
            raise CaptureError(
                f"capture needs {nbytes / 1e9:.1f} GB but {target} has {disk_free / 1e9:.1f} GB free; "
                "set spool_dir (--spool-dir) to a bigger drive or shorten the capture")
        return target

    def _arm(self, setting: bytes, ext32: bytes) -> None:
        """dsl_fpga_arm(): push the setting blocks to the FPGA."""
        t = self._t
        if not self.profile.superspeed:
            t.ctl_write(C.DSL_CTL_WORDWIDE, 0, bytes([C.bmWR_WORDWIDE]))
        words = len(setting) // 2
        t.ctl_write(C.DSL_CTL_BULK_WR, 0, bytes([words & 0xFF, (words >> 8) & 0xFF, (words >> 16) & 0xFF]))
        self._wait_hw(C.bmSYS_CLR, "FPGA sys_clr")
        if t.bulk_write(setting, 1000) != len(setting):
            raise DSLogicError("arm: setting block not fully sent")
        if self.profile.has_ch32 and t.bulk_write(ext32, 1000) != len(ext32):
            raise DSLogicError("arm: ext32 setting block not fully sent")
        t.ctl_write(C.DSL_CTL_INTRDY, 0, bytes([C.bmWR_INTRDY]))
        if not self.hw_status() & C.bmGPIF_DONE:
            raise DSLogicError("arm FPGA failed (GPIF not done)")

    def describe(self) -> str:
        """Human-readable summary of the current settings."""
        try:
            mode = self.channel_mode.descr
        except ValueError as e:
            mode = f"(invalid: {str(e).splitlines()[0]})"
        en = self.enabled_channels()
        lines = [f"{self.model} ({'USB3' if self.profile.superspeed else 'USB2'}), "
                 f"firmware {self.fw_version[0]}.{self.fw_version[1]}",
                 f"  mode       {self.operation_mode}, {mode}",
                 f"  samplerate {format_rate(self._samplerate)}, {self._sample_count} samples "
                 f"({format_time(self.duration)})",
                 f"  threshold  {self._threshold:.2f} V, filter {'on' if self.filter else 'off'}, "
                 f"clock {'external ' + ('falling' if self.clock_negedge else 'rising') if self.external_clock else 'internal'}",
                 f"  trigger    {self._trigger_text()}, position {self.trigger.position} %",
                 "  channels   " + ", ".join(
                     f"{i}:{self.channels[i].name}" + (f"({self.channels[i].color})" if self.channels[i].color else "")
                     for i in en)]
        if self.decoders:
            lines.append("  decoders   " + ", ".join(f"{d.label} ({d.id})" for d in self.decoders))
        return "\n".join(lines)

    def _trigger_text(self) -> str:
        t = self.trigger
        if t.serial is not None:
            return "serial"
        if t.stages:
            return f"advanced, {len(t.stages)} stage(s)"
        if not t.enabled:
            return "none (immediate)"
        return " AND ".join(f"{self.channels[c].name} {describe_condition(v)}" for c, v in t.simple.items())
