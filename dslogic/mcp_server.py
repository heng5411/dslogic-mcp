"""MCP server: the analyser and saved captures as tools for an AI assistant (dslogic mcp).

Runs over stdio. Captures stay loaded between calls, so decode / measure / render on a
multi-GB file only pay for the load once; tools that take `file` use the last captured
or loaded capture when it is left out. Paths may be given in Windows (D:\\x.dsl) or WSL
(/mnt/d/x.dsl) form and are converted to the form this host uses.
"""
from __future__ import annotations

import copy
import functools
import inspect
import os
import re
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import anyio
import anyio.from_thread
import anyio.to_thread
import numpy as np
from mcp.server.mcpserver import Context, Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .capture import Capture, format_rate, parse_time
from .cli import _ch, _channels, configure, parse_decoder

INSTRUCTIONS = """\
Drives a DreamSourceLab DSLogic U3Pro32/U3Pro16 logic analyser and reads its captures
(.dsl from DSView or this tool, .npz). Before calling `capture`, show the user every
setting you chose (rate, length, mode, channels and names, threshold, trigger and
position, decoders, timeout, output) and get their OK. Pick the rate at 4-10x the
fastest signal and the threshold at half the logic supply (3.3 V -> 1.65). Always set a
timeout: a trigger that never fires otherwise waits until it. Times you pass (start,
end, at) count from the capture start, e.g. "1.25ms"; decode times come back relative
to the trigger. Analysis tools reuse the last capture when `file` is omitted."""

server = MCPServer("dslogic", instructions=INSTRUCTIONS)

_open_device = None         # DSLogic.open; tests replace it with a fake analyser
_lock = threading.Lock()    # one capture / load / render at a time (matplotlib, USB)
_cache: OrderedDict[Path, tuple[tuple, Capture]] = OrderedDict()
_CACHE_SIZE = 4
_last: Capture | None = None
_active = None              # the analyser while a capture runs (for stop_capture)


def _opener():
    from .device import DSLogic
    return _open_device or DSLogic.open


# --- paths ------------------------------------------------------------------
def host_path(p: str) -> str:
    """D:\\x <-> /mnt/d/x, whichever this host understands."""
    p = os.path.expanduser(p.strip())
    if os.name == "nt":
        m = re.match(r"^/mnt/([a-zA-Z])(/.*)?$", p)
        if m:
            return f"{m[1].upper()}:" + (m[2] or "/").replace("/", "\\")
    else:
        m = re.match(r"^([a-zA-Z]):[\\/](.*)$", p)
        if m and Path(f"/mnt/{m[1].lower()}").is_dir():
            return f"/mnt/{m[1].lower()}/" + m[2].replace("\\", "/")
    return p


def _shown(p) -> str:
    """A saved path as the user's tools see it: Windows paths also get their WSL form."""
    p = str(p)
    m = re.match(r"^([a-zA-Z]):\\(.*)$", p)
    return f"{p}  (WSL: /mnt/{m[1].lower()}/{m[2].replace(chr(92), '/')})" if m else p


# --- captures ---------------------------------------------------------------
def _get(file: str | None) -> Capture:
    global _last
    if not file:
        if _last is None:
            raise ValueError("no capture yet: pass `file`, or run `capture` first")
        return _last
    path = Path(host_path(file)).resolve()
    st = path.stat()
    key = (st.st_mtime_ns, st.st_size)
    hit = _cache.get(path)
    if hit and hit[0] == key:
        _cache.move_to_end(path)
        cap = hit[1]
    else:
        cap = Capture.load(path)
        _cache[path] = (key, cap)
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)[1][1].release()
    _last = cap
    return cap


def _window(cap: Capture, start: str | None, end: str | None) -> tuple[int, int]:
    s0 = round(parse_time(start) * cap.samplerate) if start else 0
    s1 = round(parse_time(end) * cap.samplerate) if end else cap.num_samples
    s0, s1 = max(0, s0), min(cap.num_samples, s1)
    if s1 <= s0:
        raise ValueError(f"empty window {start or 0} .. {end or 'end'} "
                         f"(capture is {cap.num_samples / cap.samplerate:g} s)")
    return s0, s1


def _decoder(spec: str):
    try:
        return parse_decoder(spec)
    except SystemExit as e:             # the CLI parser exits on bad specs
        raise ValueError(str(e)) from None
    except KeyError as e:
        raise ValueError(f"decoder {spec!r} is missing {e}") from None


def _json(x):
    """Annotation data -> plain JSON values."""
    if isinstance(x, dict):
        return {str(k): _json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_json(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, bytes):
        return x.hex()
    return x


def _info(cap: Capture) -> dict[str, Any]:
    return {
        "summary": cap.summary(),
        "samplerate": format_rate(cap.samplerate),
        "samples": cap.num_samples,
        "duration_s": cap.duration,
        "channels": {str(i): cap.channels[i].name for i in cap.enabled},
        "triggered": cap.triggered,
        "trigger_s": cap.trigger_pos / cap.samplerate if cap.triggered else None,
        "decoders": [f"{d.label} [{d.settings_text()}]" for d in cap.decoders],
    }


def _decode(cap: Capture, decoders, s0: int, s1: int | None, kind: str | None,
            max_lines: int) -> list[dict]:
    out = []
    for d in decoders:
        r = {"decoder": d.label, "settings": d.settings_text()}
        try:
            anns = d.decode(cap, s0, s1)
        except NotImplementedError:
            out.append(r | {"note": "DSView-only decoder: saved into the .dsl, not decoded here"})
            continue
        if kind:
            anns = [a for a in anns if a.kind == kind]
        r |= {"count": len(anns), "shown": min(len(anns), max_lines), "annotations": [
            {"t_us": round(float(cap.time_of(a.start)) * 1e6, 4), "start": int(a.start),
             "end": int(a.end), "kind": a.kind, "text": a.text, "data": _json(a.data)}
            for a in anns[:max_lines]]}
        out.append(r)
    return out


def _pairs(d: dict | None) -> list[str] | None:
    return [",".join(f"{k}={v}" for k, v in d.items())] if d else None


# --- tools ------------------------------------------------------------------
def _tool(fn):
    """Register `fn` as a tool. Its errors (no analyser, missing file, bad decoder spec,
    overflow ...) reach the model with their message; MCP would otherwise send only
    'Error executing tool'."""
    def fail(e: BaseException):
        return ToolError(str(e) or type(e).__name__)

    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def wrapper(*a, **kw):
            try:
                return await fn(*a, **kw)
            except (Exception, SystemExit) as e:
                raise fail(e) from e
    else:
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            try:
                return fn(*a, **kw)
            except (Exception, SystemExit) as e:
                raise fail(e) from e
    return server.tool()(wrapper)


@_tool
def scan() -> list[str]:
    """List the connected DSLogic analysers. Empty: none plugged in, DSView holds it,
    or the WinUSB driver is missing."""
    from .device import DSLogic
    return [str(d) for d in DSLogic.scan()]


@_tool
def info(index: int = 0, settings: str | None = None) -> dict[str, Any]:
    """Open analyser `index` and return its settings and channel modes (which rates are
    possible with how many channels, buffer vs stream). `settings`: a DSView .dsl/.dsc
    or JSON to load first, to see what it would set."""
    with _lock, _opener()(index) as la:
        warnings = la.load_settings(host_path(settings)) if settings else []
        return {"device": la.describe(), "warnings": warnings,
                "channel_modes": [f"[{m.id}] {'stream' if m.stream else 'buffer'}  {m.descr}"
                                  for m in la.channel_modes()]}


@_tool
async def capture(
    ctx: Context,
    rate: str | None = None,
    duration: str | None = None,
    samples: str | None = None,
    channels: str | None = None,
    names: dict[str, str] | None = None,
    colors: dict[str, str] | None = None,
    mode: Literal["buffer", "stream"] | None = None,
    threshold: float | None = None,
    trigger: dict[str, str] | None = None,
    pos: int | None = None,
    decoders: list[str] | None = None,
    settings: str | None = None,
    timeout: float = 30,
    output: list[str] | None = None,
    index: int = 0,
    channel_mode: int | None = None,
    glitch_filter: bool = False,
    ext_clock: Literal["rising", "falling"] | None = None,
    no_trigger: bool = False,
    max_lines: int = 50,
) -> dict[str, Any]:
    """Capture from the analyser (confirm the settings with the user first).

    rate: "250M", "1G", "100k". duration: length, e.g. "20ms" (or samples: "10M").
    channels: "0-7,12". names: {"0": "CS", "1": "SCLK"}. mode: buffer (default; exact,
    in the analyser's memory) or stream (longer captures). threshold: volts, half the
    logic supply. trigger: {"CS": "falling"} or {"0": "rising", "1": "high"} (all must
    hold); pos: trigger position in % of the capture. decoders: ["spi:clk=1,mosi=3,
    miso=2,cs=0,cpol=0,cpha=0", "i2c:scl=4,sda=5", "uart:rx=6,baud=115200",
    "dsview:can:can_rx=5;bitrate=500000"]. settings: a DSView .dsl/.dsc whose setup to
    repeat (other arguments override it). timeout: seconds in total; a buffer capture
    uploads what it has. output: files to save (.dsl .sr .vcd .csv .npz).
    Returns the summary, timing and the first `max_lines` annotations per decoder; the
    capture stays loaded for decode / measure / render.
    """
    from .export import check_save_path
    outs = [host_path(o) for o in output or []]
    for o in outs:
        check_save_path(o)
    args = SimpleNamespace(
        settings=host_path(settings) if settings else None, spool_dir=None, spool=None,
        mode=mode, channels=channels, name=_pairs(names), color=_pairs(colors), rate=rate,
        samples=samples, time=duration, threshold=threshold, channel_mode=channel_mode,
        filter=glitch_filter, ext_clock=ext_clock, no_trigger=no_trigger,
        trigger=_pairs(trigger), pos=pos, decoder=decoders, save_settings=None)
    for spec in decoders or []:
        _decoder(spec)                      # bad specs fail before the analyser is armed
    last_report = [0.0]

    def progress(_trig, frac):
        now = time.monotonic()
        if now - last_report[0] >= 0.5:
            last_report[0] = now
            msg = _active.capture_phase if _active is not None else ""
            try:
                anyio.from_thread.run(ctx.report_progress, frac, 1.0, msg)
            except Exception:               # progress is best effort
                pass

    def work():
        global _active, _last
        with _lock:
            with _opener()(index) as la:
                warnings = configure(la, args)
                setup = la.describe()
                _active = la
                try:
                    cap = la.capture(timeout=timeout, progress=progress)
                finally:
                    _active = None
            _last = cap
            saved = [_shown(cap.save(o)) for o in outs]
            return {"setup": setup, "warnings": warnings, "summary": cap.summary(),
                    "timing": cap.timing_text(), "saved": saved,
                    "decode": _decode(cap, cap.decoders, 0, None, None, max_lines)}

    return await anyio.to_thread.run_sync(work)


@_tool
def stop_capture() -> str:
    """End the running capture now and keep what was captured (DSView's Stop)."""
    la = _active
    if la is None:
        return "no capture running"
    la.stop()
    return "stop requested"


@_tool
def load(file: str) -> dict[str, Any]:
    """Load a saved capture (.dsl/.npz) and return its rate, length, channels, trigger
    and saved decoders. Later tools use it when `file` is omitted."""
    with _lock:
        return _info(_get(file))


@_tool
def decode(file: str | None = None, decoders: list[str] | None = None,
           start: str | None = None, end: str | None = None, kind: str | None = None,
           max_lines: int = 200) -> list[dict]:
    """Decode a capture with its saved decoders, or `decoders` (as for capture) instead.
    start/end: window from the capture start ("1.2ms"). kind: keep one annotation kind
    (SPI "word"/"transfer", I2C "address"/"data", UART "data"). Each annotation has its
    time relative to the trigger (t_us), sample range, text and decoded data."""
    with _lock:
        cap = _get(file)
        decs = [_decoder(s) for s in decoders] if decoders else cap.decoders
        if not decs:
            raise ValueError("the capture has no decoders; pass `decoders`")
        s0, s1 = _window(cap, start, end) if start or end else (0, None)
        return _decode(cap, decs, s0, s1, kind, max_lines)


@_tool
def measure(channel: str, file: str | None = None, start: str | None = None,
            end: str | None = None) -> dict[str, Any]:
    """Edges, periods, mean frequency and duty cycle of one channel (number or name)
    between start and end (from the capture start; default: whole capture)."""
    with _lock:
        cap = _get(file)
        return _json(cap.measure(_ch(channel), *_window(cap, start, end)))


@_tool
def pulse(channel: str, at: str, file: str | None = None) -> dict[str, Any]:
    """The pulse on `channel` containing time `at` (from the capture start): level,
    start/end sample, width, period, frequency and duty, like DSView's hover."""
    with _lock:
        cap = _get(file)
        return _json(cap.pulse_at(_ch(channel), round(parse_time(at) * cap.samplerate)))


@_tool
def edges(channel: str, kind: Literal["both", "rising", "falling"] = "both",
          file: str | None = None, start: str | None = None, end: str | None = None,
          max_edges: int = 1000) -> dict[str, Any]:
    """Edge times on one channel (seconds from the capture start), with the widths
    between consecutive edges, so pulse trains and gaps can be read off directly."""
    with _lock:
        cap = _get(file)
        s0, s1 = _window(cap, start, end)
        e = cap.edges(_ch(channel), kind, start=s0, stop=s1)
        t = e[:max_edges] / cap.samplerate
        return {"count": len(e), "shown": len(t), "times_s": t.tolist(),
                "gaps_s": np.diff(t).tolist()}


@_tool
def render(file: str | None = None, start: str | None = None, end: str | None = None,
           channels: str | None = None, decode: bool = True, width: int = 1600,
           png: str | None = None) -> Image:
    """Picture of the waveform (with decoder rows) between start and end, returned as
    an image. channels: "0-3,7" or names "CS,SCLK". png: also save it to this path."""
    from .viewer import render as _render
    with _lock:
        cap = _get(file)
        chans = None
        if channels:
            parts = channels.split(",")
            chans = _channels(channels) if all(p.strip()[:1].isdigit() for p in parts) \
                else [_ch(p.strip()) for p in parts]
        kw = dict(channels=chans, decode=decode, width_px=width,
                  start=parse_time(start) if start else None,
                  end=parse_time(end) if end else None)
        fd, tmp = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            _render(cap, tmp, **kw)
            data = Path(tmp).read_bytes()
        finally:
            os.unlink(tmp)
    if png:
        Path(host_path(png)).write_bytes(data)
    return Image(data=data, format="png")


@_tool
def convert(output: str, file: str | None = None, start: str | None = None,
            end: str | None = None, channels: str | None = None,
            names: dict[str, str] | None = None, colors: dict[str, str] | None = None,
            decoders: list[str] | None = None) -> str:
    """Save a capture as .dsl .sr .vcd .csv or .npz, optionally cropped to start/end and
    a channel subset ("0-3"), renamed, recoloured or with decoders added."""
    from .export import check_save_path
    out = host_path(output)
    check_save_path(out)
    with _lock:
        cap = _get(file)
        if start or end or channels:
            s0, s1 = _window(cap, start, end)
            cap = cap.slice(s0, s1, channels=_channels(channels) if channels else None)
        elif names or colors or decoders:          # change a copy, not the cached original
            cap = copy.copy(cap)
            cap.channels = {i: copy.copy(c) for i, c in cap.channels.items()}
            cap.decoders = list(cap.decoders)
        for k, v in (names or {}).items():
            cap.set_name(_ch(k), v)
        for k, v in (colors or {}).items():
            cap.set_color(_ch(k), v)
        for spec in decoders or []:
            cap.add_decoder(_decoder(spec))
        return f"saved {_shown(cap.save(out))}: {cap.summary()}"


@_tool
def view(file: str) -> str:
    """Open the interactive waveform viewer on this machine's desktop (returns at once)."""
    import subprocess
    import sys
    path = host_path(file)
    if not Path(path).exists():
        raise ValueError(f"no such file: {path}")
    subprocess.Popen([sys.executable, "-m", "dslogic", "view", path],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)
    return f"viewer opened on {path}"


def main() -> None:
    server.run("stdio")
