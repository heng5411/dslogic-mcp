"""Command line: dslogic scan | info | capture | decode | view | convert | mcp  (python -m dslogic ...)"""
from __future__ import annotations

import argparse
import logging
import sys

from .capture import Capture, format_rate
from .decoders import I2C, SPI, UART, DSViewDecoder


class ProgressBar:
    """One-line progress bar on stderr: `capturing [#####-----]  48.0 %  1.2 s`.
    Starts a new line when the phase changes (waiting -> capturing -> uploading)."""

    LABELS = {"waiting": "waiting for trigger", "capturing": "capturing",
              "uploading": "uploading", "saving": "saving", "done": "done"}

    def __init__(self, enabled: bool = True, stream=None):
        import time
        self.enabled = enabled
        self.out = stream or sys.stderr
        self.phase = None
        self.t0 = self.tp = time.monotonic()
        self.width_cols = 0
        enc = getattr(self.out, "encoding", None) or "ascii"
        try:
            "\u2588\u2591".encode(enc)
            self.fill, self.empty = "\u2588", "\u2591"
        except (UnicodeEncodeError, LookupError):
            self.fill, self.empty = "#", "-"
        self.interactive = getattr(self.out, "isatty", lambda: False)()

    def update(self, phase: str, frac: float) -> None:
        import shutil
        import time
        if not self.enabled:
            return
        now = time.monotonic()
        if phase == "done":                 # finish the current bar at 100 %
            if self.phase is not None:
                self._render(self.phase, 1.0, now, final=True)
                self.phase = None
            return
        if phase != self.phase:
            if self.phase is not None:
                self._render(self.phase, 1.0 if self.phase != "waiting" else self.last, now, final=True)
            self.phase, self.tp = phase, now
        self.last = frac
        cols = shutil.get_terminal_size((80, 20)).columns
        self.width_cols = cols
        if self.interactive:
            self._render(phase, frac, now)

    def _render(self, phase, frac, now, final=False):
        label = self.LABELS.get(phase, phase)
        tail = f" {frac * 100:5.1f} %  {now - self.tp:6.1f} s"
        width = max(10, min(50, (self.width_cols or 80) - len(label) - len(tail) - 4))
        n = int(round(frac * width))
        line = f"{label:<19} [{self.fill * n}{self.empty * (width - n)}]{tail}"
        if self.interactive:
            self.out.write("\r" + line + ("\n" if final else ""))
        elif final:
            self.out.write(line + "\n")
        self.out.flush()

    def close(self) -> None:
        import time
        if self.enabled and self.phase is not None:
            self._render(self.phase, self.last, time.monotonic(), final=True)
            self.phase = None


def _channels(spec: str) -> list[int]:
    """'0-3,8,10-11' -> [0,1,2,3,8,10,11]"""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out += range(int(a), int(b) + 1)
        else:
            out.append(int(part))
    return out


def _pairs(items: list[str] | None) -> dict[str, str]:
    """['0=CS', 'SCLK=red', '1=x,2=y'] -> {'0': 'CS', ...}"""
    out = {}
    for item in items or []:
        for kv in item.split(","):
            if "=" not in kv:
                raise SystemExit(f"expected KEY=VALUE, got {kv!r}")
            k, v = kv.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _ch(v: str):
    return int(v) if v.isdigit() else v


def parse_decoder(spec: str):
    """spi:clk=1,mosi=3,miso=2,cs=0,cpol=0,cpha=0  |  i2c:scl=4,sda=5  |
    uart:rx=6,baud=115200  |  dsview:<id>:<chan>=<n>,...;<option>=<value>,..."""
    kind, _, rest = spec.partition(":")
    kind = kind.lower()
    if kind == "dsview":
        did, _, rest = rest.partition(":")
        chan_part, _, opt_part = rest.partition(";")
        chans = {k: _ch(v) for k, v in _pairs([chan_part]).items()} if chan_part else {}
        opts = {k: _num(v) for k, v in _pairs([opt_part]).items()} if opt_part else {}
        label = str(opts.pop("label", did.split(":")[-1].upper()))
        return DSViewDecoder(did, chans, opts, label=label)
    kv = _pairs([rest]) if rest else {}
    label = kv.pop("label", "")
    if kind == "spi":
        return SPI(_ch(kv.pop("clk")), _ch(kv["mosi"]) if "mosi" in kv else None,
                   _ch(kv["miso"]) if "miso" in kv else None, _ch(kv["cs"]) if "cs" in kv else None,
                   cpol=int(kv.get("cpol", 0)), cpha=int(kv.get("cpha", 0)),
                   bitorder=kv.get("bitorder", "msb-first"), wordsize=int(kv.get("wordsize", 8)),
                   cs_polarity=kv.get("cs_polarity", "active-low"), label=label or "SPI")
    if kind == "i2c":
        return I2C(_ch(kv["scl"]), _ch(kv["sda"]), label=label or "I2C")
    if kind == "uart":
        return UART(_ch(kv.get("rx", kv.get("rxtx"))), baudrate=int(kv.get("baud", kv.get("baudrate", 115200))),
                    num_data_bits=int(kv.get("bits", 8)), parity_type=kv.get("parity", "none"),
                    num_stop_bits=float(kv.get("stop", 1)), invert=kv.get("invert", "no") in ("1", "yes", "true"),
                    label=label or "UART")
    raise SystemExit(f"unknown decoder {kind!r} (spi, i2c, uart, dsview:<id>)")


def _num(v: str):
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    return v


def _print_decode(cap: Capture, limit: int):
    for d in cap.decoders:
        try:
            anns = d.decode(cap)
        except NotImplementedError:
            print(f"{d.label}: (DSView-only decoder, saved to .dsl)")
            continue
        print(f"--- {d.label} [{d.settings_text()}] ({len(anns)} annotations)")
        for a in anns[:limit]:
            t = (a.start - cap.trigger_pos) / cap.samplerate
            print(f"{t * 1e6:12.3f} us  {a.kind:9s} {a.text}")
        if len(anns) > limit:
            print(f"... {len(anns) - limit} more")


def cmd_scan(args):
    from .device import DSLogic
    devs = DSLogic.scan()
    if not devs:
        print("no DSLogic U3Pro32/U3Pro16 found")
        return 1
    for d in devs:
        print(d)


def cmd_info(args):
    from .device import DSLogic
    with DSLogic.open(args.index, firmware_dir=args.firmware_dir) as la:
        if args.settings:
            for w in la.load_settings(args.settings):
                print(f"warning: {w}")
        print(la.describe())
        print(f"  FPGA image {la.hdl_version:#04x}" if la.hdl_version is not None else "")
        print("  channel modes:")
        for m in la.channel_modes():
            print(f"    [{m.id:2d}] {'stream' if m.stream else 'buffer'}  {m.descr}")


def configure(la, args) -> list[str]:
    """Apply the capture options in `args` (as parsed for 'capture') to an open analyser.
    Returns the warnings from loading --settings. Shared with the MCP server."""
    warnings = la.load_settings(args.settings) if args.settings else []
    if args.spool_dir:
        la.spool_dir = args.spool_dir
    if args.spool:
        la.spool = args.spool
    if args.mode:
        la.operation_mode = args.mode
    if args.channels:
        la.enable_channels(_channels(args.channels))
    for k, v in _pairs(args.name).items():
        la.set_name(_ch(k), v)
    for k, v in _pairs(args.color).items():
        la.set_color(_ch(k), v)
    if args.rate:
        la.samplerate = args.rate
    if args.samples:
        la.sample_count = args.samples
    if args.time:
        la.duration = args.time
    if args.threshold is not None:
        la.threshold = args.threshold
    if args.channel_mode is not None:
        la.channel_mode = args.channel_mode
    if args.filter:
        la.filter = True
    if args.ext_clock:
        la.external_clock = True
        la.clock_negedge = args.ext_clock == "falling"
    if args.no_trigger:
        la.clear_trigger()
    for k, v in _pairs(args.trigger).items():
        la.set_trigger(_ch(k), v)
    if args.pos is not None:
        la.trigger_position = args.pos
    for spec in args.decoder or []:
        la.add_decoder(parse_decoder(spec))
    if args.save_settings:
        la.save_settings(args.save_settings)
    return warnings


def cmd_capture(args):
    from .device import DSLogic
    from .export import check_save_path
    for out in args.output or []:
        check_save_path(out)          # fail before arming, not after the capture
    with DSLogic.open(args.index, firmware_dir=args.firmware_dir) as la:
        for w in configure(la, args):
            print(f"warning: {w}", file=sys.stderr)
        print(la.describe(), file=sys.stderr)

        from .keys import CTRL_X, KeyWatcher
        if not args.quiet:
            what = "stop and save" if args.output else "stop and keep the data"
            print(f"Ctrl-X: {what}   Ctrl-C: abort without saving", file=sys.stderr)
        bar = ProgressBar(enabled=not args.quiet)
        prog = lambda trig, frac: bar.update(la.capture_phase, frac)   # noqa: E731
        try:
            with KeyWatcher({CTRL_X: la.stop}):
                cap = la.capture(timeout=args.timeout, progress=prog)
        finally:
            bar.close()
    print(cap.summary())
    print(cap.timing_text())
    if cap.meta.get("spool_dir"):
        print(f"data spooled in {cap.meta['spool_dir']} (deleted on exit)")
    for out in args.output or []:
        sbar = ProgressBar(enabled=not args.quiet)
        kw = {"progress": lambda f: sbar.update("saving", f)} if str(out).lower().endswith(".dsl") else {}
        try:
            path = cap.save(out, **kw)
        except KeyboardInterrupt:
            sbar.close()
            from pathlib import Path
            Path(out).unlink(missing_ok=True)       # never leave a half-written file
            print(f"save of {out} aborted; partial file removed", file=sys.stderr)
            raise
        finally:
            sbar.close()
        print(f"saved {path}")
    if args.print_decode:
        _print_decode(cap, args.max_lines)
    if args.view:
        _open_viewer(cap, getattr(args, "backend", "auto"))   # straight from memory: no reload


def _open_viewer(cap, backend="auto", **kw):
    """Qt (pyqtgraph) viewer when available, else the matplotlib one."""
    if backend in ("auto", "qt"):
        try:
            from .qtviewer import QtViewer
        except ImportError:
            if backend == "qt":
                raise SystemExit("the Qt viewer needs: pip install pyqtgraph PySide6")
        else:
            QtViewer(cap, **kw).show()
            return
    from .viewer import Viewer
    Viewer(cap, **{k: v for k, v in kw.items() if k != "title"}).show()


def _window(cap, args) -> tuple[int, int]:
    """--start/--end (time from capture start, e.g. 1.2s, 350ms) -> sample range."""
    from .capture import parse_time
    s0 = round(parse_time(args.start) * cap.samplerate) if args.start else 0
    s1 = round(parse_time(args.end) * cap.samplerate) if args.end else cap.num_samples
    return max(0, s0), min(cap.num_samples, s1)


def cmd_view(args):
    from .capture import parse_time
    cap = Capture.load(args.file)
    for spec in args.decoder or []:
        cap.add_decoder(parse_decoder(spec))
    print(cap.summary())
    chans = [_ch(c) for c in args.channels.split(",")] if args.channels else None
    if args.channels and all(c.isdigit() or "-" in c for c in args.channels.split(",")):
        chans = _channels(args.channels)
    kw = dict(channels=chans, decode=not args.no_decode,
              start=parse_time(args.start) if args.start else None,
              end=parse_time(args.end) if args.end else None)
    if args.png:
        from .viewer import render
        render(cap, args.png, width_px=args.width, **kw)
        print(f"saved {args.png}")
    else:
        _open_viewer(cap, args.backend, width_px=args.width, **kw)


def cmd_decode(args):
    cap = Capture.load(args.file)
    if args.decoder:
        cap.decoders = [parse_decoder(s) for s in args.decoder]
    print(cap.summary())
    if not cap.decoders:
        print("no decoders in the file; add one with --decoder")
        return 1
    _print_decode(cap, args.max_lines)
    if args.output:
        print(f"saved {cap.save(args.output)}")


def cmd_convert(args):
    from .export import check_save_path
    check_save_path(args.output)
    cap = Capture.load(args.input)
    if args.start or args.end or args.channels:
        s0, s1 = _window(cap, args)
        chans = _channels(args.channels) if args.channels else None
        cap = cap.slice(s0, s1, channels=chans)
        print(f"cropped to {cap.summary()}")
    for k, v in _pairs(args.name).items():
        cap.set_name(_ch(k), v)
    for k, v in _pairs(args.color).items():
        cap.set_color(_ch(k), v)
    for spec in args.decoder or []:
        cap.add_decoder(parse_decoder(spec))
    print(f"saved {cap.save(args.output)}")


def cmd_mcp(args):
    try:
        from .mcp_server import main as serve
    except ImportError as e:
        raise SystemExit(f"the MCP server needs: pip install 'dslogic-tool[mcp]' ({e})")
    serve()


def main(argv=None):
    p = argparse.ArgumentParser(prog="dslogic", description="DSLogic U3Pro32/U3Pro16 capture tool")
    p.add_argument("-v", "--verbose", action="count", default=0)
    sub = p.add_subparsers(dest="cmd", required=True)

    def dev_args(sp):
        sp.add_argument("--index", type=int, default=0, help="which analyser (see 'scan')")
        sp.add_argument("--firmware-dir", help="folder holding DSLogicU3Pro32.bin (default: dslogic/res, then DSView's res folder)")

    sp = sub.add_parser("scan", help="list connected analysers")
    sp.set_defaults(fn=cmd_scan)

    sp = sub.add_parser("info", help="open the analyser and print its settings and modes")
    dev_args(sp)
    sp.add_argument("--settings", help="show the result of loading settings (JSON, DSView .dsl/.dsc)")
    sp.set_defaults(fn=cmd_info)

    sp = sub.add_parser("capture", help="capture and save")
    dev_args(sp)
    sp.add_argument("-r", "--rate", help="sample rate, e.g. 250M, 1G, 100k")
    sp.add_argument("-n", "--samples", help="samples per channel, e.g. 10M")
    sp.add_argument("-t", "--time", help="capture length, e.g. 20ms (instead of -n)")
    sp.add_argument("-c", "--channels", help="enabled channels, e.g. 0-7,12")
    sp.add_argument("--name", action="append", help="channel names, e.g. 0=CS,1=SCLK")
    sp.add_argument("--color", action="append", help="channel colours, e.g. CS=red,1=#00aaff")
    sp.add_argument("-m", "--mode", choices=["buffer", "stream"], help="operation mode")
    sp.add_argument("--channel-mode", type=int, help="explicit DSView channel mode id (see 'info')")
    sp.add_argument("--threshold", type=float, help="input threshold in volts (0-5)")
    sp.add_argument("--filter", action="store_true", help="enable 1-sample glitch filter")
    sp.add_argument("--ext-clock", nargs="?", const="rising", choices=["rising", "falling"],
                    help="sample on an external clock edge")
    sp.add_argument("--trigger", action="append",
                    help="simple trigger, e.g. CS=falling or 0=rising,1=high")
    sp.add_argument("--no-trigger", action="store_true", help="clear triggers from --settings")
    sp.add_argument("--pos", type=int, help="trigger position in %% of the capture")
    sp.add_argument("--decoder", action="append",
                    help="spi:clk=1,mosi=3,miso=2,cs=0  i2c:scl=4,sda=5  uart:rx=6,baud=115200  "
                         "dsview:can:can_rx=5;bitrate=500000")
    sp.add_argument("--settings",
                    help="load settings first: JSON from --save-settings, or a DSView .dsl capture / "
                         "settings file to repeat that capture's setup")
    sp.add_argument("--spool-dir", help="folder for large captures (stream mode, GBs of data)")
    sp.add_argument("--spool", choices=["auto", "always", "never"],
                    help="spool to disk: auto (when > half the free RAM, default), always, never")
    sp.add_argument("--save-settings", help="write the final settings to JSON")
    sp.add_argument("--timeout", type=float, help="seconds before stopping (buffer mode uploads what it has)")
    sp.add_argument("-o", "--output", action="append", help="output file(s): .dsl .sr .vcd .csv .npz")
    sp.add_argument("-p", "--print-decode", action="store_true", help="print decoder output")
    sp.add_argument("--view", action="store_true", help="open the waveform viewer when done")
    sp.add_argument("--backend", choices=["auto", "qt", "mpl"], default="auto", help="viewer backend")
    sp.add_argument("--max-lines", type=int, default=200)
    sp.add_argument("-q", "--quiet", action="store_true")
    sp.set_defaults(fn=cmd_capture)

    sp = sub.add_parser("decode", help="decode a saved .dsl/.npz capture")
    sp.add_argument("file")
    sp.add_argument("--decoder", action="append", help="as for capture (replaces the file's decoders)")
    sp.add_argument("-o", "--output", help="save (e.g. .dsl with the decoders attached)")
    sp.add_argument("--max-lines", type=int, default=200)
    sp.set_defaults(fn=cmd_decode)

    sp = sub.add_parser("view", help="waveform viewer for a .dsl/.npz capture (fast on GB files)")
    sp.add_argument("file")
    sp.add_argument("--start", help="initial view start, time from capture start (e.g. 1.2s)")
    sp.add_argument("--end", help="initial view end")
    sp.add_argument("--channels", help="channels to show, e.g. 0-3,7 or names CS,SCLK")
    sp.add_argument("--decoder", action="append", help="add a decoder (as for capture)")
    sp.add_argument("--no-decode", action="store_true", help="do not draw decoder rows")
    sp.add_argument("--png", help="render to an image file instead of opening a window")
    sp.add_argument("--width", type=int, default=1600, help="window/image width in pixels")
    sp.add_argument("--backend", choices=["auto", "qt", "mpl"], default="auto",
                    help="viewer: qt = pyqtgraph (smooth, default when installed), mpl = matplotlib")
    sp.set_defaults(fn=cmd_view)

    sp = sub.add_parser("convert", help="convert/crop/rename/recolour a saved capture")
    sp.add_argument("input")
    sp.add_argument("output")
    sp.add_argument("--start", help="keep from this time (from capture start), e.g. 1.25s")
    sp.add_argument("--end", help="keep up to this time, e.g. 1.30s")
    sp.add_argument("--channels", help="keep only these channels, e.g. 0-3,7")
    sp.add_argument("--name", action="append")
    sp.add_argument("--color", action="append")
    sp.add_argument("--decoder", action="append")
    sp.set_defaults(fn=cmd_convert)

    sp = sub.add_parser("mcp", help="run as an MCP server over stdio (for AI assistants)")
    sp.set_defaults(fn=cmd_mcp)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING - 10 * args.verbose, format="%(levelname)s %(message)s")
    try:
        return args.fn(args) or 0
    except KeyboardInterrupt as e:
        print(f"\n{e or 'aborted'}" if str(e) else "\naborted", file=sys.stderr)
        return 130
    except Exception as e:
        if args.verbose:
            raise
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
