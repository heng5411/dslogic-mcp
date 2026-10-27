"""Interactive waveform viewer (matplotlib).

Built for big captures: each redraw reads only the visible range, reduced to one
min/max pair per screen pixel; a coarse per-channel summary (min/max per 64 KiB of
packed data, computed once on all CPU cores) serves zoomed-out views, so a 15 GB stream
capture opens in seconds. Decoders run on the visible window only.

Mouse (like DSView): wheel = zoom at the pointer, Shift+wheel = scroll, left-drag = pan,
double-click = fit, right-click = A/B cursors with dt. Keys: f fit, t trigger, +/- zoom,
arrows pan, c clear cursors, q close.

    from dslogic import Capture
    from dslogic.viewer import show
    show(Capture.load("run.dsl"))
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .capture import Capture, format_time

UNIT = 1 << 16                 # summary unit: 64 KiB of packed bits = 512 Ki samples
_PALETTE = ["#FFD60A", "#0A84FF", "#FF453A", "#30D158", "#BF5AF2", "#FF9F0A", "#64D2FF", "#FF375F"]   # iOS dark


def channel_summary(cap: Capture, ch: int) -> tuple[np.ndarray, np.ndarray]:
    """(min, max) of the packed bytes per 512 KiB-sample unit (built on first use)."""
    from .overview import get_overview
    ov = get_overview(cap)
    if not ov.complete(ch):
        ov.build([ch], background=False)
    return ov.mn[ch][1][::2], ov.mx[ch][1][::2]


def envelope(cap: Capture, ch: int, s0: int, s1: int, width: int, build: bool = True):
    """Return ("exact", x_samples, levels) or ("env", x_bucket_starts, lo, hi) for [s0, s1).
    Zoomed-out views use the multi-level overview (dslogic.overview); lo/hi are NaN where
    it is still being built. build=False never waits for the overview."""
    from .overview import get_overview
    s0, s1 = max(0, int(s0)), min(cap.num_samples, int(s1))
    if s1 <= s0:
        return ("exact", np.array([s0]), np.array([0]))
    spp = (s1 - s0) / max(1, width)
    if spp <= 16:
        lv = cap.samples(ch, s0, s1)
        return ("exact", np.arange(s0, s1), lv)
    ov = get_overview(cap)
    lvl = ov.level_for(spp)
    if lvl is not None:
        if ch not in ov.valid:
            if not build:
                ov._alloc(ch)               # nothing built yet: all buckets unknown
            else:
                ov.build([ch], background=False)
        elif build and not ov.complete(ch) and ov.total_jobs == 0:
            ov.build([ch], background=False)
        x, lo, hi = ov.buckets(ch, lvl, s0, s1, width)
        return ("env", x, lo, hi)
    b0, b1 = s0 // 8, -(-s1 // 8)
    seg = np.asarray(cap.data[ch][b0:b1])
    b = np.unique(np.linspace(0, len(seg), width + 1).astype(np.int64)[:-1])
    smn = np.minimum.reduceat(seg, b)
    smx = np.maximum.reduceat(seg, b)
    x = (b + b0) * 8
    lo = (smn == 255).astype(np.int8)       # every bit high
    hi = (smx != 0).astype(np.int8)         # some bit high
    return ("env", x, lo, hi)


class _TimeFormatter:
    """Ruler labels with as many digits as the tick spacing needs (DSView-like)."""

    def __new__(cls):
        from matplotlib.ticker import Formatter

        class F(Formatter):
            def format_ticks(self, values):
                if len(values) == 0:
                    return []
                step = float(np.min(np.diff(values))) if len(values) > 1 else abs(values[0]) or 1.0
                return [format_time_res(v, step) for v in values]

            def __call__(self, x, pos=None):
                return format_time_res(x, 1e-9)

        return F()


def format_time_res(sec: float, res: float) -> str:
    """Time with enough decimals to resolve `res` seconds, in a unit picked from `sec`."""
    a = max(abs(sec), abs(res))
    for unit, div in (("s", 1.0), ("ms", 1e-3), ("us", 1e-6), ("ns", 1e-9)):
        if a >= div or unit == "ns":
            break
    dec = 0 if res <= 0 else max(0, int(np.ceil(-np.log10(res / div) - 1e-9)))
    return f"{sec / div:.{min(dec, 9)}f} {unit}"


class _Row:
    def __init__(self, kind, name, color, ch=None, dec=None, line=None):
        self.kind, self.name, self.color, self.ch, self.dec, self.line = kind, name, color, ch, dec, line
        self.key = ("ch", ch) if kind == "ch" else ("dec", id(dec), line)
        self.artists = []


def decoder_rows(d) -> list[tuple[str, str | None]]:
    """(row title, data field) for each row a decoder gets: SPI shows MOSI and MISO on
    separate rows; other decoders get one row."""
    from .decoders import SPI
    base = d.label.split(":", 1)[1] if ":" in d.label and d.label.split(":", 1)[0].isdigit() else d.label
    if isinstance(d, SPI):
        lines = [k for k in ("mosi", "miso") if d.channels.get(k) is not None]
        return [(f"{base} {k.upper()}", k) for k in lines] or [(base, None)]
    return [(base, None)]


def _decoder_source(cap, d, line):
    """(channel index, name) a decoder row reads: its data line, else its first channel."""
    keys = [line] if line else []
    keys += [k for k in ("sda", "rxtx") if k in d.channels and k not in keys]     # data before clock
    keys += [k for k, v in d.channels.items() if v is not None and k not in keys]
    for k in keys:
        v = d.channels.get(k)
        if v is None:
            continue
        try:
            i = cap.index(v)
        except KeyError:
            continue
        return i, cap.channels[i].name
    return None, ""


def build_rows(cap: Capture, channels=None, decode: bool = True) -> list:
    """Channel rows, with each decoder's rows placed right after the last channel it
    reads, titled with the channel it decodes and coloured like it."""
    chans = [cap.index(c) for c in channels] if channels else cap.enabled
    rows = [_Row("ch", cap.channels[c].name, cap.channels[c].color or _PALETTE[k % len(_PALETTE)], ch=c)
            for k, c in enumerate(chans)]
    if not decode:
        return rows
    color_of = {r.ch: r.color for r in rows}
    for d in cap.decoders:
        if not _can_decode(d):
            continue
        used = []
        for v in d.channels.values():
            try:
                used.append(cap.index(v))
            except (KeyError, TypeError):
                pass
        pos = max([i for i, r in enumerate(rows) if r.kind == "ch" and r.ch in used], default=len(rows) - 1)
        while pos + 1 < len(rows) and rows[pos + 1].kind == "dec":
            pos += 1                                  # after any decoder rows already there
        new = []
        for name, line in decoder_rows(d):
            src, src_name = _decoder_source(cap, d, line)
            if channels and src not in color_of:
                continue                              # the channel it decodes is not shown
            r = _Row("dec", name, color_of.get(src, "#9aa0a6"), dec=d, line=line)
            r.sub, r.src = src_name, src
            r.fmt, r.wbits = "hex", word_bits(d)               # how values are written
            if isinstance(getattr(d, "mode", None), int):         # SPI: mode, CPOL, CPHA at a glance
                r.mode_text = f"mode {d.mode}  CPOL {d.options['cpol']}  CPHA {d.options['cpha']}"
            new.append(r)
        rows[pos + 1:pos + 1] = new
    # two decoder rows that would read the same: tell them apart by their clock/first line
    seen = {}
    for r in rows:
        if r.kind == "dec":
            seen.setdefault((r.name, r.sub), []).append(r)
    for group in seen.values():
        if len(group) > 1:
            for r in group:
                ck = r.dec.channels.get("clk") or next(iter(r.dec.channels.values()), None)
                try:
                    r.sub += f"  clk {cap.channels[cap.index(ck)].name}"
                except (KeyError, TypeError):
                    pass
    return rows


class Viewer:
    """DSView-style viewer.

    Mouse: left-drag = zoom to region, right/middle-drag = pan, wheel = zoom at pointer,
    Shift/Ctrl+wheel = scroll, double-click = add cursor (drag it to move, right-click it
    to delete), click = pin the pulse measurement, Shift+drag = measure between edges,
    drag a row name = reorder rows, right double-click or f = fit.
    Keys: f fit, t trigger, +/- zoom, arrows pan, c clear cursors and measurements.

    Redraws never block the window: zoom/pan rescales what is drawn at once, while the
    new view is computed in a background thread (rows in parallel) and swapped in.
    """
    ROW = 1.4
    BG, FG, GRID = "#1b1d21", "#c8ccd4", "#3a3f47"
    CURSOR_COLORS = ("#ffd54f", "#4dd0e1", "#f48fb1", "#aed581", "#ce93d8", "#ffab91")

    def __init__(self, cap: Capture, channels=None, start: float | None = None,
                 end: float | None = None, decode: bool = True, width_px: int = 1600,
                 height_px: int | None = None, title: str | None = None, interactive: bool = True):
        import threading
        import matplotlib
        if not interactive:
            matplotlib.use("Agg")
        matplotlib.rcParams["toolbar"] = "None"
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection, PolyCollection
        from matplotlib.lines import Line2D
        from matplotlib.patches import Rectangle
        self.plt = plt
        self.cap = cap
        self.rate = cap.samplerate
        self.res = 1.0 / self.rate
        self.rows = build_rows(cap, channels, decode)
        h = height_px or max(300, 62 * len(self.rows) + 140)
        self.fig, self.ax = plt.subplots(figsize=(width_px / 100, h / 100), dpi=100)
        ax = self.ax
        self.fig.patch.set_facecolor(self.BG)
        ax.set_facecolor(self.BG)
        ax.xaxis.tick_top()
        ax.xaxis.set_major_formatter(_TimeFormatter())
        ax.tick_params(colors=self.FG, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(self.GRID)
        ax.grid(axis="x", color=self.GRID, lw=0.6)
        title_text = title or cap.summary().split(", trigger")[0]
        self._async = interactive and matplotlib.get_backend().lower() != "agg"
        # interactive: title goes to the window title bar (text rendering is the main draw cost)
        self.title = self.fig.suptitle("" if self._async else title_text, color=self.FG,
                                       fontsize=9, va="top")
        if self._async and getattr(self.fig.canvas, "manager", None) is not None:
            self.fig.canvas.manager.set_window_title(f"dslogic view - {title_text}")
        self.status = self.fig.text(0.01, 0.01, "left-drag: zoom   right-drag: pan   wheel: zoom   "
                                    "double-click: cursor   click: pulse   shift+drag: measure   "
                                    "drag names: reorder   f: fit", color="#9aa0a6", fontsize=8,
                                    family="monospace")
        self.mstatus = self.fig.text(0.01, 0.03, "", color="#7ee787", fontsize=8.5, family="monospace")
        self.cstatus = self.fig.text(0.99, 0.01, "", color="#ffd54f", fontsize=8.5, ha="right",
                                     family="monospace")
        self._tk_status = self._make_tk_status() if self._async else None
        if self._tk_status:                  # native Tk labels: no per-frame text rendering
            for t, var in zip((self.status, self.mstatus, self.cstatus), self._tk_status):
                t.set_visible(False)
                t.set_text = var.set
                t.get_text = var.get
            self.status.set_text("left-drag: zoom   right-drag: pan   wheel: zoom   double-click: "
                                 "cursor   click: pulse   shift+drag: measure   drag names: reorder   f: fit")
        self._layout()
        # per-row artists, created once and updated in place
        for r in self.rows:
            if r.kind == "ch":
                hi = Line2D([], [], color=r.color, lw=1.0, drawstyle="steps-post")
                lo = Line2D([], [], color=r.color, lw=1.0, drawstyle="steps-post")
                act = LineCollection([], colors=r.color, linewidths=1.0, alpha=0.75)
                r.artists = [ax.add_line(hi), ax.add_line(lo), ax.add_collection(act)]
            else:
                r.artists = [ax.add_collection(PolyCollection([], facecolors="#2f6f9f", alpha=0.55,
                                                               edgecolors="none"))]
                r.texts = []
        self.full = (0.0, cap.num_samples / self.rate)
        self.min_span = 16 / self.rate
        if cap.triggered:
            tx = cap.trigger_pos / self.rate
            ax.axvline(tx, color="#ff4d4d", ls="--", lw=1, alpha=0.9)
            ax.text(tx, 1.005, "T", color="#ff4d4d", ha="center", va="bottom", fontsize=9,
                    transform=ax.get_xaxis_transform())
        self._hover = ax.axvline(0, color="#ffffff", lw=0.6, alpha=0.0)
        (self._pulse_line,) = ax.plot([], [], color="#7ee787", lw=2, solid_capstyle="butt")
        self._band = ax.add_patch(Rectangle((0, 0), 0, 1, transform=ax.get_xaxis_transform(),
                                            color="#4dd0e1", alpha=0.18, visible=False))
        (self._insert_line,) = ax.plot([0, 1], [0, 0], color="#4dd0e1", lw=2, visible=False,
                                       transform=ax.get_yaxis_transform(), clip_on=False)
        self.cursors: list[float] = []
        self.pins: list[dict] = []           # pinned measurements: {key, t0, t1, label, color}
        self._overlay = []
        self._results: dict = {}             # row key -> last computed data
        self._mode = None
        self._m: dict = {}
        self._pulse = self._pulse_key = None
        self._decode_cache: dict = {}
        self._gen = 0
        self._ready = None
        self._lock = threading.Lock()
        # Fast-changing overlays are blitted over a saved background instead of redrawing
        # the figure (hover/status/zoom band/drag indicators).
        self._bg = None
        self._last_full = 0.0
        self._req_t = 0.0
        self._animated = [self._hover, self._pulse_line, self._band, self._insert_line]
        if not self._tk_status:
            self._animated += [self.status, self.mstatus]
        if self._async:
            for a in self._animated:
                a.set_animated(True)
            self.fig.canvas.mpl_connect("draw_event", self._on_draw)
            from concurrent.futures import ThreadPoolExecutor
            self._worker = ThreadPoolExecutor(max_workers=1)
            self._poll = self.fig.canvas.new_timer(interval=15)
            self._poll.add_callback(self._apply_ready)
            self._poll.start()
            self.fig.canvas.mpl_connect("close_event", lambda e: self._shutdown())
            self._pin_timer = self.fig.canvas.new_timer(interval=280)
            self._pin_timer.single_shot = True
            self._pin_timer.add_callback(self._fire_pin)
        self._apply_layout()
        ax.set_xlim(start if start is not None else self.full[0], end if end is not None else self.full[1])
        ax.callbacks.connect("xlim_changed", lambda a: (self._clip_texts(), self._request()))
        c = self.fig.canvas
        for name, fn in (("key_press_event", self._on_key), ("scroll_event", self._on_scroll),
                         ("button_press_event", self._on_press), ("button_release_event", self._on_release),
                         ("motion_notify_event", self._on_motion)):
            c.mpl_connect(name, fn)
        c.mpl_connect("resize_event", lambda e: (self._layout(), self._request()))
        self.redraw()

    # --- blitting -----------------------------------------------------------
    def _shutdown(self):
        self._poll.stop()
        self._pin_timer.stop()
        self._worker.shutdown(wait=False, cancel_futures=True)

    def _on_draw(self, ev):
        self._last_full = time.monotonic()
        c = self.fig.canvas
        self._bg = c.copy_from_bbox(self.fig.bbox)
        for a in self._animated:
            self.fig.draw_artist(a)
        c.blit(self.fig.bbox)

    def _blit(self):
        """Redraw only the animated overlays (cheap); full redraw if no background yet."""
        c = self.fig.canvas
        if not self._async or self._bg is None:
            c.draw_idle()
            return
        c.restore_region(self._bg)
        for a in self._animated:
            self.fig.draw_artist(a)
        c.blit(self.fig.bbox)

    # --- layout -------------------------------------------------------------
    def _fig_h(self) -> float:
        return self.fig.get_size_inches()[1] * self.fig.dpi

    def _make_tk_status(self):
        """Status lines as Tk labels under the plot (TkAgg only, e.g. Windows)."""
        try:
            import tkinter as tk
            widget = self.fig.canvas.get_tk_widget()
        except Exception:
            return None
        bar = tk.Frame(widget.master, bg=self.BG)
        bar.pack(side=tk.BOTTOM, fill=tk.X, before=widget)
        out = []
        for row, (color, anchor) in enumerate((("#9aa0a6", "w"), ("#7ee787", "w"), ("#ffd54f", "e"))):
            var = tk.StringVar(bar, "")
            tk.Label(bar, textvariable=var, bg=self.BG, fg=color, anchor=anchor,
                     font=("Consolas" if __import__("sys").platform == "win32" else "Monospace", 9)
                     ).pack(side=tk.TOP, fill=tk.X, padx=6)
            out.append(var)
        return out[0], out[1], out[2]          # status, measurement, cursors

    def _layout(self):
        h = self._fig_h()
        bottom = 10 / h if getattr(self, "_tk_status", None) else 48 / h
        self.fig.subplots_adjust(left=0.12, right=0.99, top=1 - 54 / h, bottom=bottom)
        self.title.set_y(1 - 6 / h)
        self.status.set_y(6 / h)
        self.cstatus.set_y(6 / h)
        self.mstatus.set_y(24 / h)

    def _base(self, i: int) -> float:
        return -self.ROW * i

    def _apply_layout(self):
        """Row order changed (or first draw): labels, limits and data offsets."""
        n = len(self.rows)
        self.ax.set_yticks([self._base(i) + 0.5 for i in range(n)],
                           [f"{r.ch}  {r.name}" if r.kind == "ch" else r.name for r in self.rows])
        for lab, r in zip(self.ax.get_yticklabels(), self.rows):
            lab.set_color(r.color)
            lab.set_fontsize(9)
        self.ax.set_ylim(self._base(n - 1) - 0.25, 1.3)
        for i, r in enumerate(self.rows):
            if r.key in self._results:
                self._apply_row(i, r, self._results[r.key])
        self._draw_overlay()

    # --- view ---------------------------------------------------------------
    def set_view(self, x0: float, x1: float) -> None:
        span = min(max(x1 - x0, self.min_span), (self.full[1] - self.full[0]) * 1.05)
        pad = (self.full[1] - self.full[0]) * 0.025
        x0 = min(max(x0, self.full[0] - pad), self.full[1] + pad - span)
        self.ax.set_xlim(x0, x0 + span)

    def zoom(self, factor: float, center: float | None = None) -> None:
        x0, x1 = self.ax.get_xlim()
        c = (x0 + x1) / 2 if center is None else center
        self.set_view(c - (c - x0) * factor, c + (x1 - c) * factor)

    def pan(self, dt: float) -> None:
        x0, x1 = self.ax.get_xlim()
        self.set_view(x0 + dt, x1 + dt)

    def _width(self) -> int:
        return max(100, int(self.ax.get_window_extent().width))

    def _px(self) -> float:
        x0, x1 = self.ax.get_xlim()
        return (x1 - x0) / self._width()

    # --- compute (background) and apply (GUI thread) ------------------------
    def _request(self):
        x0, x1 = self.ax.get_xlim()
        s0 = int(max(0, x0 * self.rate))
        s1 = int(min(self.cap.num_samples, x1 * self.rate + 1))
        job = (s0, s1, self._width(), list(self.rows))
        if not self._async:
            return
        with self._lock:
            self._gen += 1
            gen = self._gen
            self._req_t = time.monotonic()
        self._worker.submit(self._compute_job, gen, job)

    def _compute_job(self, gen, job):
        if gen != self._gen:
            return                                     # a newer view is already queued
        try:
            res = self._compute(*job)
        except Exception as e:                          # keep the viewer alive
            res = {"error": str(e)}
        with self._lock:
            if gen == self._gen:
                self._ready = (gen, res)

    MIN_FRAME = 1 / 30                      # cap full redraws at ~30 fps

    def _apply_ready(self):
        now = time.monotonic()
        if now - self._last_full < self.MIN_FRAME:
            return                          # keep the result; a newer one may replace it
        with self._lock:
            ready, self._ready = self._ready, None
        if ready is not None and ready[0] == self._gen:
            self._apply(ready[1])
        elif self._req_t > self._last_full and now - self._req_t > 0.06:
            self.fig.canvas.draw_idle()     # slow view (e.g. first overview of a huge file):
            self._last_full = now           # show the stretched old picture meanwhile

    def _compute(self, s0, s1, w, rows) -> dict:
        from concurrent.futures import ThreadPoolExecutor
        chans = [r for r in rows if r.kind == "ch"]
        decs = [r for r in rows if r.kind == "dec"]
        out = {}
        with ThreadPoolExecutor(max_workers=min(8, max(1, len(chans)))) as pool:
            for r, env in zip(chans, pool.map(lambda r: envelope(self.cap, r.ch, s0, s1, w), chans)):
                out[r.key] = env
        for r in decs:
            out[r.key] = ("dec",) + self._decode_window(r.dec, s0, s1)
        return out

    def _decode_window(self, d, s0, s1):
        if s1 - s0 > 200_000_000:
            return ("zoom", [])
        c = self._decode_cache.get(id(d))
        if c is None or not (c[0] <= s0 and s1 <= c[1]) or c[1] - c[0] > 8 * max(1, s1 - s0) + 4096:
            span = s1 - s0
            a, b = max(0, s0 - span), min(self.cap.num_samples, s1 + span)
            try:
                anns = d.decode(self.cap, a, b)
            except Exception as e:
                return ("error", str(e))
            words = [x for x in anns if x.kind != "transfer"] or anns
            c = (a, b, words)
            self._decode_cache[id(d)] = c
        return ("ok", [x for x in c[2] if x.end >= s0 and x.start <= s1])

    def redraw(self):
        """Compute and draw the current view now (used for images and tests)."""
        x0, x1 = self.ax.get_xlim()
        s0 = int(max(0, x0 * self.rate))
        s1 = int(min(self.cap.num_samples, x1 * self.rate + 1))
        self._apply(self._compute(s0, s1, self._width(), list(self.rows)))

    def _apply(self, res: dict):
        if "error" in res:
            self.mstatus.set_text(f"draw error: {res['error']}")
            return
        self._results.update(res)
        for i, r in enumerate(self.rows):
            if r.key in res:
                self._apply_row(i, r, res[r.key])
        self._clip_texts()
        self.fig.canvas.draw_idle()

    def _apply_row(self, i, r, data):
        base = self._base(i)
        if r.kind == "ch":
            hi, lo, act = r.artists
            kind, x, *rest = data
            t = x / self.rate
            if kind == "exact":
                lv = rest[0]
                hi.set_data(np.append(t, t[-1] + self.res), np.append(lv, lv[-1]) + base)
                lo.set_data([], [])
                act.set_segments([])
            else:
                l, h = rest
                hi.set_data(np.append(t, t[-1]), np.append(h, h[-1]) + base)
                lo.set_data(np.append(t, t[-1]), np.append(l, l[-1]) + base)
                m = l != h
                act.set_segments(np.stack([np.column_stack([t[m], np.full(m.sum(), base)]),
                                           np.column_stack([t[m], np.full(m.sum(), base + 1.0)])], axis=1))
            return
        poly = r.artists[0]
        for tx in r.texts:
            tx.set_visible(False)
        status, items = data[1], data[2]
        if status != "ok":
            poly.set_verts([])
            msg = "zoom in to decode" if status == "zoom" else f"{r.name}: {items}"
            self._text(r, 0, sum(self.ax.get_xlim()) / 2, base + 0.5, msg, "#9aa0a6")
            return
        px = self._px() * self.rate
        verts, k = [], 0
        for a in items[:3000]:
            x0, x1 = a.start / self.rate, max(a.end, a.start + 1) / self.rate
            verts.append([(x0, base + 0.1), (x0, base + 0.9), (x1, base + 0.9), (x1, base + 0.1)])
            label = _short(a, r.line)
            if not label:
                continue
            if (a.end - a.start) / px > 6.5 * max(1, len(label)) and k < 400:
                self._text(r, k, (x0 + x1) / 2, base + 0.5, label, "#e8eaed")
                k += 1
        poly.set_verts(verts)

    def _text(self, r, k, x, y, s, color):
        while len(r.texts) <= k:
            t = self.ax.text(0, 0, "", ha="center", va="center", fontsize=7.5, clip_on=True)
            t._dsl_pool = True
            r.texts.append(t)
        t = r.texts[k]
        t.set_position((x, y))
        t.set_text(s)
        t.set_color(color)
        t.set_visible(True)

    def _clip_texts(self):
        """Hide data-positioned labels far outside the view (deep zoom would overflow the
        font renderer: FT2Font._set_transform TypeError / raster overflow)."""
        x0, x1 = self.ax.get_xlim()
        span = x1 - x0
        for t in self.ax.texts:
            inside = x0 - span <= t.get_position()[0] <= x1 + span
            if getattr(t, "_dsl_pool", False):
                if not inside:                  # pooled decoder labels: shown by _apply_row only
                    t.set_visible(False)
            else:
                t.set_visible(inside)

    # --- overlays: cursors and pinned measurements --------------------------
    def _row_index(self, key) -> int | None:
        for i, r in enumerate(self.rows):
            if r.key == key:
                return i
        return None

    def _draw_overlay(self):
        for a in self._overlay:
            a.remove()
        self._overlay = []
        ax = self.ax
        xt = ax.get_xaxis_transform()
        for k, x in enumerate(self.cursors):
            col = self.CURSOR_COLORS[k % len(self.CURSOR_COLORS)]
            self._overlay.append(ax.axvline(x, color=col, lw=1))
            self._overlay.append(ax.text(x, 0.005, str(k + 1), color=col, ha="center", va="bottom",
                                         fontsize=9, transform=xt, fontweight="bold"))
        for p in self.pins:
            i = self._row_index(p["key"])
            if i is None:
                continue
            y = self._base(i) + 1.12
            self._overlay += ax.plot([p["t0"], p["t1"]], [y, y], color=p["color"], lw=2, marker="|", ms=9)
            self._overlay.append(ax.text((p["t0"] + p["t1"]) / 2, y + 0.06, p["label"], color=p["color"],
                                         ha="center", va="bottom", fontsize=8))
        cs = sorted(self.cursors)
        parts = []
        for a, b in list(zip(cs, cs[1:]))[-3:]:
            dt = b - a
            parts.append(f"{format_time_res(dt, self.res)}" + (f" ({_fmt_hz(1 / dt)})" if dt else ""))
        self.cstatus.set_text(("cursors dt: " + "  |  ".join(parts)) if parts else "")
        self._clip_texts()
        self.fig.canvas.draw_idle()

    # --- measurement helpers ------------------------------------------------
    def _row_at(self, y) -> int | None:
        if y is None:
            return None
        i = int(round((0.5 - y) / self.ROW))
        return i if 0 <= i < len(self.rows) else None

    def _ch_row_at(self, y) -> int | None:
        i = self._row_at(y)
        return i if i is not None and self.rows[i].kind == "ch" else None

    def _search_span(self) -> int:
        x0, x1 = self.ax.get_xlim()
        return int(min(1 << 24, max(4096, (x1 - x0) * self.rate)))

    def _snap(self, i: int, t: float) -> float:
        """Snap a time to the nearest edge of row i's channel within 8 pixels."""
        ch, s = self.rows[i].ch, int(min(max(t * self.rate, 0), self.cap.num_samples - 1))
        lim = int(8 * self._px() * self.rate) + 2
        cands = [e for e in (self.cap.prev_edge(ch, s, lim), self.cap.next_edge(ch, s, lim)) if e is not None]
        best = min(cands, key=lambda e: abs(e - s), default=None)
        return best / self.rate if best is not None else t

    def _hover_pulse(self, x, y):
        i = self._ch_row_at(y)
        if i is None:
            self._pulse_line.set_data([], [])
            self.mstatus.set_text("")
            self._pulse = None
            return
        s = int(x * self.rate)
        key = (i, s // max(1, int(self._px() * self.rate)))
        if key == self._pulse_key:
            return
        self._pulse_key = key
        p = self.cap.pulse_at(self.rows[i].ch, s, max_span=self._search_span())
        self._pulse = (i, p)
        if p["width"] is None:
            self._pulse_line.set_data([], [])
            self.mstatus.set_text(f"{p['name']}: {'high' if p['level'] else 'low'}, no edge on both "
                                  "sides within the view (zoom out)")
            return
        y = self._base(i) + 1.12
        self._pulse_line.set_data([p["start"] / self.rate, p["end"] / self.rate], [y, y])
        self.mstatus.set_text(_pulse_text(p))

    def _hover_decode(self, x, y):
        i = self._row_at(y)
        if i is None or self.rows[i].kind != "dec":
            return
        data = self._results.get(self.rows[i].key)
        if not data or data[1] != "ok":
            return
        s = x * self.rate
        hit = [a for a in data[2] if a.start <= s <= max(a.end, a.start + 1)]
        self.mstatus.set_text(f"{self.rows[i].dec.label}:  {hit[0].text}" if hit else "")

    def _pin_pulse_at(self, i, t):
        p = self.cap.pulse_at(self.rows[i].ch, int(t * self.rate), max_span=self._search_span())
        if p["width"] is None:
            return
        label = f"{'H' if p['level'] else 'L'} {format_time(p['width'])}"
        if p["frequency"]:
            label += f"  {_fmt_hz(p['frequency'])}  {p['duty'] * 100:.1f}%"
        self.pins.append(dict(key=self.rows[i].key, t0=p["start"] / self.rate, t1=p["end"] / self.rate,
                              label=label, color="#ffd54f"))
        self._draw_overlay()

    def _measure_text(self, i, a, b):
        m = self.cap.measure(self.rows[i].ch, int(round(a * self.rate)), int(round(b * self.rate)))
        dt = b - a
        txt = f"{m['name']}  dt = {format_time_res(dt, self.res)}"
        if dt > 0:
            txt += f"  (1/dt = {_fmt_hz(1 / dt)})"
        txt += f"  edges {m['edges']}"
        if m["frequency"]:
            txt += f"  periods {m['periods']}  avg {_fmt_hz(m['frequency'])}"
            if m["duty"] is not None:
                txt += f"  duty {m['duty'] * 100:.1f}%"
        return txt

    def _fire_pin(self):
        pend, self._m["pin"] = self._m.get("pin"), None
        if pend:
            self._pin_pulse_at(*pend)

    def _cursor_near(self, ev) -> int | None:
        if ev.x is None:
            return None
        best, bd = None, 6.0
        for k, t in enumerate(self.cursors):
            d = abs(self.ax.transData.transform((t, 0))[0] - ev.x)
            if d <= bd:
                best, bd = k, d
        return best

    def _label_row(self, ev) -> int | None:
        bb = self.ax.get_window_extent()
        if ev.x is None or ev.x >= bb.x0 or not (bb.y0 <= ev.y <= bb.y1):
            return None
        return self._row_at(self.ax.transData.inverted().transform((bb.x0, ev.y))[1])

    # --- mouse / keys -------------------------------------------------------
    def _on_scroll(self, ev):
        if ev.inaxes is not self.ax:
            return
        up = ev.button == "up"
        if ev.key in ("shift", "control", "ctrl"):
            x0, x1 = self.ax.get_xlim()
            self.pan((-0.15 if up else 0.15) * (x1 - x0))
        else:
            self.zoom(0.8 if up else 1.25, ev.xdata)

    def _on_press(self, ev):
        lr = self._label_row(ev) if ev.inaxes is None else None
        if lr is not None and ev.button == 1:
            self._mode, self._m = "reorder", {"from": lr}
            return
        if ev.inaxes is not self.ax or ev.xdata is None:
            return
        if ev.dblclick:
            if self._async:
                self._pin_timer.stop()
            self._m["pin"] = None
            if ev.button == 1:
                self.cursors.append(ev.xdata)
                self._draw_overlay()
            elif ev.button == 3:
                self.set_view(*self.full)
            self._mode = None
            return
        if ev.button == 1 and ev.key == "shift" and self._ch_row_at(ev.ydata) is not None:
            i = self._ch_row_at(ev.ydata)
            self._mode, self._m = "measure", {"row": i, "t0": self._snap(i, ev.xdata)}
            return
        if ev.button == 1 and self._cursor_near(ev) is not None:
            self._mode, self._m = "cursor", {"k": self._cursor_near(ev)}
            return
        if ev.button == 1:
            self._mode, self._m = "zoom", {"px": ev.x, "t0": ev.xdata, "y": ev.ydata}
            return
        if ev.button in (2, 3):
            self._mode, self._m = "pan", {"px": ev.x, "lim": self.ax.get_xlim(), "moved": False,
                                          "button": ev.button}

    def _on_motion(self, ev):
        mode = self._mode
        if mode == "pan" and ev.x is not None:
            x0, x1 = self._m["lim"]
            dx = ev.x - self._m["px"]
            if abs(dx) > 2:
                self._m["moved"] = True
            dt = dx * (x1 - x0) / self._width()
            self.set_view(x0 - dt, x1 - dt)
            return
        if mode == "zoom" and ev.xdata is not None:
            a, b = sorted((self._m["t0"], ev.xdata))
            self._band.set_bounds(a, 0, b - a, 1)
            self._band.set_visible(abs(ev.x - self._m["px"]) > 5)
            self._blit()
            return
        if mode == "cursor" and ev.xdata is not None:
            self.cursors[self._m["k"]] = ev.xdata
            self._draw_overlay()
            return
        if mode == "measure" and ev.xdata is not None:
            i = self._m["row"]
            a, b = sorted((self._m["t0"], self._snap(i, ev.xdata)))
            y = self._base(i) + 1.12
            self._pulse_line.set_data([a, b], [y, y])
            self.mstatus.set_text(self._measure_text(i, a, b))
            self._blit()
            return
        if mode == "reorder":
            bb = self.ax.get_window_extent()
            if ev.y is not None:
                y = self.ax.transData.inverted().transform((bb.x0, ev.y))[1]
                slot = int(np.clip(np.floor((1.3 - y) / self.ROW + 0.5), 0, len(self.rows)))
                self._m["to"] = slot
                yline = self._base(slot) + 1.2
                self._insert_line.set_data([-0.12, 1.0], [yline, yline])
                self._insert_line.set_visible(True)
                self._blit()
            return
        if ev.inaxes is not self.ax or ev.xdata is None:
            return
        self._hover.set_xdata([ev.xdata, ev.xdata])
        self._hover.set_alpha(0.35)
        s = int(ev.xdata * self.rate)
        txt = f"t = {format_time_res(ev.xdata, self.res)}"
        if self.cap.triggered:
            d = ev.xdata - self.cap.trigger_pos / self.rate
            txt += f"  (T{'+' if d >= 0 else '-'}{format_time_res(abs(d), self.res)})"
        if 0 <= s < self.cap.num_samples:
            vals = " ".join(f"{r.name}={int(self.cap.samples(r.ch, s, s + 1)[0])}"
                            for r in self.rows if r.kind == "ch")
            txt += f"   sample {s}   {vals}"
        self.status.set_text(txt)
        self._hover_pulse(ev.xdata, ev.ydata)
        self._hover_decode(ev.xdata, ev.ydata)
        self._blit()

    def _on_release(self, ev):
        mode, m = self._mode, self._m
        self._mode = None
        if mode == "zoom":
            self._band.set_visible(False)
            if ev.x is not None and abs(ev.x - m["px"]) > 5 and ev.xdata is not None:
                a, b = sorted((m["t0"], ev.xdata))
                self.set_view(a, b)
            else:
                i = self._ch_row_at(m["y"])
                if i is not None:
                    self._m = {"pin": (i, m["t0"])}
                    if self._async:
                        self._pin_timer.start()   # wait: a double-click adds a cursor instead
                    else:
                        self._fire_pin()
            self.fig.canvas.draw_idle()
        elif mode == "pan":
            if not m["moved"] and m["button"] == 3:
                k = self._cursor_near(ev)
                if k is not None:
                    del self.cursors[k]
                    self._draw_overlay()
        elif mode == "measure" and ev.xdata is not None:
            i = m["row"]
            a, b = sorted((m["t0"], self._snap(i, ev.xdata)))
            txt = self._measure_text(i, a, b)
            self.pins.append(dict(key=self.rows[i].key, t0=a, t1=b, label=txt.split("  ", 1)[1],
                                  color="#ff9f43"))
            self._pulse_line.set_data([], [])
            self.mstatus.set_text(txt)
            self._draw_overlay()
        elif mode == "reorder":
            self._insert_line.set_visible(False)
            src, dst = m["from"], m.get("to", m["from"])
            if dst > src:
                dst -= 1
            if dst != src:
                self.rows.insert(dst, self.rows.pop(src))
                self._apply_layout()
            self.fig.canvas.draw_idle()

    def move_row(self, src: int, dst: int) -> None:
        """Move row `src` to position `dst` (same as dragging its name)."""
        self.rows.insert(dst, self.rows.pop(src))
        self._apply_layout()

    def _on_key(self, ev):
        x0, x1 = self.ax.get_xlim()
        span = x1 - x0
        if ev.key in ("f", "home"):
            self.set_view(*self.full)
        elif ev.key == "t" and self.cap.triggered:
            c = self.cap.trigger_pos / self.rate
            self.set_view(c - span / 2, c + span / 2)
        elif ev.key in ("+", "="):
            self.zoom(0.5)
        elif ev.key in ("-", "_"):
            self.zoom(2.0)
        elif ev.key == "right":
            self.pan(span / 4)
        elif ev.key == "left":
            self.pan(-span / 4)
        elif ev.key == "c":
            self.cursors, self.pins = [], []
            self._draw_overlay()

    def show(self):
        self.plt.show()

    def save(self, path):
        self.fig.savefig(path, facecolor=self.BG)


def _pulse_text(p: dict) -> str:
    txt = f"{p['name']}  {'high' if p['level'] else 'low'} pulse {format_time(p['width'])}"
    if p["frequency"]:
        txt += (f"   period {format_time(p['period'])}   freq {_fmt_hz(p['frequency'])}"
                f"   duty {p['duty'] * 100:.1f}%")
    return txt


def _fmt_hz(hz: float) -> str:
    for unit, div in (("GHz", 1e9), ("MHz", 1e6), ("kHz", 1e3)):
        if hz >= div:
            return f"{hz / div:.4g} {unit}"
    return f"{hz:.4g} Hz"


def _can_decode(d) -> bool:
    from .decoders import DSViewDecoder
    return not isinstance(d, DSViewDecoder)


FORMATS = ("hex", "dec", "oct", "bin", "ascii")
FORMAT_NAMES = {"hex": "Hex", "dec": "Decimal", "oct": "Octal", "bin": "Binary", "ascii": "ASCII"}


def format_word(v: int, fmt: str = "hex", bits: int = 8) -> str:
    """A decoded value as the row shows it: hex 0x9F (the default), dec 159, oct 0o237,
    bin 0b10011111, ascii A (unprintable: \\x9F). Hex, octal and binary are padded to the
    word size."""
    if fmt == "dec":
        return str(v)
    if fmt == "oct":
        return f"0o{v:0{(bits + 2) // 3}o}"
    if fmt == "bin":
        return f"0b{v:0{bits}b}"
    if fmt == "ascii":
        return chr(v) if 32 <= v < 127 else f"\\x{v:02X}"
    return f"0x{v:0{(bits + 3) // 4}X}"


def word_bits(d) -> int:
    """Bits per decoded word for a decoder (SPI word size, UART data bits, else 8)."""
    o = getattr(d, "options", {}) or {}
    return int(o.get("wordsize") or o.get("num_data_bits") or 8)


def _short(a, line: str | None = None, fmt: str = "hex", bits: int = 8) -> str:
    """Box label: one field when the row shows one line (e.g. SPI MOSI row: 'C0')."""
    d = a.data
    if line:
        v = d.get(line)
        return format_word(v, fmt, bits) if isinstance(v, int) else ""
    if "mosi" in d or "miso" in d:
        return "/".join(format_word(d[k], fmt, bits) for k in ("mosi", "miso") if isinstance(d.get(k), int))
    if "address" in d:
        return f"{'R' if d.get('rw') == 'read' else 'W'}{format_word(d['address'], fmt, 7 if fmt != 'hex' else 8)}"
    if "value" in d:
        return format_word(d["value"], fmt, bits)
    return a.text[:12]


def show(cap: Capture, **kw) -> Viewer:
    v = Viewer(cap, **kw)
    v.show()
    return v


def render(cap: Capture, path, **kw) -> None:
    """Save a PNG/SVG of the waveform without opening a window."""
    v = Viewer(cap, interactive=False, **kw)
    v.save(path)
    v.plt.close(v.fig)
