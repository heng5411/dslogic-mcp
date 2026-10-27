import numpy as np
import pytest

from dslogic import SPI, Capture
from dslogic.cli import main
from dslogic.viewer import UNIT, envelope

from signals import spi

pytest.importorskip("matplotlib")


def make_cap():
    cs, clk, mosi, miso = spi([0x9F, 0x12], [0x00, 0xC2], spb=8)
    pad = np.ones(40_000, np.uint8)
    z = np.zeros(40_000, np.uint8)
    return Capture.from_arrays(100_000_000, {0: np.concatenate([pad, cs, pad]),
                                             1: np.concatenate([z, clk, z]),
                                             2: np.concatenate([z, miso, z]),
                                             3: np.concatenate([z, mosi, z])})


def test_envelope_exact_and_buckets_match_data():
    cap = make_cap()
    kind, x, lv = envelope(cap, 1, 40_030, 40_060, 1000)
    assert kind == "exact" and np.array_equal(lv, cap.samples(1, 40_030, 40_060))
    kind, x, lo, hi = envelope(cap, 1, 0, cap.num_samples, 100)
    assert kind == "env"
    s = cap.samples(1)
    for k in range(len(x)):                   # each bucket: lo = all high, hi = any high
        a, b = x[k], x[k + 1] if k + 1 < len(x) else cap.num_samples
        seg = s[a:b]
        assert hi[k] == int(seg.any()) and lo[k] == int(seg.all())


def test_envelope_summary_level_on_long_capture():
    n = UNIT * 8 * 64                         # 64 summary units
    a = np.zeros(n, np.uint8)
    a[n // 2: n // 2 + 3] = 1                 # one short pulse in the middle
    cap = Capture.from_arrays(1e9, {0: a})
    kind, x, lo, hi = envelope(cap, 0, 0, n, 16)
    assert kind == "env" and hi.sum() == 1 and lo.sum() == 0


def test_render_png_and_cli(tmp_path):
    from dslogic.viewer import render
    cap = make_cap()
    cap.set_name(0, "CS")
    cap.add_decoder(SPI(clk=1, mosi=3, miso=2, cs="CS"))
    render(cap, tmp_path / "a.png", start=0.0004, end=0.0006)
    assert (tmp_path / "a.png").stat().st_size > 10_000
    src = cap.save(tmp_path / "c.dsl")
    assert main(["view", str(src), "--png", str(tmp_path / "b.png"), "--channels", "0-3"]) == 0
    assert (tmp_path / "b.png").exists()


def test_convert_crop(tmp_path):
    cap = make_cap()
    src = cap.save(tmp_path / "c.dsl")
    assert main(["convert", str(src), str(tmp_path / "crop.dsl"), "--start", "300us",
                 "--end", "500us", "--channels", "0,1"]) == 0
    c = Capture.load(tmp_path / "crop.dsl")
    assert c.enabled == [0, 1] and c.num_samples == 20_000
    assert np.array_equal(c[1], cap.samples(1, 30_000, 50_000))


def test_mouse_like_dsview():
    from matplotlib.backend_bases import MouseEvent
    from dslogic.viewer import Viewer
    cap = make_cap()
    v = Viewer(cap, interactive=False, decode=False)
    c, ax = v.fig.canvas, v.ax

    def ev(name, t, y=0.5, **kw):
        x, yp = ax.transData.transform((t, y))
        c.callbacks.process(name, MouseEvent(name, c, x, yp, **kw))

    full = ax.get_xlim()
    p = 0.0003
    ev("scroll_event", p, button="up", step=1)                 # wheel zooms in at the pointer
    x0, x1 = ax.get_xlim()
    assert x1 - x0 == pytest.approx((full[1] - full[0]) * 0.8)
    assert (p - x0) / (x1 - x0) == pytest.approx((p - full[0]) / (full[1] - full[0]))
    ev("scroll_event", p, button="down", step=-1, key="shift")  # shift+wheel scrolls
    assert ax.get_xlim()[1] - ax.get_xlim()[0] == pytest.approx(x1 - x0)
    a = ax.get_xlim()
    m = (a[0] + a[1]) / 2
    one_px = (a[1] - a[0]) / v._width()
    ev("button_press_event", m, button=3)                       # right-drag pans
    ev("motion_notify_event", m + (a[1] - a[0]) / 10)
    ev("button_release_event", m + (a[1] - a[0]) / 10, button=3)
    assert ax.get_xlim()[0] == pytest.approx(a[0] - (a[1] - a[0]) / 10, abs=2 * one_px)
    a = ax.get_xlim()
    z0, z1 = a[0] + (a[1] - a[0]) * 0.2, a[0] + (a[1] - a[0]) * 0.4
    ev("button_press_event", z0, button=1)                      # left-drag zooms to the region
    ev("motion_notify_event", z1)
    ev("button_release_event", z1, button=1)
    assert ax.get_xlim() == pytest.approx((z0, z1), abs=2 * one_px)
    b = ax.get_xlim()
    ev("button_press_event", b[0] + (b[1] - b[0]) * 0.3, button=1, dblclick=True)   # cursors
    ev("button_release_event", b[0] + (b[1] - b[0]) * 0.3, button=1)
    ev("button_press_event", b[0] + (b[1] - b[0]) * 0.7, button=1, dblclick=True)
    ev("button_release_event", b[0] + (b[1] - b[0]) * 0.7, button=1)
    assert len(v.cursors) == 2 and "cursors dt" in v.cstatus.get_text()
    ev("button_press_event", v.cursors[1], button=3)            # right-click a cursor deletes it
    ev("button_release_event", v.cursors[1], button=3)
    assert len(v.cursors) == 1
    b = ax.get_xlim()
    ev("button_press_event", (b[0] + b[1]) / 2, button=3, dblclick=True)   # right double-click: fit
    assert ax.get_xlim()[1] - ax.get_xlim()[0] == pytest.approx(full[1] - full[0])


def test_reorder_rows_by_dragging_names():
    from matplotlib.backend_bases import MouseEvent
    from dslogic.viewer import Viewer
    cap = make_cap()
    cap.add_decoder(SPI(clk=1, mosi=3, miso=2, cs=0))
    v = Viewer(cap, interactive=False)
    c, ax = v.fig.canvas, v.ax
    bb = ax.get_window_extent()
    assert [r.name for r in v.rows][-2:] == ["SPI MOSI", "SPI MISO"]

    def lab(name, row, **kw):                # an event in the name column, at a row's centre
        _, y = ax.transData.transform((0, v._base(row) + 0.5))
        c.callbacks.process(name, MouseEvent(name, c, bb.x0 - 40, y, **kw))

    lab("button_press_event", 4, button=1)   # drag the SPI MOSI row to the top
    lab("motion_notify_event", 0)
    c.callbacks.process("motion_notify_event",
                        MouseEvent("motion_notify_event", c, bb.x0 - 40,
                                   ax.transData.transform((0, 1.25))[1]))
    lab("button_release_event", 0, button=1)
    assert v.rows[0].name == "SPI MOSI" and [r.ch for r in v.rows[1:5]] == [0, 1, 2, 3]
    v.move_row(1, 4)
    assert [r.ch for r in v.rows[1:5]] == [1, 2, 3, 0]
    assert [t.get_text() for t in ax.get_yticklabels()][1] == "1  1"


def test_ruler_and_readout_resolution():
    from dslogic.viewer import Viewer, format_time_res
    assert format_time_res(0.005000123, 4e-9) == "5.000123 ms"
    assert format_time_res(0.005000124, 1e-8) == "5.00012 ms"
    assert format_time_res(3e-7, 4e-9) == "300 ns"
    v = Viewer(make_cap(), interactive=False, decode=False)
    v.set_view(0.0004, 0.0004 + 200e-9)
    labels = v.ax.xaxis.get_major_formatter().format_ticks(v.ax.get_xticks())
    assert len(set(labels)) == len(labels)      # every tick distinct when deeply zoomed


def clock_cap():
    one = np.r_[np.ones(30, np.uint8), np.zeros(70, np.uint8)]       # 1 MHz, 30 % at 100 MHz
    return Capture.from_arrays(100_000_000, {0: np.r_[np.zeros(500), np.tile(one, 50), np.zeros(500)]})


def test_pulse_at_and_measure():
    cap = clock_cap()
    p = cap.pulse_at(0, 515)
    assert (p["level"], p["start"], p["end"]) == (1, 500, 530)
    assert p["width"] == pytest.approx(300e-9) and p["frequency"] == pytest.approx(1e6)
    assert p["duty"] == pytest.approx(0.3)
    assert cap.pulse_at(0, 560)["width"] == pytest.approx(700e-9)
    assert cap.pulse_at(0, 100)["width"] is None                 # idle before the first edge
    m = cap.measure(0, 0, cap.num_samples)
    assert (m["edges"], m["rising"], m["periods"]) == (100, 50, 49)
    m = cap.measure(0, 500, 1000)                                # snapped on two rising edges
    assert (m["rising"], m["periods"]) == (6, 5) and m["frequency"] == pytest.approx(1e6)
    assert m["frequency"] == pytest.approx(1e6) and m["duty"] == pytest.approx(0.3)


def test_viewer_hover_click_and_shift_drag_measure():
    from matplotlib.backend_bases import MouseEvent
    from dslogic.viewer import Viewer
    cap = clock_cap()
    v = Viewer(cap, interactive=False, decode=False)
    c, ax = v.fig.canvas, v.ax
    v.set_view(4.9e-6, 12e-6)
    v.redraw()

    def ev(name, t, **kw):
        x, y = ax.transData.transform((t, 0.5))
        c.callbacks.process(name, MouseEvent(name, c, x, y, **kw))

    ev("motion_notify_event", 5.15e-6)
    assert "high pulse 300.000 ns" in v.mstatus.get_text() and "1 MHz" in v.mstatus.get_text()
    assert "duty 30.0%" in v.mstatus.get_text()
    ev("button_press_event", 5.15e-6, button=1)
    ev("button_release_event", 5.15e-6, button=1)
    assert any("H 300.000 ns" in t.get_text() for t in ax.texts)
    ev("button_press_event", 5.02e-6, button=1, key="shift")      # snaps to the edge at 5.00 us
    ev("motion_notify_event", 9.98e-6)
    ev("button_release_event", 9.98e-6, button=1, key="shift")    # snaps to 10.00 us
    assert "dt = 5.00 us" in v.mstatus.get_text() and "periods 5" in v.mstatus.get_text()
    assert "avg 1 MHz" in v.mstatus.get_text()


def test_deep_zoom_far_from_labels_draws():
    """Regression: labels billions of pixels off-screen crashed the font renderer."""
    from dslogic.viewer import Viewer
    cap = make_cap()
    cap.triggered, cap.trigger_pos = True, 100
    cap.add_decoder(SPI(clk=1, mosi=3, miso=2, cs=0))
    v = Viewer(cap, interactive=False)
    v.set_view(0.0007, 0.0007 + 200e-9)
    v.fig.canvas.draw()                                          # before the debounced redraw
    v.redraw()
    v.fig.canvas.draw()
