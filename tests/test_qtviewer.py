import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("pyqtgraph")
pytest.importorskip("PySide6")

from pyqtgraph.Qt import QtCore  # noqa: E402

from dslogic import SPI, Capture  # noqa: E402
from dslogic.qtviewer import QtViewer  # noqa: E402

from signals import spi  # noqa: E402

L, R = QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.RightButton
NOMOD, SHIFT = QtCore.Qt.KeyboardModifier.NoModifier, QtCore.Qt.KeyboardModifier.ShiftModifier


def row(v, name):
    return [r.name for r in v.rows].index(name)


def make_cap():
    cs, clk, mosi, miso = spi([0x9F, 0x12], [0x00, 0xC2], spb=8)
    pad, z = np.ones(40_000, np.uint8), np.zeros(40_000, np.uint8)
    one = np.r_[np.ones(30, np.uint8), np.zeros(70, np.uint8)]
    clock = np.r_[z, np.tile(one, (len(cs) + 40_000) // 100 + 1)][: 80_000 + len(cs)]
    cap = Capture.from_arrays(100_000_000, {0: np.r_[pad, cs, pad], 1: np.r_[z, clk, z],
                                            2: np.r_[z, miso, z], 3: np.r_[z, mosi, z], 4: clock})
    cap.set_name(0, "CS")
    cap.add_decoder(SPI(clk=1, mosi=3, miso=2, cs="CS"))
    return cap


class Ev:
    """Minimal stand-in for pyqtgraph's MouseDragEvent / MouseClickEvent."""

    def __init__(self, vb, t0, t1, y, button, mods=NOMOD, start=False, finish=False, double=False):
        self.vb, self._b, self._m, self._s, self._f, self._d = vb, button, mods, start, finish, double
        self.p0 = vb.mapViewToScene(QtCore.QPointF(t0, y))
        self.p1 = vb.mapViewToScene(QtCore.QPointF(t1, y))

    def button(self): return self._b
    def modifiers(self): return self._m
    def isStart(self): return self._s
    def isFinish(self): return self._f
    def double(self): return self._d
    def accept(self): pass
    def ignore(self): pass
    def buttonDownScenePos(self): return self.p0
    def scenePos(self): return self.p1
    def lastScenePos(self): return self.p0
    def screenPos(self): return self.p1
    def buttonDownScreenPos(self): return self.p0


@pytest.fixture
def v():
    viewer = QtViewer(make_cap())
    viewer.win.show()
    viewer.process(0.3)
    yield viewer
    viewer.win.close()
    viewer.process(0.05)


def test_rows_split_and_data_drawn(v):
    assert [r.name for r in v.rows] == ["CS", "1", "2", "3", "SPI MOSI", "SPI MISO", "4"]
    v.set_view(400e-6, 410e-6)
    v.process(0.4)
    x, y = v.rows[1].items[0].getData()
    assert len(x) > 10 and np.nanmax(y) == pytest.approx(v._base(1) + 1)
    mosi, miso = v.rows[4], v.rows[5]
    assert [t.toPlainText() for t in mosi.texts if t.isVisible()] == ["0x9F", "0x12"]
    assert [t.toPlainText() for t in miso.texts if t.isVisible()] == ["0x00", "0xC2"]


def test_drag_zoom_pan_cursor_measure(v):
    vb = v.vb
    v.set_view(390e-6, 430e-6)
    v.process(0.2)
    y = v._base(row(v, '4')) + 0.5                                   # the clock row
    vb.mouseDragEvent(Ev(vb, 400e-6, 410e-6, y, L, start=True))     # left-drag: zoom box
    vb.mouseDragEvent(Ev(vb, 400e-6, 410e-6, y, L, finish=True))
    assert v.view() == pytest.approx((400e-6, 410e-6), abs=2e-8)
    vb.mouseDragEvent(Ev(vb, 405e-6, 404e-6, y, R, start=True))     # right-drag: pan
    assert v.view()[0] == pytest.approx(401e-6, abs=2e-8)
    vb.mouseClickEvent(Ev(vb, 403e-6, 403e-6, y, L, double=True))   # double-click: cursor
    vb.mouseClickEvent(Ev(vb, 405e-6, 405e-6, y, L, double=True))
    assert len(v.cursors) == 2 and "2.00 us" in v.cstatus.text()
    menu = v.cursor_menu(v.cursors[0], QtCore.QPointF(0, 0), show=False)   # right-click menu
    [a for a in menu.actions() if a.text() == "Delete cursor 1"][0].trigger()
    assert len(v.cursors) == 1
    vb.mouseDragEvent(Ev(vb, 400.02e-6, 405.02e-6, y, L, SHIFT, start=True))   # measure
    vb.mouseDragEvent(Ev(vb, 400.02e-6, 405.02e-6, y, L, SHIFT, finish=True))
    assert "periods" in v.mstatus.text() and "1 MHz" in v.mstatus.text()
    vb.mouseDragEvent(Ev(vb, 404e-6, 403e-6, y, R, start=True))     # right-drag still pans
    menu = v.context_menu(QtCore.QPointF(0, 0), 402e-6, y, show=False)
    [a for a in menu.actions() if a.text() == "Fit all"][0].trigger()
    assert v.view() == pytest.approx(v.full, abs=1e-9)


def test_click_pins_pulse_and_reorder(v):
    v.set_view(400e-6, 410e-6)
    v.process(0.2)
    v._pending_pin = (400.15e-6, v._base(row(v, '4')) + 0.5)
    v._fire_pin()
    assert v.pins and v.pins[-1]["label"].startswith("H 300.000 ns")
    v.move_row(4, 0)
    assert v.rows[0].name == "SPI MOSI" and v.rows[4].name == "3"
    assert v._name_items[0].toPlainText().split()[1:3] == ["SPI", "MOSI"]     # after the colour swatch


def test_context_menus(v, tmp_path):
    y = v._base(row(v, '4')) + 0.5
    m = v.context_menu(QtCore.QPointF(0, 0), 400.15e-6, y, show=False)
    acts = {a.text(): a for a in m.actions() if a.text()}
    assert {"Fit all", "Zoom in here", "Pin pulse measurement here", "Copy time",
            "Save visible range as..."} <= set(acts)
    assert not acts["Clear cursors"].isEnabled()
    acts["Add cursor at 400.15 us"].trigger()
    acts["Pin pulse measurement here"].trigger()
    assert len(v.cursors) == 1 and v.pins
    rm = v.row_menu(row(v, "1"), QtCore.QPointF(0, 0), show=False)
    [a for a in rm.actions() if a.text() == "Hide '1'"][0].trigger()
    assert "1" not in [r.name for r in v.rows] and len(v._hidden) == 1
    v.show_hidden_rows()
    assert [r.name for r in v.rows][-1] == "1"
    v.set_view(400e-6, 420e-6)
    v.save_range(tmp_path / "part.dsl")
    part = Capture.load(tmp_path / "part.dsl")
    assert part.num_samples == pytest.approx(2000, abs=2) and part.channels[0].name == "CS"


def test_glass_widgets_states(v):
    from dslogic import qtglass as G
    b = v.btn_save
    b.set_busy("Saving")
    assert b.state == "busy" and not b.isEnabled()
    b.done("Saved")
    assert b.state == "done" and b.isEnabled() and b._label_override == "Saved"
    b._back_to_idle()
    assert b.state == "idle"
    done = []
    b.run_busy(lambda: 42, done_label="Saved")
    for _ in range(100):
        v.process(0.02)
        if b.state == "done":
            break
    assert b.state == "done"
    b.run_busy(lambda: 1 / 0, on_error=done.append)
    for _ in range(100):
        v.process(0.02)
        if b.state == "error":
            break
    assert b.state == "error" and isinstance(done[0], ZeroDivisionError)
    seen = []
    v.mode_switch.changed.connect(seen.append)
    v.mode_switch.set_index(1)
    assert seen == [1] and v.mode == "pan"
    v.set_mode("measure")
    assert v.mode_switch.index == 2
    v.chip.busy("Rendering", delay_ms=0)
    v.process(0.05)
    assert v.chip.state == "busy"
    v.chip.ready()
    assert v.chip.state == "idle" and v.chip.text == "Ready"


def test_left_drag_follows_mode(v):
    vb = v.vb
    v.set_view(400e-6, 420e-6)
    v.process(0.1)
    y = v._base(row(v, '4')) + 0.5
    v.set_mode("pan")
    vb.mouseDragEvent(Ev(vb, 410e-6, 409e-6, y, L, start=True))      # left-drag pans
    assert v.view()[0] == pytest.approx(401e-6, abs=3e-8)
    v.set_mode("measure")
    vb.mouseDragEvent(Ev(vb, 402.02e-6, 405.02e-6, y, L, start=True))
    vb.mouseDragEvent(Ev(vb, 402.02e-6, 405.02e-6, y, L, finish=True))
    assert v.pins and "periods 3" in v.mstatus.text()


def test_decoders_grouped_named_and_coloured():
    from dslogic.viewer import build_rows
    cap = make_cap()
    cap.set_name(1, "spi1_sclk"); cap.set_name(3, "spi1_mosi"); cap.set_name(2, "spi1_miso")
    cap.decoders = [SPI(clk=1, mosi=3, miso=2, cs="CS", label="0:SPI"), SPI(clk=4, mosi=3, label="0:SPI")]
    rows = build_rows(cap)
    names = [(r.name, getattr(r, "sub", "")) for r in rows]
    assert names == [("CS", ""), ("spi1_sclk", ""), ("spi1_miso", ""), ("spi1_mosi", ""),
                     ("SPI MOSI", "spi1_mosi  clk spi1_sclk"), ("SPI MISO", "spi1_miso"),
                     ("4", ""), ("SPI MOSI", "spi1_mosi  clk 4")]
    assert rows[4].mode_text == "mode 0  CPOL 0  CPHA 0"
    assert rows[4].color == rows[3].color and rows[5].color == rows[2].color
    only = build_rows(cap, channels=[0, 4])
    assert [r.name for r in only] == ["CS", "4"]          # decoded channels hidden: rows left out
    part = build_rows(cap, channels=[0, 1, 3])
    assert [(r.name, getattr(r, "sub", "")) for r in part] == [
        ("CS", ""), ("spi1_sclk", ""), ("spi1_mosi", ""),
        ("SPI MOSI", "spi1_mosi  clk spi1_sclk"), ("SPI MOSI", "spi1_mosi  clk 4")]   # same data line


def test_fab_fans_out_and_tucks_back(v):
    from dslogic import qtglass as G
    fab = v.fab
    assert not fab.opened and not any(it.isVisible() for it in fab.items)
    fab.set_open(True, animate=False)
    assert all(it.isVisible() and it.appear == 1.0 for it in fab.items)
    tops = [it.geometry().top() for it in fab.items]
    assert tops == sorted(tops, reverse=True) and tops[0] < fab.fab.geometry().top()   # stacked above
    centre = fab.fab.geometry().center().x()
    for it in fab.items:                                      # icon wells line up with the button
        well = it.geometry().right() - G.FabAction.M - G.FabAction.H / 2
        assert abs(well - centre) <= 2
    v._refresh_glass()
    assert fab.backdrop is not None and fab.origin.contains(fab.items[-1].geometry().center())
    fab._set_t(0.5)                                           # half way: the nearest are out first
    assert fab.items[0].appear > fab.items[-1].appear
    fab.set_open(False, animate=False)
    assert not any(it.isVisible() for it in fab.items)
    fab._clicked()                                            # click: opens and stays open
    assert fab.opened and fab.pinned
    fab._maybe_close()
    assert fab.opened
    v.btn_save.set_busy("Saving")                             # busy action spins on the button too
    assert fab.fab.state == "busy"
    v.btn_save.done("Saved")
    assert fab.fab.state == "done"


def test_decoded_word_spans_its_bits_and_hover_marks_samples(v):
    v.set_view(400.2e-6, 401.8e-6)
    v.process(0.4)
    i = row(v, "SPI MOSI")
    data = v._results[v.rows[i].key]
    w = [a for a in data[2] if a.kind == "word"][0]
    c0, c1 = v._cell(w)
    bit = (w.bits[-1] - w.bits[0]) / 7
    assert c0 == pytest.approx(w.bits[0] - bit / 2) and c1 == pytest.approx(w.bits[-1] + bit / 2)
    digits = [t.toPlainText() for t in v.rows[i].bit_texts if t.isVisible()]
    assert "".join(digits[:8]) == "10011111"                  # 0x9F, bit by bit
    v._hover_dec(i, int((c0 + c1) / 2))
    assert v._word_band.isVisible() and "0x9F = 159" in v.mstatus.text() and "1001 1111" in v.mstatus.text()
    xs, ys = v._bit_dots.getData()
    assert len(xs) == 16                                      # 8 on MOSI, 8 on the clock
    v._clear_word_hover()
    assert not v._word_band.isVisible()


def test_drag_row_card_live_swap_and_commit(v):
    v.set_view(400e-6, 410e-6)
    v.process(0.4)
    names = [r.name for r in v.rows]                       # CS 1 2 3 SPI MOSI SPI MISO 4
    lvb = v.lvb
    src = names.index("4")
    y_src = v._base(src) + 0.5
    lvb.mouseDragEvent(Ev(lvb, 0.5, 0.5, y_src, L, start=True))          # pick up row "4"
    assert v._rd is not None and v._rd["card"].isVisible()
    assert not v.rows[src].items[0].isVisible()                         # its slot is the gap
    lvb.mouseDragEvent(Ev(lvb, 0.5, 0.5, y_src, L))
    y_to = v._base(1) + 0.5                                             # hover over row "1"
    ev = Ev(lvb, 0.5, 0.5, y_src, L)
    ev.p1 = lvb.mapViewToScene(QtCore.QPointF(0.5, y_to))
    lvb.mouseDragEvent(ev)
    assert v._rd["live"].index(src) == 1                                # live swap, not committed
    assert [r.name for r in v.rows] == names
    for _ in range(40):                                                 # neighbours slide down
        v._row_drag_tick()
    assert v._rd["off"][1] == pytest.approx(1.0)                        # row "1" moved one slot down
    fin = Ev(lvb, 0.5, 0.5, y_src, L, finish=True)
    fin.p1 = ev.p1
    lvb.mouseDragEvent(fin)                                             # release: drop and commit
    for _ in range(60):
        v.process(0.01)
        if v._rd is None:
            break
    assert v._rd is None
    assert [r.name for r in v.rows] == ["CS", "4", "1", "2", "3", "SPI MOSI", "SPI MISO"]
    assert all(it.isVisible() for r in v.rows for it in r.items)
    assert all(it.pos().y() == 0 for r in v.rows for it in r.items)


def test_theme_switch_and_name_badges(v):
    from dslogic import qtglass as G
    v.set_theme("light", animate=False)
    assert G.THEME == "light" and G.SURFACE == "#FFFFFF" and v.btn_theme.text() == "Dark theme"
    assert v.rows[0].items[0].opts["pen"].color().name() == G.theme_color(v.rows[0].color)
    v.set_theme("dark", animate=False)
    assert G.THEME == "dark" and v.btn_theme.text() == "Light theme" and not v._theme_auto
    cells = v._name_items
    assert len({c.badge_w for c in cells}) == 1                      # numbers line up down the column
    assert [c.badge() for c in cells] == ["0", "1", "2", "3", "3", "2", "4"]   # decoders: channel they read
    assert cells[0].toPlainText() == "0 CS"


def test_hover_pulse_drawn_on_trace(v):
    v.set_view(399.6e-6, 404.6e-6)
    v.process(0.3)
    i = row(v, "4")
    pos = v.vb.mapViewToScene(QtCore.QPointF(402.15e-6, v._base(i) + 0.5))
    v._on_move(pos)
    v.process(0.2)                                                  # looked up on the hover thread
    x, y = v._pulse_trace.getData()
    assert x[0] == pytest.approx(402.0e-6, abs=2e-8) and x[-1] == pytest.approx(402.3e-6, abs=2e-8)
    assert list(y) == [v._base(i), v._base(i) + 1, v._base(i) + 1, v._base(i)]   # high pulse outline
    tag = v._pulse_tag.toPlainText()
    assert "300" in tag and "1 MHz" in tag and "30.0%" in tag
    v.process(0.3)
    assert v._pulse_tag.isVisible()
    v._on_move(v.vb.mapViewToScene(QtCore.QPointF(402.15e-6, v._base(row(v, "SPI MOSI")) + 0.5)))
    v.process(0.3)
    assert not v._pulse_tag.isVisible()


def test_rename_channel_in_place(v, tmp_path):
    i = row(v, "CS")
    v.start_rename(i)
    _, ed = v._rename
    assert ed.isVisible() and ed.text() == "CS"
    ed.setText("nCS")
    v.finish_rename(True)
    assert v.rows[i].name == "nCS" and v.cap.channels[0].name == "nCS"
    assert v.cap.decoders[0].channels["cs"] == "nCS"                  # decoder bound by name still works
    v.start_rename(i)
    v._rename[1].setText("ignored")
    v.finish_rename(False)                                           # Esc: unchanged
    assert v.rows[i].name == "nCS"
    j = row(v, "3")
    v.rename_row(j, "mosi")
    assert [r.sub for r in v.rows if r.name == "SPI MOSI"] == ["mosi"]
    v.set_view(400e-6, 420e-6)
    v.save_range(tmp_path / "r.dsl")
    part = Capture.load(tmp_path / "r.dsl")
    assert [part.channels[k].name for k in range(4)] == ["nCS", "1", "2", "mosi"]


def test_about_sheet_credits_author(v):
    import dslogic
    sheet = v.show_about()
    assert sheet.isVisible() and sheet.author.text() == dslogic.__author__
    assert dslogic.__email__ in sheet.email.text() and "gitea" not in sheet.email.text()
    assert sheet.pages.currentIndex() == 0
    sheet.tabs.set_index(1)                                          # Open source tab
    assert sheet.pages.currentIndex() == 1
    texts = [w.text() for w in sheet.pages.currentWidget().findChildren(type(sheet.author))]
    assert any("DSView" in t for t in texts)
    m = v.context_menu(QtCore.QPointF(0, 0), 400e-6, 0.5, show=False)
    assert "About dslogic-tool..." in [a.text() for a in m.actions()]
    sheet.accept()


def test_spi_mode_shown():
    from dslogic import I2C, UART
    d = SPI(clk=1, mosi=3, cpol=1, cpha=0)
    assert d.mode == 2
    assert d.settings_text().startswith("mode 2: CPOL 1 (clock idles high), CPHA 0 (data read on the falling edge)")
    assert UART(rx=0, baudrate=9600, parity_type="even").settings_text() == "9600 baud, 8E1, lsb-first"
    assert "unshifted" in I2C(scl=0, sda=1).settings_text()
    cap = make_cap()
    cap.decoders = [d]
    v = QtViewer(cap)
    cell = v._name_items[[r.kind for r in v.rows].index("dec")]
    assert cell.row.mode_text == "mode 2  CPOL 1  CPHA 0" and "CPOL 1" in cell.toolTip()
    assert cell.toPlainText().endswith("mode 2  CPOL 1  CPHA 0")
    v.win.close()


def test_hover_never_sticks(v):
    v.set_view(399.6e-6, 404.6e-6)
    v.process(0.3)
    i = row(v, "4")
    v._on_move(v.vb.mapViewToScene(QtCore.QPointF(402.15e-6, v._base(i) + 0.5)))
    v.process(0.3)
    assert v._pulse_shown and "CS=" in v.status.text()               # levels arrive from the worker
    v._on_move(QtCore.QPointF(-50, -50))                             # pointer leaves the plot
    v.process(0.3)
    assert not v._pulse_shown and not v._pulse_tag.isVisible() and not v._hover_line.isVisible()
    v._on_move(v.vb.mapViewToScene(QtCore.QPointF(402.15e-6, v._base(i) + 0.5)))
    v.process(0.3)
    v.set_view(400.6e-6, 405.6e-6)                                   # view moves, pointer stays
    v.process(0.4)
    x, _ = v._pulse_trace.getData()
    t = v.vb.mapSceneToView(v._last_hover).x()
    assert x[0] <= t <= x[-1]                                        # highlight follows what is under it
    j = row(v, "SPI MOSI")
    v._on_move(v.vb.mapViewToScene(QtCore.QPointF(400.9e-6, v._base(j) + 0.5)))
    assert v._word_band.isVisible()
    v._hover_off()
    assert not v._word_band.isVisible()


def test_rename_ends_when_clicking_elsewhere(v):
    from pyqtgraph.Qt import QtGui, QtTest
    i = row(v, "CS")
    v.start_rename(i)
    _, ed = v._rename
    ed.setText("chip_sel")
    QtTest.QTest.mouseClick(v.glw.viewport(), QtCore.Qt.MouseButton.LeftButton,
                            QtCore.Qt.KeyboardModifier.NoModifier, QtCore.QPoint(40, 200))
    v.process(0.1)
    assert v._rename is None and v.rows[i].name == "chip_sel"          # kept, editor gone
    v.start_rename(i)
    v.start_rename(row(v, "1"))                                      # straight to another name
    assert v._rename[0] == row(v, "1") and v.rows[i].name == "chip_sel"
    v.finish_rename(False)
    assert not any(c.hover for c in v._name_items)


def test_pulse_tag_announces_geometry_changes(v):
    tag = v._pulse_tag
    tag.setHtml("<b>300 ns</b>", v.status.font())
    r1 = tag.boundingRect()
    tag.setAnchor((0.5, 1.0))
    r2 = tag.boundingRect()
    assert r2.bottom() < r1.bottom() and r2.width() == r1.width()
    tag.setHtml("<b>300 ns</b> 1 MHz 30.0%", v.status.font())
    assert tag.boundingRect().width() > r2.width()


def test_hover_repaints_old_and_new_area(v):
    """Every change of the hover visuals repaints where they were, not only where they are."""
    v.set_view(399.6e-6, 404.6e-6)
    v.process(0.3)
    i = row(v, "4")
    v._on_move(v.vb.mapViewToScene(QtCore.QPointF(402.15e-6, v._base(i) + 0.5)))
    v.process(0.4)
    vt = v.glw.viewportTransform()
    old = v._pulse_tag.deviceTransform(vt).mapRect(v._pulse_tag.boundingRect()).toAlignedRect()
    assert v._hover_painted.contains(old.center())
    v._on_move(v.vb.mapViewToScene(QtCore.QPointF(403.15e-6, v._base(i) + 0.5)))   # next pulse
    v.process(0.4)
    new = v._pulse_tag.deviceTransform(vt).mapRect(v._pulse_tag.boundingRect()).toAlignedRect()
    assert new.center() != old.center() and v._hover_painted.contains(new.center())
    v._hover_off()
    v.process(0.3)
    assert v._hover_painted.isEmpty()


def test_name_column_resizes(v):
    from dslogic.qtviewer import NAME_W, NAME_W_RANGE
    v.set_name_width(300)
    v.process(0.2)
    assert v.name_w == 300 and v.central.sidebar == 300
    assert abs(v.lvb.width() - 300) < 30                             # the column follows
    assert abs(v._name_items[0].w - v.lvb.width()) < 1              # and so do the cards
    assert abs(v.col_handle.geometry().center().x() - 300) <= 1     # handle sits on the edge
    v.set_name_width(5)
    assert v.name_w == NAME_W_RANGE[0]
    v.set_name_width(NAME_W)


def test_decoder_value_formats(v):
    from dslogic.viewer import format_word
    assert [format_word(0x9F, f) for f in ("hex", "dec", "oct", "bin", "ascii")] == \
        ["0x9F", "159", "0o237", "0b10011111", "\\x9F"]
    assert format_word(0x41, "ascii") == "A" and format_word(0x3FF, "hex", 10) == "0x3FF"
    v.set_view(400e-6, 410e-6)
    v.process(0.4)
    i = row(v, "SPI MOSI")
    assert v.rows[i].fmt == "hex"                                     # default
    v.set_row_format(i, "dec")
    assert [t.toPlainText() for t in v.rows[i].texts if t.isVisible()] == ["159", "18"]
    v.cycle_row_format(i)                                            # dec -> oct
    assert v.rows[i].fmt == "oct"
    m = v.row_menu(i, QtCore.QPointF(0, 0), show=False)
    sub = [a.menu() for a in m.actions() if a.text() == "Show values as"][0]
    [a for a in sub.actions() if a.text() == "Use for all decoder rows"][0].trigger()
    assert {r.fmt for r in v.rows if r.kind == "dec"} == {"oct"}
    v.set_row_format(i, "hex", all_rows=True)
    j = row(v, "SPI MISO")
    assert [t.toPlainText() for t in v.rows[j].texts if t.isVisible()] == ["0x00", "0xC2"]


def test_click_format_pill_cycles(v):
    v.process(0.2)
    i = row(v, "SPI MOSI")
    cell = v._name_items[i]
    assert not cell.pill.isEmpty()
    p = v.lvb.mapSceneToView(cell.to_scene(cell.pill.center()))
    v.lvb.mouseClickEvent(Ev(v.lvb, p.x(), p.x(), p.y(), L))
    assert v.rows[i].fmt == "dec" and v._rename is None              # a pill click is not a rename
