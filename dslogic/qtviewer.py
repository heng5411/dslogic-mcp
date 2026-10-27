"""Waveform viewer on pyqtgraph (Qt): smooth zoom/pan on GB captures.

Zoom and pan only change the view transform, which Qt redraws in a few milliseconds.
The detail for the new view (one min/max per pixel, see viewer.envelope) is computed in
a background thread, channels in parallel, and swapped in when ready, so the window never
waits for data.

Mouse:
    left-drag            zoom to the selected region
    right/middle-drag    pan
    wheel / Shift+wheel  zoom at the pointer / scroll
    right-click          menu (fit, trigger, cursors, measurements, rows, save ...)
    double-click         add a cursor (drag it to move; right-click it for its menu)
    click                pin the measurement of the pulse under the pointer
    Shift+left-drag      measure along a channel (ends snap to edges)
    drag a row name      reorder rows (left column; right-click a name to hide/move it)
    hover a decoded word its bits are marked on the traces; value in the status bar
    round button         (bottom right) hover or click: fit, trigger, cursors, save ...
Keys: f fit, t trigger, + / - zoom, arrows pan, c clear cursors and measurements.

    from dslogic import Capture
    from dslogic.qtviewer import show
    show(Capture.load("run.dsl"))
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from . import qtglass as G
from .capture import Capture, format_rate, format_time
from .viewer import (FORMAT_NAMES, FORMATS, _fmt_hz, _pulse_text, _short, build_rows, envelope, format_time_res,
                     format_word)

NAME_X = 0.0                       # name cells start at the column's left edge

ROW = 1.4
NAME_W = 232                       # name column, px (default; drag its edge to resize)
NAME_W_RANGE = (110, 480)
CURSOR_COLORS = ("#FFD60A", "#64D2FF", "#FF375F", "#30D158", "#BF5AF2", "#FF9F0A")   # iOS dark
MONO = "Consolas" if os.name == "nt" else "Monospace"


class _TimeAxis(pg.AxisItem):
    """Ruler labels with the digits the tick spacing needs."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.enableAutoSIPrefix(False)

    def tickStrings(self, values, scale, spacing):
        return [format_time_res(v, spacing) for v in values]


class _Bridge(QtCore.QObject):
    """Delivers background results to the GUI thread (queued signal)."""
    ready = QtCore.Signal(object)


class _Cursor(pg.InfiniteLine):
    def __init__(self, viewer, x, color, number):
        self.base_color = color
        color = G.theme_color(color)
        super().__init__(pos=x, angle=90, movable=True, pen=pg.mkPen(color, width=1),
                         hoverPen=pg.mkPen(color, width=3),
                         label=str(number), labelOpts={"position": 0.03, "color": color,
                                                       "fill": G.RAISED})
        self.viewer = viewer
        self.sigPositionChanged.connect(lambda *_: viewer._cursor_status())

    def restyle(self, color: str) -> None:
        self.setPen(pg.mkPen(color, width=1))
        self.setHoverPen(pg.mkPen(color, width=3))
        self.label.setColor(color)
        self.label.fill = pg.mkBrush(G.RAISED)
        self.label.update()

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            ev.accept()
            self.viewer.cursor_menu(self, ev.screenPos())
        else:
            super().mouseClickEvent(ev)


class _WaveViewBox(pg.ViewBox):
    """ViewBox with DSView-style mouse handling (x only, rows fixed)."""

    def __init__(self, viewer):
        super().__init__(enableMenu=False)
        self.viewer = viewer
        self.setMouseEnabled(x=True, y=False)
        self._band = QtWidgets.QGraphicsRectItem()
        self._band.setPen(pg.mkPen(None))
        self._band.setBrush(pg.mkBrush(10, 132, 255, 40))
        self._band.setZValue(1e9)
        self._band.hide()
        self.addItem(self._band, ignoreBounds=True)

    def wheelEvent(self, ev, axis=None):
        if ev.modifiers() & (QtCore.Qt.KeyboardModifier.ShiftModifier | QtCore.Qt.KeyboardModifier.ControlModifier):
            (x0, x1), _ = self.viewRange()
            steps = ev.delta() / 120 if hasattr(ev, "delta") else np.sign(ev.angleDelta().y())
            self.viewer.pan(-0.15 * steps * (x1 - x0))
            ev.accept()
        else:
            super().wheelEvent(ev, axis=0)

    def mouseDragEvent(self, ev, axis=None):
        ev.accept()
        btn = ev.button()
        v = self.viewer
        p0, p1 = self.mapSceneToView(ev.buttonDownScenePos()), self.mapSceneToView(ev.scenePos())
        if btn == QtCore.Qt.MouseButton.LeftButton:
            shift = bool(ev.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier)
            if ev.isStart():
                row = v._ch_row_at(p0.y())
                mode = "measure" if shift else v.mode
                self._mode = "zoom" if mode == "measure" and row is None else mode
                self._row = row
            if self._mode == "measure":
                v._measure_drag(self._row, p0.x(), p1.x(), finish=ev.isFinish())
                return
            if self._mode == "pan":
                last = self.mapSceneToView(ev.lastScenePos())
                self.translateBy(x=last.x() - p1.x())
                return
            (x0, x1), (y0, y1) = self.viewRange()
            a, b = sorted((p0.x(), p1.x()))
            if ev.isFinish():
                self._band.hide()
                if abs(ev.screenPos().x() - ev.buttonDownScreenPos().x()) > 5:
                    v.set_view(a, b)
            else:
                self._band.setRect(QtCore.QRectF(a, y0, b - a, y1 - y0))
                self._band.show()
        elif btn in (QtCore.Qt.MouseButton.RightButton, QtCore.Qt.MouseButton.MiddleButton):
            last = self.mapSceneToView(ev.lastScenePos())
            self.translateBy(x=last.x() - p1.x())

    def mouseClickEvent(self, ev):
        v = self.viewer
        p = self.mapSceneToView(ev.scenePos())
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            ev.accept()
            if ev.double():
                v._pin_timer.stop()
                v.add_cursor(p.x())
            else:
                v._pending_pin = (p.x(), p.y())
                v._pin_timer.start(QtWidgets.QApplication.doubleClickInterval())
        elif ev.button() == QtCore.Qt.MouseButton.RightButton:
            ev.accept()                       # right-drag pans; a right click opens the menu
            if not ev.double():
                v.context_menu(ev.screenPos(), p.x(), p.y())


class _LabelViewBox(pg.ViewBox):
    """Row-name column; drag a name up/down to reorder rows."""

    def __init__(self, viewer):
        super().__init__(enableMenu=False, enableMouse=False)
        self.viewer = viewer
        self.setMouseEnabled(x=False, y=False)
        self.setXRange(0, 1, padding=0)

    def wheelEvent(self, ev, axis=None):
        ev.ignore()

    def mouseClickEvent(self, ev):
        v = self.viewer
        i = v._row_at(self.mapSceneToView(ev.scenePos()).y())
        on_pill = i is not None and v._name_items[i].on_pill(ev.scenePos())
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            ev.accept()
            v.row_menu(i, ev.screenPos())
        elif ev.button() == QtCore.Qt.MouseButton.LeftButton and on_pill:
            ev.accept()
            if not ev.double():
                v.cycle_row_format(i)                 # hex -> dec -> oct -> bin -> ascii
        elif ev.button() == QtCore.Qt.MouseButton.LeftButton and ev.double() and i is not None:
            ev.accept()
            v.start_rename(i)

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            ev.ignore()
            return
        ev.accept()
        v = self.viewer
        if ev.isStart():
            src = v._row_at(self.mapSceneToView(ev.buttonDownScenePos()).y())
            if src is not None:
                v.row_drag_start(src, ev.buttonDownScenePos())
        if v._rd is None:
            return
        if ev.isFinish():
            v.row_drag_finish()
        else:
            v.row_drag_move(ev.scenePos())


class _Tag(QtWidgets.QGraphicsItem):
    """A label chip at a fixed pixel size on the plot. Unlike pg.TextItem it announces every
    change of size or anchor to Qt, so a moved or hidden chip never leaves a stale copy."""

    def __init__(self):
        super().__init__()
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self._doc = QtGui.QTextDocument()
        self._doc.setDocumentMargin(3)
        self.anchor = (0.5, 0.5)
        self.fill, self.border = QtGui.QColor(G.RAISED), QtGui.QColor(G.ORANGE)
        self._rect = QtCore.QRectF()

    def setHtml(self, html: str, font: QtGui.QFont) -> None:
        self.prepareGeometryChange()
        self._doc.setDefaultFont(font)
        self._doc.setHtml(html)
        self._layout()

    def setAnchor(self, anchor) -> None:
        if tuple(anchor) != self.anchor:
            self.prepareGeometryChange()
            self.anchor = tuple(anchor)
            self._layout()

    def set_colors(self, fill, border) -> None:
        self.fill, self.border = QtGui.QColor(fill), QtGui.QColor(border)
        self.update()

    def _layout(self) -> None:
        sz = self._doc.size()
        w, h = sz.width() + 8, sz.height()
        self._rect = QtCore.QRectF(-w * self.anchor[0], -h * self.anchor[1], w, h)

    def toPlainText(self) -> str:
        return self._doc.toPlainText()

    def boundingRect(self):
        return self._rect.adjusted(-2, -2, 2, 2)

    def paint(self, p, *args):
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        path = QtGui.QPainterPath()
        path.addRoundedRect(self._rect.adjusted(0.5, 0.5, -0.5, -0.5), 6, 6)
        p.fillPath(path, self.fill)
        p.setPen(QtGui.QPen(self.border, 1))
        p.drawPath(path)
        p.translate(self._rect.x() + 4, self._rect.y())
        self._doc.drawContents(p)


class _ClickAway(QtCore.QObject):
    """While a name is being edited: a mouse press anywhere but the editor ends the edit."""

    def __init__(self, editor, done):
        super().__init__(editor)
        self.ed, self.done = editor, done

    def eventFilter(self, obj, ev):
        if (ev.type() == QtCore.QEvent.Type.MouseButtonPress and isinstance(obj, QtWidgets.QWidget)
                and obj is not self.ed and not self.ed.isAncestorOf(obj)):
            QtCore.QTimer.singleShot(0, self.done)       # after this press is delivered
        return False


class _ColumnHandle(QtWidgets.QWidget):
    """The name column's right edge: drag to resize the column, double-click for the
    default width. Shows a resize cursor and an accent line while hovered or dragged."""

    def __init__(self, parent, viewer):
        super().__init__(parent)
        self.v = viewer
        self.lit = 0.0
        self._la = G._Anim(self, "lit", 140)
        self._grab = None
        self.setCursor(QtCore.Qt.CursorShape.SplitHCursor)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setToolTip("Drag to resize the name column (double-click: default width)")

    def enterEvent(self, ev):
        self._la.to(1.0)

    def leaveEvent(self, ev):
        if self._grab is None:
            self._la.to(0.0)

    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self._grab = (ev.globalPosition().x(), self.v.name_w)

    def mouseMoveEvent(self, ev):
        if self._grab is not None:
            x0, w0 = self._grab
            self.v.set_name_width(w0 + ev.globalPosition().x() - x0)

    def mouseReleaseEvent(self, ev):
        if self._grab is not None:
            self._grab = None
            self.v.save_name_width()
            if not self.underMouse():
                self._la.to(0.0)

    def mouseDoubleClickEvent(self, ev):
        self.v.set_name_width(NAME_W)
        self.v.save_name_width()

    def paintEvent(self, ev):
        if self.lit <= 0.01:
            return
        p = QtGui.QPainter(self)
        c = QtGui.QColor(G.ACCENT)
        c.setAlphaF(0.9 * self.lit)
        p.fillRect(QtCore.QRectF(self.width() / 2 - 1, 0, 2, self.height()), c)


class _NameCell(QtWidgets.QGraphicsItem):
    """One row of the name column, drawn like an iOS list cell: a raised card, the channel
    number in a fixed-width badge (so numbers line up down the column), then the name.
    Decoder rows get an outlined badge holding the channel they decode."""

    def __init__(self, viewer):
        super().__init__()
        self.v = viewer
        self.row = None
        self.w, self.h, self.badge_w = 160.0, 30.0, 28.0
        self.hover = False
        self.pill = QtCore.QRectF()                  # value-format pill (decoder rows)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)    # drags go to the column
        self.setAcceptHoverEvents(True)

    def set_row(self, r, badge_w: float) -> None:
        self.row, self.badge_w = r, badge_w
        tip = ""
        if r.kind == "dec":
            try:
                tip = f"{r.dec.label}: {r.dec.settings_text()}"
            except Exception:
                tip = r.dec.label
        self.setToolTip(tip)
        self.update()

    def set_size(self, w: float, h: float) -> None:
        if (w, h) != (self.w, self.h):
            self.prepareGeometryChange()
            self.w, self.h = w, h

    def to_scene(self, pt: QtCore.QPointF) -> QtCore.QPointF:
        """Item point -> scene point. (mapToScene() is wrong for an item that ignores
        transformations inside a scaled view; go through the screen transform instead.)"""
        vt = self.v.glw.viewportTransform()
        return vt.inverted()[0].map(self.deviceTransform(vt).map(pt))

    def from_scene(self, pt: QtCore.QPointF) -> QtCore.QPointF:
        vt = self.v.glw.viewportTransform()
        return self.deviceTransform(vt).inverted()[0].map(vt.map(pt))

    def on_pill(self, scene_pos) -> bool:
        return not self.pill.isEmpty() and self.pill.contains(self.from_scene(scene_pos))

    def badge(self) -> str:
        r = self.row
        if r is None:
            return ""
        n = r.ch if r.kind == "ch" else getattr(r, "src", None)
        return "" if n is None else str(n)

    def toPlainText(self) -> str:
        r = self.row
        return " ".join(x for x in (self.badge(), r.name, getattr(r, "sub", ""), getattr(r, "mode_text", ""))
                        if x) if r else ""

    def boundingRect(self):
        return QtCore.QRectF(0, -self.h / 2, self.w, self.h)

    def hoverEnterEvent(self, ev):
        for c in self.v._name_items:                 # only one card lit, whatever events were lost
            if c is not self and c.hover:
                c.hover = False
                c.update()
        self.hover = True
        self.update()

    def hoverLeaveEvent(self, ev):
        self.hover = False
        self.update()

    def paint(self, p, opt, widget=None):
        r = self.row
        if r is None:
            return
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)
        dark = G.is_dark()
        card = QtCore.QRectF(8, -self.h / 2 + 2, self.w - 14, self.h - 4)
        path = QtGui.QPainterPath()
        path.addRoundedRect(card, 10, 10)
        if dark:
            g = QtGui.QLinearGradient(card.topLeft(), card.bottomLeft())
            g.setColorAt(0, QtGui.QColor(255, 255, 255, 17 if self.hover else 11))
            g.setColorAt(1, QtGui.QColor(255, 255, 255, 10 if self.hover else 5))
            p.fillPath(path, g)
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 16), 1))
            p.drawLine(QtCore.QPointF(card.left() + 9, card.top() + 0.5), QtCore.QPointF(card.right() - 9, card.top() + 0.5))
        else:
            p.fillPath(path.translated(0, 1), QtGui.QColor(0, 0, 0, 12))           # contact shadow
            p.fillPath(path, QtGui.QColor("#FFFFFF" if not self.hover else "#FAFAFC"))
            p.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 14), 1))
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            p.drawPath(path)
        col = QtGui.QColor(G.theme_color(r.color))
        bh = min(22.0, card.height() - 10)
        badge = QtCore.QRectF(card.left() + 8, -bh / 2, self.badge_w, bh)
        bp = QtGui.QPainterPath()
        bp.addRoundedRect(badge, 6, 6)
        if r.kind == "ch":
            fill = QtGui.QColor(col)
            fill.setAlpha(46 if dark else 34)
            p.fillPath(bp, fill)
        else:
            edge = QtGui.QColor(col)
            edge.setAlpha(150)
            p.setPen(QtGui.QPen(edge, 1.1, QtCore.Qt.PenStyle.DashLine))
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            p.drawPath(bp)
        p.setFont(G.ui_font(9, QtGui.QFont.Weight.DemiBold, tabular=True))
        p.setPen(col)
        p.drawText(badge, QtCore.Qt.AlignmentFlag.AlignCenter, self.badge())
        x = badge.right() + 10
        avail = card.right() - 10 - x
        self.pill = QtCore.QRectF()
        if r.kind == "dec":                     # value format; click it to switch
            pf = G.ui_font(8, QtGui.QFont.Weight.DemiBold)
            pt = getattr(r, "fmt", "hex")
            pw = QtGui.QFontMetricsF(pf).horizontalAdvance(pt) + 12
            self.pill = QtCore.QRectF(card.right() - 8 - pw, -9 if card.height() < 40 else -card.height() / 2 + 6,
                                      pw, 17)
            pp = QtGui.QPainterPath()
            pp.addRoundedRect(self.pill, 8.5, 8.5)
            pc = QtGui.QColor(G.ACCENT)
            pc.setAlpha(36 if dark else 26)
            p.fillPath(pp, pc)
            p.setFont(pf)
            p.setPen(QtGui.QColor(G.ACCENT))
            p.drawText(self.pill, QtCore.Qt.AlignmentFlag.AlignCenter, pt)
        title_avail = avail - (self.pill.width() + 6 if not self.pill.isEmpty() else 0)
        name_f = G.ui_font(10, QtGui.QFont.Weight.Medium)
        sub_f = G.ui_font(8.5)
        fm, sfm = QtGui.QFontMetricsF(name_f), QtGui.QFontMetricsF(sub_f)
        sub = getattr(r, "sub", "")
        mode = getattr(r, "mode_text", "")
        left = QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
        lines = [x for x in (sub, mode) if x]
        if lines and card.height() < fm.height() + len(lines) * sfm.height() + 6:
            lines = ["  ".join(x for x in (mode, sub) if x)]            # cramped: mode first, one line
        if lines and card.height() >= fm.height() + len(lines) * sfm.height() + 6:
            th = fm.height() + len(lines) * sfm.height()               # title over its detail lines
            p.setFont(name_f)
            p.setPen(QtGui.QColor(G.INK))
            p.drawText(QtCore.QRectF(x, -th / 2, title_avail, fm.height()), left,
                       fm.elidedText(r.name, QtCore.Qt.TextElideMode.ElideRight, title_avail))
            p.setFont(sub_f)
            p.setPen(QtGui.QColor(G.INK_DIM))
            for k, text in enumerate(lines):
                p.drawText(QtCore.QRectF(x, -th / 2 + fm.height() + k * sfm.height(), avail, sfm.height()), left,
                           sfm.elidedText(text, QtCore.Qt.TextElideMode.ElideRight, avail))
            return
        name = fm.elidedText(r.name, QtCore.Qt.TextElideMode.ElideRight, title_avail)
        p.setFont(name_f)
        p.setPen(QtGui.QColor(G.INK))
        p.drawText(QtCore.QRectF(x, -card.height() / 2, title_avail, card.height()), left, name)
        rest = title_avail - fm.horizontalAdvance(name) - 8
        if lines and rest > 24:
            p.setFont(sub_f)
            p.setPen(QtGui.QColor(G.INK_DIM))
            p.drawText(QtCore.QRectF(card.right() - 10 - rest, -card.height() / 2, rest, card.height()), left,
                       sfm.elidedText(lines[0], QtCore.Qt.TextElideMode.ElideRight, rest))


CREDITS = (
    ("DSView and libsigrok4DSL", "DreamSourceLab. The USB protocol, FPGA bitstreams and .dsl format "
                                 "this tool is ported from (GPL-3.0)."),
    ("pyqtgraph", "The fast plotting the waveform view is built on (MIT)."),
    ("Qt for Python (PySide6)", "The window and widgets (LGPL-3.0)."),
    ("NumPy", "Bit packing, edge search and overviews (BSD-3-Clause)."),
    ("python-libusb1", "USB access to the analyser (LGPL-2.1)."),
)


class _AboutSheet(QtWidgets.QDialog):
    """About: who wrote the tool (Author tab) and what it stands on (Open source tab)."""

    def __init__(self, parent):
        super().__init__(parent, QtCore.Qt.WindowType.Dialog | QtCore.Qt.WindowType.FramelessWindowHint)
        from . import __author__, __email__, __version__
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowTitle("About dslogic-tool")
        self.setFixedWidth(460)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(46, 34, 40, 40)            # card inset 18 for the shadow
        lay.setSpacing(0)

        def label(text, size, weight=QtGui.QFont.Weight.Normal, color=None, wrap=False):
            lab = QtWidgets.QLabel(text)
            lab.setFont(G.ui_font(size, weight))
            lab.setStyleSheet(f"color:{color or G.INK}; background: transparent")
            lab.setWordWrap(wrap)
            return lab

        head = QtWidgets.QHBoxLayout()
        head.setSpacing(14)
        mark = QtWidgets.QLabel()
        mark.setFixedSize(52, 52)
        mark.setPixmap(self._mark(52))
        head.addWidget(mark)
        names = QtWidgets.QVBoxLayout()
        names.setSpacing(2)
        names.addStretch(1)
        names.addWidget(label("dslogic-tool", 15, QtGui.QFont.Weight.DemiBold))
        names.addWidget(label(f"Version {__version__}", 9.5, color=G.INK_DIM))
        names.addStretch(1)
        head.addLayout(names, 1)
        done = QtWidgets.QPushButton("Done")            # iOS sheet: bold blue text, top right
        done.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        done.setFont(G.ui_font(10.5, QtGui.QFont.Weight.DemiBold))
        acc = QtGui.QColor(G.ACCENT)
        done.setStyleSheet(f"QPushButton{{background:transparent;color:{acc.name()};border:none;padding:4px 2px}}"
                           f"QPushButton:hover{{color:{acc.lighter(125).name()}}}"
                           f"QPushButton:pressed{{color:{QtGui.QColor(acc.red(), acc.green(), acc.blue(), 110).name(QtGui.QColor.NameFormat.HexArgb)}}}")
        done.clicked.connect(self.accept)
        done.setDefault(True)
        self.done_btn = done
        head.addWidget(done, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        lay.addLayout(head)
        lay.addSpacing(22)

        self.tabs = G.LiquidSwitch([("Author", ""), ("Open source", "")])
        self.pages = QtWidgets.QStackedWidget()
        self.tabs.changed.connect(self.pages.setCurrentIndex)
        lay.addWidget(self.tabs, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        lay.addSpacing(20)

        author = QtWidgets.QWidget()
        al = QtWidgets.QVBoxLayout(author)
        al.setContentsMargins(0, 0, 0, 0)
        al.setSpacing(4)
        al.addWidget(label("Written by", 9, color=G.INK_DIM))
        self.author = label(__author__, 13, QtGui.QFont.Weight.DemiBold)
        al.addWidget(self.author)
        al.addSpacing(12)
        link = label(f'<a href="mailto:{__email__}" style="color:{G.ACCENT}; text-decoration:none">'
                     f'{__email__}</a>', 9.5)
        self.email = link
        link.setOpenExternalLinks(True)
        link.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextBrowserInteraction)
        al.addWidget(link)
        al.addSpacing(6)
        al.addWidget(label("Free software under the GNU GPL, version 3 or later.", 9, color=G.INK_DIM, wrap=True))
        al.addStretch(1)
        self.pages.addWidget(author)

        oss = QtWidgets.QWidget()
        ol = QtWidgets.QVBoxLayout(oss)
        ol.setContentsMargins(0, 0, 0, 0)
        ol.setSpacing(10)
        for name, what in CREDITS:
            item = QtWidgets.QVBoxLayout()
            item.setSpacing(1)
            item.addWidget(label(name, 10, QtGui.QFont.Weight.Medium))
            item.addWidget(label(what, 9, color=G.INK_DIM, wrap=True))
            ol.addLayout(item)
        self.pages.addWidget(oss)
        lay.addWidget(self.pages)

    @staticmethod
    def _mark(size: int) -> QtGui.QPixmap:
        """App mark: a blue rounded tile with a digital waveform."""
        dpr = 2.0
        pm = QtGui.QPixmap(int(size * dpr), int(size * dpr))
        pm.setDevicePixelRatio(dpr)
        pm.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        r = QtCore.QRectF(0, 0, size, size)
        g = QtGui.QLinearGradient(r.topLeft(), r.bottomLeft())
        g.setColorAt(0, QtGui.QColor(G.ACCENT).lighter(118))
        g.setColorAt(1, QtGui.QColor(G.ACCENT).darker(118))
        path = QtGui.QPainterPath()
        path.addRoundedRect(r, size * 0.24, size * 0.24)
        p.fillPath(path, g)
        p.setPen(QtGui.QPen(QtGui.QColor("#FFFFFF"), 2.6, QtCore.Qt.PenStyle.SolidLine,
                            QtCore.Qt.PenCapStyle.RoundCap, QtCore.Qt.PenJoinStyle.RoundJoin))
        lo, hi = size * 0.66, size * 0.34
        xs = [0.18, 0.32, 0.32, 0.5, 0.5, 0.62, 0.62, 0.82]
        ys = [lo, lo, hi, hi, lo, lo, hi, hi]
        p.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(x * size, y) for x, y in zip(xs, ys)]))
        p.end()
        return pm

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        card = QtCore.QRectF(self.rect()).adjusted(18, 16, -18, -20)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        for k in range(12):                                   # soft, deep shadow
            p.setBrush(QtGui.QColor(0, 0, 0, int(18 * (1 - k / 12)) if G.is_dark() else int(9 * (1 - k / 12))))
            p.drawRoundedRect(card.adjusted(-k, -k + 4, k, k + 6), 18 + k, 18 + k)
        path = QtGui.QPainterPath()
        path.addRoundedRect(card, 18, 18)
        p.fillPath(path, QtGui.QColor(G.RAISED))
        p.setPen(QtGui.QPen(QtGui.QColor(G.MENU_BORDER), 1))
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawPath(path)


class QtViewer:
    def __init__(self, cap: Capture, channels=None, start: float | None = None, end: float | None = None,
                 decode: bool = True, width_px: int = 1600, height_px: int | None = None,
                 title: str | None = None, theme: str | None = None):
        """theme: "light" or "dark"; None follows the system (and its changes)."""
        self.app = pg.mkQApp("dslogic view")
        self._theme_auto = theme is None
        G.set_theme(theme or G.system_theme())
        pg.setConfigOptions(antialias=False, background=None, foreground=G.INK)
        self.cap = cap
        self.rate = cap.samplerate
        self.res = 1.0 / self.rate
        self.rows = build_rows(cap, channels, decode)
        self.full = (0.0, cap.num_samples / self.rate)
        self.cursors: list[_Cursor] = []
        self.pins: list[dict] = []
        self._results: dict = {}
        self._decode_cache: dict = {}
        self._gen = 0
        self._lock = threading.Lock()
        self._worker = ThreadPoolExecutor(max_workers=1)
        self._dec_worker = ThreadPoolExecutor(max_workers=1)
        from .overview import get_overview
        self._ov = get_overview(cap)             # overview pyramid, built in the background
        self._pool = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4))
        self._bridge = _Bridge()
        self._bridge.ready.connect(self._apply)
        self._pending_pin = None
        self._pulse_key = None
        self._hidden: list = []                 # (row, name item) hidden from view
        self._rd = None                          # row drag in progress
        self._rename = None                      # (row, editor) while a name is edited
        self._rd_timer = QtCore.QTimer()
        self._rd_timer.timeout.connect(self._row_drag_tick)

        # --- window: top bar, plot, status bar; actions on a floating action button ------
        self.mode = "zoom"
        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle("dslogic view - " + (title or cap.summary().split(", trigger")[0]))
        h = height_px or max(380, 58 * len(self.rows) + 200)
        self.win.resize(width_px, h)
        central = G.Surface()
        self.name_w = self._load_name_width()
        central.sidebar = self.name_w
        self.central = central
        bars = []
        # Windows 11: the window itself becomes glass (system acrylic shows the blurred desktop)
        # and the greys are drawn slightly translucent over it.
        self.window_glass = False
        if G.window_glass_supported():
            self.win.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
            self.window_glass = G.enable_window_glass(self.win)
            if not self.window_glass:
                self.win.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, False)
        lay = QtWidgets.QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # top bar: what is open on the left, what a drag does in the middle, background work right
        self.topbar = G.Bar("bottom")
        bars.append(self.topbar)
        self.topbar.setFixedHeight(46)
        tl = QtWidgets.QHBoxLayout(self.topbar)
        tl.setContentsMargins(16, 0, 12, 0)
        tl.setSpacing(10)
        src = cap.meta.get("path")
        self.title = QtWidgets.QLabel(title or (os.path.basename(str(src)) if src else "Capture"))
        self.title.setFont(G.ui_font(10.5, QtGui.QFont.Weight.DemiBold))
        n_ch = sum(r.kind == "ch" for r in self.rows)
        info = f"{n_ch} channel{'s' if n_ch != 1 else ''}, {format_rate(self.rate)}, {format_time(cap.duration)}"
        self.info = QtWidgets.QLabel(info)
        self.info.setFont(G.ui_font(9.5))
        self.info.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        left = QtWidgets.QHBoxLayout()
        left.setSpacing(10)
        left.addWidget(self.title, 0)
        left.addWidget(self.info, 1)
        self.mode_switch = G.LiquidSwitch([("Zoom", "zoom"), ("Pan", "pan"), ("Measure", "measure")])
        self.mode_switch.setToolTip("What a left-drag does (Z, P, M)")
        self.mode_switch.changed.connect(lambda i: self.set_mode(("zoom", "pan", "measure")[i], from_switch=True))
        self.chip = G.StatusChip()
        right = QtWidgets.QHBoxLayout()
        right.addStretch(1)
        right.addWidget(self.chip)
        tl.addLayout(left, 1)
        tl.addWidget(self.mode_switch, 0)
        tl.addLayout(right, 1)                        # equal sides keep the switch centred
        lay.addWidget(self.topbar)

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(None)
        self.glw.setStyleSheet("background: transparent; border: none")
        self.glw.viewport().setAutoFillBackground(False)
        lay.addWidget(self.glw, 1)

        # status bar (VS Code style): pointer readout, measurement, cursor gaps
        self.statusbar = G.Bar("top")
        bars.append(self.statusbar)
        self.statusbar.setFixedHeight(28)
        fl = QtWidgets.QHBoxLayout(self.statusbar)
        fl.setContentsMargins(14, 0, 14, 0)
        fl.setSpacing(0)
        self.labels = []
        self._label_roles = ("INK_DIM", "INK", "AMBER")
        for stretch in (5, 6, 3):
            lab = QtWidgets.QLabel("")
            lab.setFont(G.ui_font(9, tabular=True))
            lab.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
            lab.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            if self.labels:
                fl.addWidget(G.Hairline())
            fl.addWidget(lab, stretch)
            self.labels.append(lab)
        self.status, self.mstatus, self.cstatus = self.labels
        self.status.setText("Drag to zoom, right-drag to pan, right-click for more")
        lay.addWidget(self.statusbar)
        if self.window_glass:
            central.opacity = 0.86
            for b_ in bars:
                b_.opacity = 0.9

        # floating action button: the actions fan out above it on hover
        self.fab = G.FabMenu(central)
        self.btn_fit = self.fab.add("Fit all", "fit", tip="Show the whole capture (F)")
        self.btn_fit.clicked.connect(self.fit)
        self.btn_trig = self.fab.add("Go to trigger", "trigger", tip="Centre the view on the trigger (T)")
        self.btn_trig.clicked.connect(lambda: self._center(self.cap.trigger_pos / self.rate))
        self.btn_trig.setEnabled(cap.triggered)
        self.btn_cursor = self.fab.add("Add cursor", "cursor", tip="Add a cursor in the middle of the view")
        self.btn_cursor.clicked.connect(lambda: self.add_cursor(sum(self.view()) / 2))
        self.btn_clear = self.fab.add("Clear marks", "clear", tip="Remove cursors and pinned measurements (C)")
        self.btn_clear.clicked.connect(lambda: (self.clear_cursors(), self.clear_pins()))
        self.btn_shot = self.fab.add("Screenshot", "camera", tip="Save a PNG of the window",
                                     busy_labels=("Saving", "Saved"), new_group=True)
        self.btn_shot.clicked.connect(self._save_screenshot_dialog)
        self.btn_save = self.fab.add("Save range", "save", tip="Save the visible range as a capture file",
                                     prominent=True, busy_labels=("Saving", "Saved"))
        self.btn_save.clicked.connect(self._save_range_dialog)
        self.btn_theme = self.fab.add("Light theme", "sun", tip="Switch between light and dark",
                                      busy_labels=("Dark theme",), new_group=True)
        self.btn_theme.clicked.connect(lambda: self.set_theme("light" if G.is_dark() else "dark"))
        self.btn_about = self.fab.add("About", "info", tip="Author and credits")
        self.btn_about.clicked.connect(self.show_about)
        self.fab.request_backdrop = lambda: self._refresh_glass()

        self.win.setCentralWidget(central)
        central.resizeEvent = lambda ev: (QtWidgets.QWidget.resizeEvent(central, ev), self._place_overlays())
        self.win.keyPressEvent = self._on_key

        # --- name column + plot ---------------------------------------------------
        self.lvb = _LabelViewBox(self)
        # names column: a PlotItem with an invisible bottom axis, so its rows line up with
        # the waveform plot above the time ruler
        self.lplot = pg.PlotItem(viewBox=self.lvb)
        self.lplot.hideAxis("top")
        self.lplot.hideAxis("left")
        self.lplot.showAxis("bottom")
        lax = self.lplot.getAxis("bottom")
        lax.setStyle(showValues=False, tickLength=0)
        lax.setPen(pg.mkPen(0, 0, 0, 0))
        lax.setTicks([[], []])
        self.lplot.hideButtons()
        self.glw.addItem(self.lplot, row=0, col=0)
        self.glw.ci.layout.setColumnFixedWidth(0, self.name_w)
        self.col_handle = _ColumnHandle(central, self)
        self.glw.ci.layout.setHorizontalSpacing(0)
        self.vb = _WaveViewBox(self)
        self.plot = pg.PlotItem(viewBox=self.vb, axisItems={"bottom": _TimeAxis("bottom")})
        self.glw.addItem(self.plot, row=0, col=1)
        self.plot.hideAxis("top")
        self.plot.hideAxis("left")
        self.plot.showAxis("bottom")
        tax = self.plot.getAxis("bottom")
        tax.setStyle(maxTickLevel=0, tickTextOffset=5)   # major ticks / grid lines only
        tax.setGrid(22)
        tax.setTickFont(G.ui_font(8.5, tabular=True))
        tax.setTextPen(pg.mkPen(G.INK_DIM))
        tax.setPen(pg.mkPen(G.BORDER))
        self.glw.ci.layout.setContentsMargins(0, 0, 10, 2)
        self.lvb.setYLink(self.vb)
        lax.setHeight(tax.height())
        self._pad_timer = QtCore.QTimer()
        self._pad_timer.setSingleShot(True)
        self._pad_timer.timeout.connect(lambda: (self._update_padding(), self._place_overlays()))
        self.vb.sigResized.connect(lambda *_: (self._update_padding(), self._pad_timer.start(0)))
        self.lvb.sigResized.connect(lambda *_: self._pad_timer.start(0))     # column resized
        self.plot.hideButtons()
        self.plot.setMenuEnabled(False)
        self.lplot.setMenuEnabled(False)
        pad = (self.full[1] - self.full[0]) * 0.025
        self.vb.setLimits(xMin=self.full[0] - pad, xMax=self.full[1] + pad,
                          minXRange=16 / self.rate, maxXRange=(self.full[1] - self.full[0]) * 1.05)

        if cap.triggered:
            tx = cap.trigger_pos / self.rate
            self._trig_line = pg.InfiniteLine(tx, angle=90, pen=pg.mkPen(G.CORAL, width=1.2,
                                                                         style=QtCore.Qt.PenStyle.DashLine),
                                              label="T", labelOpts={"position": 0.97, "color": "#FFFFFF",
                                                                   "fill": G.CORAL})
            self.plot.addItem(self._trig_line)
        self._pulse_curve = pg.PlotCurveItem(pen=pg.mkPen(G.ORANGE, width=3))
        self._pulse_curve.setZValue(50)
        self.plot.addItem(self._pulse_curve, ignoreBounds=True)
        self._pulse_trace = pg.PlotCurveItem()           # the hovered pulse, drawn bold
        self._pulse_dim = pg.PlotCurveItem(connect="pairs")
        self._pulse_tag = _Tag()
        for z, it in ((55, self._pulse_trace), (56, self._pulse_dim), (57, self._pulse_tag)):
            it.setZValue(z)
            it.setOpacity(0.0)
            it.hide()
            self.plot.addItem(it, ignoreBounds=True)
        self._pulse_shown = False
        self._pulse_alpha = 0.0
        self._last_hover = None
        self._hover_painted = QtGui.QRegion()
        self._hover_gen = 0
        self._hover_pool = ThreadPoolExecutor(max_workers=1)
        self._hover_sig = G._Signal()
        self._hover_sig.sig.connect(self._hover_result)
        self._rehover_timer = QtCore.QTimer()
        self._rehover_timer.setSingleShot(True)
        self._rehover_timer.timeout.connect(self._rehover)
        self.glw.leaveEvent = lambda ev: (pg.GraphicsView.leaveEvent(self.glw, ev), self._hover_off(),
                                          self._clear_name_hover())
        self._pulse_pop = G._Anim(self.glw, "_pulse_alpha", 140, target=self,      # fades all three together
                                  on_change=self._set_pulse_opacity)
        self._hover_line = pg.InfiniteLine(angle=90, pen=pg.mkPen(255, 255, 255, 40))
        self.plot.addItem(self._hover_line, ignoreBounds=True)
        # hovering a decoded word: its time span lights up and the samples it was read from
        # are marked on the traces
        self._word_band = pg.LinearRegionItem(movable=False, brush=pg.mkBrush(10, 132, 255, 30),
                                              pen=pg.mkPen(None))
        for ln in self._word_band.lines:
            ln.setPen(pg.mkPen(10, 132, 255, 140))
        self._word_band.setZValue(-50)
        self._word_band.hide()
        self.plot.addItem(self._word_band, ignoreBounds=True)
        self._bit_dots = pg.ScatterPlotItem(size=7, pen=pg.mkPen(G.SURFACE, width=1.5), brush=pg.mkBrush("#FFFFFF"))
        self._bit_dots.setZValue(60)
        self.plot.addItem(self._bit_dots, ignoreBounds=True)
        self._hover_word = None
        self._insert_line = pg.InfiniteLine(angle=0, pen=pg.mkPen(G.ACCENT, width=2))
        self._row_rules = []
        self._insert_line.hide()
        self.plot.addItem(self._insert_line, ignoreBounds=True)
        self._pin_items = []

        # --- per-row items (created once, data swapped in) -------------------------
        self._name_items = []
        for r in self.rows:
            if r.kind == "ch":
                pen = pg.mkPen(r.color, width=1)       # 1 px: Qt's fast line path
                hi = pg.PlotCurveItem(pen=pen)
                lo = pg.PlotCurveItem(pen=pen)
                act = pg.PlotCurveItem(pen=pen, connect="pairs")
                for it in (hi, lo, act):
                    self.plot.addItem(it)
                r.items = (hi, lo, act)
            else:
                cc = QtGui.QColor(r.color)
                fill, edge = QtGui.QColor(cc), QtGui.QColor(cc)
                fill.setAlpha(40)
                edge.setAlpha(220)
                cells = QtWidgets.QGraphicsPathItem()
                pen = QtGui.QPen(edge, 1.2)
                pen.setCosmetic(True)
                pen.setJoinStyle(QtCore.Qt.PenJoinStyle.MiterJoin)
                cells.setPen(pen)
                cells.setBrush(QtGui.QBrush(fill))
                ticks = QtWidgets.QGraphicsPathItem()   # bit boundaries inside the words
                tp = QtGui.QPen(QtGui.QColor(cc.red(), cc.green(), cc.blue(), 90), 1)
                tp.setCosmetic(True)
                ticks.setPen(tp)
                self.plot.addItem(cells, ignoreBounds=True)
                self.plot.addItem(ticks, ignoreBounds=True)
                r.items = (cells, ticks)
                r.texts = []
                r.bit_texts = []
            t = _NameCell(self)
            self.lvb.addItem(t)
            self._name_items.append(t)
        self._layout_rows()

        self._pin_timer = QtCore.QTimer()
        self._pin_timer.setSingleShot(True)
        self._pin_timer.timeout.connect(self._fire_pin)
        self.vb.sigXRangeChanged.connect(lambda *_: (self._request(), self._schedule_glass(),
                                                     self._rehover_timer.start(60)))
        self.plot.scene().sigMouseMoved.connect(self._on_move)
        self.win.closeEvent = self._on_close
        self.set_mode("zoom")
        self._glass_timer = QtCore.QTimer()
        self._glass_timer.setSingleShot(True)
        self._glass_timer.timeout.connect(self._refresh_glass)
        self._place_overlays()
        self.set_view(start if start is not None else self.full[0], end if end is not None else self.full[1])
        self._ov_signal = G._Signal()
        self._ov_signal.sig.connect(self._on_overview_progress)
        self._ov_timer = QtCore.QTimer()
        self._ov_timer.setSingleShot(True)
        self._ov_timer.timeout.connect(self._request)
        self._ov.build([r.ch for r in self.rows if r.kind == "ch"] + [r.ch for r, _ in self._hidden if r.kind == "ch"],
                       background=True, on_progress=lambda f: self._ov_signal.sig.emit(f))
        self._apply_theme()
        try:                                   # follow the OS until a theme is picked here
            self.app.styleHints().colorSchemeChanged.connect(self._system_theme_changed)
        except Exception:
            pass
        self._request()

    # --- theme -------------------------------------------------------------------------
    def set_theme(self, name: str, animate: bool = True) -> None:
        """"light" or "dark" (the theme item on the action button)."""
        self._theme_auto = False
        self._switch_theme(name, animate)

    def _system_theme_changed(self, *_):
        if self._theme_auto:
            self._switch_theme(G.system_theme(), True)

    def _switch_theme(self, name: str, animate: bool) -> None:
        if name == G.THEME:
            return
        fade = self.win.grab() if animate and self.win.isVisible() else None
        G.set_theme(name)
        self._apply_theme()
        if fade is not None:
            G.CrossFade(self.win, fade)

    def _apply_theme(self) -> None:
        """Restyle everything for the current G theme."""
        dark = G.is_dark()
        self.app.setStyleSheet(f"QToolTip{{background:{G.RAISED};color:{G.INK};border:1px solid {G.MENU_BORDER};"
                               "padding:4px 6px;border-radius:4px}")
        self.title.setStyleSheet(f"color:{G.INK}")
        self.info.setStyleSheet(f"color:{G.INK_DIM}")
        for lab, role in zip(self.labels, self._label_roles):
            lab.setStyleSheet(f"color:{getattr(G, role)}; background: transparent")
        self.btn_theme.setText("Light theme" if dark else "Dark theme")
        self.btn_theme.icon_name = "sun" if dark else "moon"
        self.fab._layout()
        tax = self.plot.getAxis("bottom")
        tax.setTextPen(pg.mkPen(G.INK_DIM))
        tax.setPen(pg.mkPen(G.BORDER))
        tax.setGrid(22 if dark else 30)
        acc = QtGui.QColor(G.ACCENT)
        self.vb._band.setBrush(pg.mkBrush(acc.red(), acc.green(), acc.blue(), 40))
        self._word_band.setBrush(pg.mkBrush(acc.red(), acc.green(), acc.blue(), 30 if dark else 22))
        for ln in self._word_band.lines:
            ln.setPen(pg.mkPen(acc.red(), acc.green(), acc.blue(), 140))
        self._insert_line.setPen(pg.mkPen(G.ACCENT, width=2))
        self._hover_line.setPen(pg.mkPen(G.line(40)))
        self._bit_dots.setPen(pg.mkPen(G.SURFACE, width=1.5))
        self._bit_dots.setBrush(pg.mkBrush(G.INK))
        self._pulse_curve.setPen(pg.mkPen(G.ORANGE, width=3))
        self._style_pulse_items()
        if getattr(self, "_trig_line", None) is not None:
            self._trig_line.setPen(pg.mkPen(G.CORAL, width=1.2, style=QtCore.Qt.PenStyle.DashLine))
            self._trig_line.label.fill = pg.mkBrush(G.CORAL)
            self._trig_line.label.update()
        for r in self.rows + [h[0] for h in self._hidden]:
            col = G.theme_color(r.color)
            if r.kind == "ch":
                for it in r.items:
                    it.setPen(pg.mkPen(col, width=1))
            else:
                cells, ticks = r.items
                fill, edge, tk = QtGui.QColor(col), QtGui.QColor(col), QtGui.QColor(col)
                fill.setAlpha(40 if dark else 30)
                edge.setAlpha(220)
                tk.setAlpha(90)
                pen = QtGui.QPen(edge, 1.2)
                pen.setCosmetic(True)
                cells.setPen(pen)
                cells.setBrush(QtGui.QBrush(fill))
                tp = QtGui.QPen(tk, 1)
                tp.setCosmetic(True)
                ticks.setPen(tp)
        for k, c in enumerate(self.cursors):
            c.restyle(G.theme_color(c.base_color))
        self.central.update()
        for w in (self.topbar, self.statusbar, self.chip, self.mode_switch, self.fab.fab, *self.fab.items):
            w.update()
        G.set_window_dark(self.win, dark)
        self._layout_rows()                     # rules, decoder labels, pins, name cells
        self._schedule_glass()

    # --- layout ------------------------------------------------------------------
    @staticmethod
    def _base(i: int) -> float:
        return -ROW * i

    def _layout_rows(self):
        n = len(self.rows)
        self._update_padding()
        nums = [str(n) for r in self.rows for n in (r.ch, getattr(r, "src", None)) if n is not None]
        fm = QtGui.QFontMetricsF(G.ui_font(9, QtGui.QFont.Weight.DemiBold, tabular=True))
        badge_w = max(26.0, fm.horizontalAdvance("0" * max(map(len, nums), default=1)) + 14)
        for i, (r, t) in enumerate(zip(self.rows, self._name_items)):
            t.set_row(r, badge_w)
            t.setPos(NAME_X, self._base(i) + 0.5)
        for it in self._row_rules:
            self.plot.removeItem(it)
        self._row_rules = []
        for i in range(1, n):                 # faint rules between rows
            rule = pg.InfiniteLine(self._base(i) + 1.2, angle=0, pen=pg.mkPen(G.line(12)))
            self.plot.addItem(rule, ignoreBounds=True)
            self._row_rules.append(rule)
        for i, r in enumerate(self.rows):
            if r.key in self._results:
                self._apply_row(i, r, self._results[r.key])
        self._draw_pins()

    def _pads(self) -> tuple[float, float]:
        """Pixels left empty above the first row and below the last."""
        return 14.0, 14.0

    def _update_padding(self):
        """Fit all rows in the plot height."""
        n = len(self.rows)
        y0, y1 = self._base(n - 1) - 0.25, 1.3
        top, bottom = self._pads()
        h = max(60.0, self.vb.height() - top - bottom)
        dpp = (y1 - y0) / h
        lo, hi = y0 - bottom * dpp, y1 + top * dpp
        self.vb.setLimits(yMin=lo, yMax=hi)
        self.vb.setYRange(lo, hi, padding=0)
        cell_h = max(18.0, min(ROW / dpp - 4, 64.0))
        for t in getattr(self, "_name_items", ()):
            t.set_size(max(60.0, self.lvb.width()), cell_h)
        self._label_frac = 0.015                     # cursor numbers along the bottom
        for c in self.cursors:
            c.label.setPosition(self._label_frac)
        self._schedule_glass()

    # --- name column width ---------------------------------------------------------------
    @staticmethod
    def _settings() -> QtCore.QSettings:
        return QtCore.QSettings("dslogic", "viewer")

    def _load_name_width(self) -> int:
        try:
            w = int(self._settings().value("name_width", NAME_W))
        except (TypeError, ValueError):
            w = NAME_W
        return int(min(max(w, NAME_W_RANGE[0]), NAME_W_RANGE[1]))

    def save_name_width(self) -> None:
        self._settings().setValue("name_width", self.name_w)

    def set_name_width(self, w: float) -> None:
        """Width of the name column in pixels (clamped to NAME_W_RANGE)."""
        w = int(min(max(w, NAME_W_RANGE[0]), NAME_W_RANGE[1]))
        if w == self.name_w:
            return
        self.name_w = w
        self.glw.ci.layout.setColumnFixedWidth(0, w)
        self.central.sidebar = w
        self.central.update()
        self._pad_timer.start(0)                 # cards and the action button follow the layout

    def _place_overlays(self):
        """Keep the floating action button in the plot's bottom-right corner and the
        column handle on the name column's edge."""
        if not hasattr(self, "vb"):
            return
        g = self.glw.geometry()
        self.col_handle.setGeometry(g.x() + self.name_w - 4, g.y(), 8,
                                    g.height() - self.plot.getAxis("bottom").height())
        self.col_handle.raise_()
        sr = self.vb.sceneBoundingRect()
        br = self.glw.mapTo(self.central, self.glw.mapFromScene(sr.bottomRight()))
        c = QtCore.QPointF(br.x() - 22 - G.Fab.SIZE / 2, br.y() - 18 - G.Fab.SIZE / 2)
        self.fab.place(c)
        self._schedule_glass()

    def _schedule_glass(self):
        if hasattr(self, "_glass_timer") and not self._glass_timer.isActive():
            self._glass_timer.start(30)

    def _sample_backdrop(self, rect: QtCore.QRect, scale: float = 1 / 3) -> QtGui.QPixmap:
        """Low-resolution blurred copy of what lies under `rect` (surface + plot), for the glass."""
        w, h = max(1, int(rect.width() * scale)), max(1, int(rect.height() * scale))
        pm = QtGui.QPixmap(w, h)
        pm.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        p.scale(scale, scale)
        p.translate(-rect.x(), -rect.y())
        G.paint_surface(p, QtCore.QRectF(self.central.rect()), self.central.sidebar, self.central.opacity)
        g = self.glw.geometry()
        p.translate(g.x(), g.y())
        # straight from the scene: rendering the view widget itself would go through its
        # paint bookkeeping, which belongs to what is on screen
        local = QtCore.QRectF(rect.translated(-g.topLeft()))
        src = self.glw.mapToScene(local.toAlignedRect()).boundingRect()
        self.glw.scene().render(p, local, src, QtCore.Qt.AspectRatioMode.IgnoreAspectRatio)
        p.end()
        return G.blur_pixmap(pm, 4.0)

    def _refresh_glass(self):
        rect = self.fab.region()
        self.fab.set_backdrop(self._sample_backdrop(rect), rect)

    # --- drag a row to reorder: lifted card + rows sliding out of the way ---------------
    def _row_strip(self, i: int) -> QtCore.QRect:
        """Pixel rect (in glw coordinates) of row i: name column + plot."""
        top = self.vb.mapViewToScene(QtCore.QPointF(0, self._base(i) + 1.2)).y()
        bot = self.vb.mapViewToScene(QtCore.QPointF(0, self._base(i) - 0.2)).y()
        left = self.lplot.sceneBoundingRect().left()
        right = self.vb.sceneBoundingRect().right()
        pt = self.glw.mapFromScene(QtCore.QPointF(left, top))
        pb = self.glw.mapFromScene(QtCore.QPointF(right, bot))
        return QtCore.QRect(pt, pb).normalized()

    def _row_artists(self, k: int) -> list:
        return list(self.rows[k].items) + self._texts(self.rows[k])

    @staticmethod
    def _texts(r) -> list:
        return list(getattr(r, "texts", [])) + list(getattr(r, "bit_texts", []))

    def _shift_row(self, k: int, dy: float) -> None:
        """Draw row k dy (view units) away from where its data puts it."""
        rd = self._rd
        for it in self.rows[k].items:                 # curves / bars: plain offset
            it.setPos(0, dy)
        for t in self._texts(self.rows[k]):           # labels: their pos is their location
            p0 = rd["tpos"].get(id(t))
            if p0 is not None:
                t.setPos(p0.x(), p0.y() + dy)
        self._name_items[k].setPos(NAME_X, self._base(k) + 0.5 + dy)

    def row_drag_start(self, src: int, scene_pos) -> None:
        strip = self._row_strip(src)
        inset = 10                                    # keep the card's rounded ends on screen
        strip = strip.adjusted(inset, 0, -inset, 0)
        snap = self.glw.grab(strip)
        card = G.DragCard(self.central, snap)
        self._rd = {"src": src, "live": list(range(len(self.rows))), "off": [0.0] * len(self.rows),
                    "card": card, "grab_dy": self.glw.mapFromScene(scene_pos).y() - strip.top(),
                    "x": self.glw.mapTo(self.central, strip.topLeft()).x(), "done": False,
                    "tpos": {id(t): QtCore.QPointF(t.pos()) for r in self.rows for t in self._texts(r)}}
        for it in self._row_artists(src) + [self._name_items[src]]:
            it.setVisible(False)                      # its slot becomes the gap
        for it in self._pin_items:
            it.setVisible(False)
        self._hover_off()
        self._place_card(scene_pos)
        card.pick_up()
        self._rd_timer.start(16)

    def _place_card(self, scene_pos) -> None:
        rd = self._rd
        y = self.glw.mapTo(self.central, self.glw.mapFromScene(scene_pos)).y() - rd["grab_dy"]
        rd["card"].move(rd["x"] - G.DragCard.PAD, y - G.DragCard.PAD)

    def row_drag_move(self, scene_pos) -> None:
        rd = self._rd
        self._place_card(scene_pos)
        # the card's centre decides the slot (view y grows upward, pixels downward)
        y = self.vb.mapSceneToView(scene_pos).y() + (rd["grab_dy"] - self._row_half_px()) * self._dpp()
        slot = int(np.clip(round((0.5 - y) / ROW), 0, len(self.rows) - 1))
        live = rd["live"]
        if live.index(rd["src"]) != slot:
            live.remove(rd["src"])
            live.insert(slot, rd["src"])              # neighbours start sliding (see _row_drag_tick)

    def _dpp(self) -> float:
        (y0, y1) = self.vb.viewRange()[1]
        return (y1 - y0) / max(1.0, self.vb.height())

    def _row_half_px(self) -> float:
        return 0.7 / max(1e-9, self._dpp())

    def _row_drag_tick(self) -> None:
        rd = self._rd
        if rd is None:
            self._rd_timer.stop()
            return
        moving = False
        for k in range(len(self.rows)):
            target = rd["live"].index(k) - k
            off = rd["off"][k]
            if abs(target - off) > 1e-3:
                off += (target - off) * (1.0 if G.reduced_motion() else 0.28)
                if abs(target - off) < 0.004:
                    off = float(target)
                rd["off"][k] = off
                moving = True
                self._shift_row(k, -ROW * off)
        if not moving and rd["done"]:
            self._rd_timer.stop()

    def row_drag_finish(self) -> None:
        rd = self._rd
        slot = rd["live"].index(rd["src"])
        strip = self._row_strip(slot).adjusted(10, 0, -10, 0)    # where the gap is now
        target = self.glw.mapTo(self.central, strip.topLeft()) - QtCore.QPoint(G.DragCard.PAD, G.DragCard.PAD)
        rd["done"] = True
        rd["card"].drop(target, self._row_drag_commit)

    def _row_drag_commit(self) -> None:
        rd, self._rd = self._rd, None
        if rd is None:
            return
        order = rd["live"]
        self.rows = [self.rows[k] for k in order]
        self._name_items = [self._name_items[k] for k in order]
        for r in self.rows:
            for it in r.items:
                it.setPos(0, 0)
        src_row = self.rows[order.index(rd["src"])]
        for it in list(src_row.items):
            it.setVisible(True)
        for t in self._name_items:
            t.setVisible(True)
        self._layout_rows()                           # redraw rows at their new places
        rd["card"].hide()
        rd["card"].deleteLater()

    def move_row(self, src: int, dst: int) -> None:
        r = self.rows.pop(src)
        t = self._name_items.pop(src)
        self.rows.insert(dst, r)
        self._name_items.insert(dst, t)
        self._layout_rows()

    # --- view --------------------------------------------------------------------------
    def set_view(self, x0: float, x1: float) -> None:
        self.vb.setXRange(x0, x1, padding=0)

    def view(self) -> tuple[float, float]:
        return tuple(self.vb.viewRange()[0])

    def fit(self):
        self.set_view(*self.full)

    def zoom(self, factor: float, center: float | None = None):
        x0, x1 = self.view()
        c = (x0 + x1) / 2 if center is None else center
        self.set_view(c - (c - x0) * factor, c + (x1 - c) * factor)

    def pan(self, dt: float):
        x0, x1 = self.view()
        self.set_view(x0 + dt, x1 + dt)

    def _width(self) -> int:
        return max(100, int(self.vb.width()))

    def _px(self) -> float:
        x0, x1 = self.view()
        return (x1 - x0) / self._width()

    # --- background compute ------------------------------------------------------------
    def _request(self):
        x0, x1 = self.view()
        s0 = int(max(0, x0 * self.rate))
        s1 = int(min(self.cap.num_samples, x1 * self.rate + 1))
        with self._lock:
            self._gen += 1
            gen = self._gen
        if self._ov.progress >= 1.0:
            self.chip.busy("Rendering")
        rows = list(self.rows)
        self._worker.submit(self._job, gen, s0, s1, self._width(), rows)
        if any(r.kind == "dec" for r in rows):         # decoders never hold up the traces
            self._dec_worker.submit(self._dec_job, gen, s0, s1, rows)

    def _job(self, gen, s0, s1, w, rows):
        if gen != self._gen:
            return
        try:
            out = {}
            chans = [r for r in rows if r.kind == "ch"]
            envs = self._pool.map(lambda r: envelope(self.cap, r.ch, s0, s1, w, build=False), chans)
            for r, env in zip(chans, envs):
                out[r.key] = env
        except Exception as e:  # keep the window alive
            out = {"error": str(e)}
        if gen == self._gen:
            self._bridge.ready.emit((gen, out))

    def _dec_job(self, gen, s0, s1, rows):
        if gen != self._gen:
            return
        decs, seen = [], set()
        for r in rows:
            if r.kind == "dec" and id(r.dec) not in seen:
                seen.add(id(r.dec))
                decs.append(r.dec)
        # several decoders decode at the same time (their edge scans run outside the GIL)
        results = dict(zip(map(id, decs), self._pool.map(lambda d: self._decode_window(d, s0, s1), decs)))
        out = {r.key: ("dec",) + results[id(r.dec)] for r in rows if r.kind == "dec"}
        if gen == self._gen:
            self._bridge.ready.emit((gen, out))

    def _on_overview_progress(self, frac: float) -> None:
        """Called (via a queued signal) while the overview builds: show progress and redraw
        the zoomed-out view so it fills in left to right."""
        if frac >= 1.0:
            self.chip.ready()
            self._request()
            return
        self.chip.busy(f"Building overview {int(frac * 100)}%", delay_ms=0)
        if not self._ov_timer.isActive():
            self._ov_timer.start(250)

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
            c = (a, b, [x for x in anns if x.kind != "transfer"] or anns)
            self._decode_cache[id(d)] = c
        return ("ok", [x for x in c[2] if x.end >= s0 and x.start <= s1])

    def _apply(self, msg):
        gen, out = msg
        if gen != self._gen:
            return
        if self._ov.progress >= 1.0:
            self.chip.ready()
        if "error" in out:
            self.mstatus.setText(f"Could not draw this view: {out['error']}")
            return
        self._results.update(out)
        for i, r in enumerate(self.rows):
            if r.key in out:
                self._apply_row(i, r, out[r.key])
        self._schedule_glass()

    def _apply_row(self, i, r, data):
        base = self._base(i)
        if r.kind == "ch":
            hi, lo, act = r.items
            kind, x, *rest = data
            t = np.append(x, x[-1] + 1 if len(x) == 1 else x[-1] + (x[-1] - x[-2] if len(x) > 1 else 1)) / self.rate
            if kind == "exact":
                hi.setData(t, rest[0].astype(float) + base, stepMode="center")
                lo.setData([], [], stepMode=None)
                act.setData([], [])
            else:
                l, h = rest
                hi.setData(t, h.astype(float) + base, stepMode="center", connect="finite")
                lo.setData(t, l.astype(float) + base, stepMode="center", connect="finite")
                m = (l != h) & ~np.isnan(np.asarray(l, float))      # NaN = overview not built yet
                xs = np.repeat(t[:-1][m], 2)
                ys = np.tile([base, base + 1.0], int(m.sum()))
                act.setData(xs, ys)
            return
        self._apply_dec(i, r, data)

    def _cell(self, a) -> tuple[float, float]:
        """Samples a decoded word covers on the wire: whole bit cells around the samples it
        was read at (an SPI byte read at 8 clock edges spans those 8 bit times)."""
        b = a.bits
        if b and len(b) > 1:
            half = (b[-1] - b[0]) / (len(b) - 1) / 2
            return b[0] - half, b[-1] + half
        return float(a.start), float(max(a.end, a.start + 1))

    def _apply_dec(self, i, r, data):
        """Decoded words as bus cells (DSView / scope style hexagons) lined up with the
        bits they came from; closer in, bit boundaries and then bit values appear."""
        base = self._base(i)
        cells, ticks = r.items
        for tx in self._texts(r):
            tx.hide()
        status, items = data[1], data[2]
        if status != "ok":
            r.cells = (np.zeros((0, 2)), [])
            cells.setPath(QtGui.QPainterPath())
            ticks.setPath(QtGui.QPainterPath())
            x0, x1 = self.view()
            self._label(r.texts, 0, (x0 + x1) / 2, base + 0.5,
                        "zoom in to decode" if status == "zoom" else f"error: {items}", G.INK_DIM)
            return
        vis = [a for a in items if self._label_of(r, a)]
        rate = self.rate
        spp = max(1e-9, self._px() * rate)                     # samples per pixel
        y0, y1, ym = base + 0.16, base + 0.84, base + 0.5
        path, tick = QtGui.QPainterPath(), QtGui.QPainterPath()
        c = np.array([self._cell(a) for a in vis], float).reshape(-1, 2)
        r.cells = (c, vis)                                     # for hover lookups
        wpx = (c[:, 1] - c[:, 0]) / spp
        wide = (wpx >= 7) if len(vis) <= 20000 else np.zeros(len(vis), bool)
        # words too narrow to tell apart: one solid run per cluster, labelled with its size
        runs, run = [], None
        for a0, a1 in c[~wide]:
            if run is not None and a0 - run[1] < 1.5 * spp:
                run[1] = max(run[1], a1)
                run[2] += 1
                continue
            if run is not None:
                runs.append(run)
            run = [a0, a1, 1]
        if run is not None:
            runs.append(run)
        n_lab = 0
        for a0, a1, cnt in runs:
            a1 = max(a1, a0 + spp)
            path.addRect(QtCore.QRectF(a0 / rate, y0, (a1 - a0) / rate, y1 - y0))
            label = f"{cnt} words" if cnt > 1 else ""
            if label and (a1 - a0) / spp > 7 * len(label) + 16 and n_lab < 400:
                self._label(r.texts, n_lab, (a0 + a1) / 2 / rate, ym, label, G.INK_DIM)
                n_lab += 1
        idx = np.flatnonzero(wide)
        nbits = sum(len(vis[k].bits or ()) for k in idx)
        bit_px = min((wpx[k] / len(vis[k].bits) for k in idx if vis[k].bits), default=0.0)
        show_ticks = bit_px >= 9
        show_bits = bit_px >= 13 and nbits <= 768 and getattr(r, "src", None) is not None
        levels = {}
        if show_bits:
            pos = np.concatenate([np.asarray(vis[k].bits, np.int64) for k in idx if vis[k].bits])
            levels = dict(zip(pos.tolist(), self.cap.levels_at(r.src, pos).tolist()))
        n_bit = 0
        for k in idx:
            a = vis[k]
            a0, a1 = c[k]
            e = min(4 * spp, (a1 - a0) / 2)                     # 4 px points at both ends
            poly = QtGui.QPolygonF([QtCore.QPointF(x / rate, y) for x, y in
                                    ((a0, ym), (a0 + e, y1), (a1 - e, y1), (a1, ym), (a1 - e, y0), (a0 + e, y0))])
            path.addPolygon(poly)
            path.closeSubpath()
            b = a.bits
            if show_ticks and b and len(b) > 1:
                for m in range(1, len(b)):
                    x = (b[m - 1] + b[m]) / 2 / rate
                    tick.moveTo(x, y0)
                    tick.lineTo(x, y0 + 0.12)
            label = self._label_of(r, a)
            if wpx[k] > 8 * len(label) + 12 and n_lab < 400:
                self._label(r.texts, n_lab, (a0 + a1) / 2 / rate, base + (0.62 if show_bits else 0.5),
                            label, G.INK)
                n_lab += 1
            if show_bits and b:
                for x in b:
                    self._label(r.bit_texts, n_bit, x / rate, base + 0.31, str(levels.get(int(x), "")),
                                G.INK_DIM, small=True)
                    n_bit += 1
        cells.setPath(path)
        ticks.setPath(tick)

    def _label(self, pool, k, x, y, s, color, small=False):
        while len(pool) <= k:
            t = pg.TextItem("", anchor=(0.5, 0.5))
            t.setFont(G.ui_font(7.5, tabular=True) if small else
                      G.ui_font(9, QtGui.QFont.Weight.Medium, tabular=True))
            self.plot.addItem(t, ignoreBounds=True)
            pool.append(t)
        t = pool[k]
        t.setText(s, color=color)
        t.setPos(x, y)
        t.show()

    # --- hovering a decoded word --------------------------------------------------------
    def _hover_dec(self, i: int, s: int) -> None:
        r = self.rows[i]
        c, vis = getattr(r, "cells", (np.zeros((0, 2)), []))
        k = int(np.searchsorted(c[:, 0], s, "right")) - 1
        hit = vis[k] if 0 <= k < len(vis) and c[k, 1] >= s else None
        if hit is None:
            self._clear_word_hover()
            self.mstatus.setText("")
            return
        if hit is self._hover_word:
            return
        self._hover_word = hit
        c0, c1 = self._cell(hit)
        self._word_band.setRegion((c0 / self.rate, c1 / self.rate))
        self._word_band.show()
        xs, ys, wire = [], [], None
        if hit.bits:
            bits = np.asarray(hit.bits, np.int64)
            rows_of = {x.ch: j for j, x in enumerate(self.rows) if x.kind == "ch"}
            clk = next((r.dec.channels.get(k) for k in ("clk", "scl") if r.dec.channels.get(k) is not None), None)
            for ch in (getattr(r, "src", None), clk):
                try:
                    ch = self.cap.index(ch) if ch is not None else None
                except (KeyError, TypeError):
                    ch = None
                if ch is None:
                    continue
                lv = self.cap.levels_at(ch, bits)
                if ch == getattr(r, "src", None):
                    wire = lv
                if ch in rows_of:
                    xs.extend((bits / self.rate).tolist())
                    ys.extend((self._base(rows_of[ch]) + lv).tolist())
        self._bit_dots.setData(xs, ys)
        self.mstatus.setText(self._word_text(r, hit, c0, c1, wire))

    @staticmethod
    def _label_of(r, a) -> str:
        return _short(a, r.line, getattr(r, "fmt", "hex"), getattr(r, "wbits", 8))

    def set_row_format(self, i: int, fmt: str, all_rows: bool = False) -> None:
        """Write decoder row i's values as "hex", "dec", "oct", "bin" or "ascii"
        (all_rows: every decoder row)."""
        if fmt not in FORMATS:
            raise ValueError(f"format must be one of {FORMATS}")
        targets = [k for k, r in enumerate(self.rows) if r.kind == "dec"] if all_rows else [i]
        for k in targets:
            r = self.rows[k]
            if r.kind != "dec":
                continue
            r.fmt = fmt
            self._name_items[k].update()
            if r.key in self._results:
                self._apply_row(k, r, self._results[r.key])
        self._clear_word_hover()

    def cycle_row_format(self, i: int) -> None:
        r = self.rows[i]
        self.set_row_format(i, FORMATS[(FORMATS.index(getattr(r, "fmt", "hex")) + 1) % len(FORMATS)])

    def _format_menu(self, m, i: int) -> None:
        """'Show values as' submenu for decoder row i."""
        r = self.rows[i]
        sub = m.addMenu("Show values as")
        sub.setStyleSheet(m.styleSheet())
        sub.setWindowFlag(QtCore.Qt.WindowType.FramelessWindowHint)
        sub.setWindowFlag(QtCore.Qt.WindowType.NoDropShadowWindowHint)
        sub.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        group = QtGui.QActionGroup(sub)
        for f in FORMATS:
            a = sub.addAction(f"{FORMAT_NAMES[f]}   {format_word(0x41, f, getattr(r, 'wbits', 8))}",
                              lambda f=f: self.set_row_format(i, f))
            a.setCheckable(True)
            a.setChecked(getattr(r, "fmt", "hex") == f)
            group.addAction(a)
        sub.addSeparator()
        sub.addAction("Use for all decoder rows", lambda: self.set_row_format(i, r.fmt, all_rows=True))

    def _word_text(self, r, a, c0, c1, wire) -> str:
        v = a.data.get(r.line) if r.line else a.data.get("value", a.data.get("address"))
        if not isinstance(v, int):
            return f"{r.name}  {a.text}"
        bits = getattr(r, "wbits", 8)
        txt = f"{r.name}  0x{v:0{(bits + 3) // 4}X} = {v} = 0o{v:o}"
        if 32 <= v < 127:
            txt += f" = '{chr(v)}'"
        if wire is not None and len(wire):
            b = "".join(str(int(x)) for x in wire)
            txt += "  bits " + " ".join(b[k:k + 4] for k in range(0, len(b), 4))
        if not r.line and a.text:
            txt += f"  ({a.text})"
        return txt + f"   at {format_time_res(c0 / self.rate, self.res)}, {format_time_res((c1 - c0) / self.rate, self.res)}"

    def _clear_word_hover(self) -> None:
        if self._hover_word is not None:
            self._hover_word = None
            self._word_band.hide()
            self._bit_dots.setData([], [])

    # --- rows / hover / measurements -------------------------------------------------
    def _row_at(self, y) -> int | None:
        i = int(round((0.5 - y) / ROW))
        return i if 0 <= i < len(self.rows) else None

    def _ch_row_at(self, y) -> int | None:
        i = self._row_at(y)
        return i if i is not None and self.rows[i].kind == "ch" else None

    def _search_span(self) -> int:
        x0, x1 = self.view()
        return int(min(1 << 24, max(4096, (x1 - x0) * self.rate)))

    def _on_move(self, pos):
        """Pointer over the plot. The cheap parts (time, hover line, decoded word) update
        here; channel levels and the pulse under the pointer are looked up on a worker
        thread (they may read compressed blocks) and only the latest answer is shown."""
        if not self.vb.sceneBoundingRect().contains(pos):
            self._hover_off()
            return
        self._last_hover = QtCore.QPointF(pos)
        p = self.vb.mapSceneToView(pos)
        x, y = p.x(), p.y()
        self._hover_line.setPos(x)
        self._hover_line.show()
        self._schedule_glass()
        s = int(x * self.rate)
        txt = f"t = {format_time_res(x, self.res)}"
        if self.cap.triggered:
            d = x - self.cap.trigger_pos / self.rate
            txt += f"  (T{'+' if d >= 0 else '-'}{format_time_res(abs(d), self.res)})"
        self.status.setText(txt)                 # levels follow from the worker
        want = None
        i = self._row_at(y)
        if i is not None and self.rows[i].kind == "dec":
            self._hide_pulse()
            self._hover_dec(i, s)
        else:
            self._clear_word_hover()
            if i is None:
                self._hide_pulse()
                self.mstatus.setText("")
            else:
                key = (i, s // max(1, int(self._px() * self.rate)))
                if key != self._pulse_key:
                    self._pulse_key = key
                    want = (i, self.rows[i].ch, self.rows[i], key)
        self._hover_gen += 1
        chans = [(r.name, r.ch) for r in self.rows if r.kind == "ch"]
        self._hover_pool.submit(self._hover_job, self._hover_gen, s, txt, chans, want, self._search_span())
        self._flush_hover_paint()

    def _hover_job(self, gen, s, txt, chans, want, span):
        status_stale = gen != self._hover_gen
        pulse_stale = want is None or want[3] != self._pulse_key
        if status_stale and pulse_stale:
            return                               # the pointer has moved on
        pr = None
        try:
            if not status_stale and 0 <= s < self.cap.num_samples:
                vals = " ".join(f"{n}={int(self.cap.levels_at(ch, [s])[0])}" for n, ch in chans)
                txt += f"   sample {s}   {vals}"
            if not pulse_stale:
                pr = self.cap.pulse_at(want[1], s, max_span=span)
        except Exception:                        # keep hovering alive whatever the data does
            return
        self._hover_sig.sig.emit((gen, txt, want, pr))

    def _hover_result(self, msg):
        gen, txt, want, pr = msg
        if gen == self._hover_gen:
            self.status.setText(txt)
        if want is None or pr is None or want[3] != self._pulse_key:
            return
        i, _, row, _ = want
        if i >= len(self.rows) or self.rows[i] is not row:
            return
        if pr["width"] is None:
            self._hide_pulse(keep_key=True)
            self.mstatus.setText(f"{pr['name']}: {'high' if pr['level'] else 'low'}, no edge on both "
                                 "sides within the view (zoom out)")
            self._flush_hover_paint()
            return
        self._show_pulse(i, pr)
        self.mstatus.setText(_pulse_text(pr))
        self._flush_hover_paint()

    def _hover_items(self):
        return (self._pulse_trace, self._pulse_dim, self._pulse_tag, self._hover_line,
                self._bit_dots, self._word_band)

    def _flush_hover_paint(self) -> None:
        """Repaint where the hover visuals were and where they are now, with a margin.
        Qt's own minimal update can miss a pixel or two of a moving highlight (fractional
        display scaling, antialiased glow), which then stays on screen until the view
        moves; an explicit viewport update of both areas never leaves anything behind."""
        vt = self.glw.viewportTransform()
        now = QtGui.QRegion()
        for it in self._hover_items():
            if it.isVisible() and it.opacity() > 0:
                r = it.deviceTransform(vt).mapRect(it.boundingRect()).toAlignedRect()
                now = now.united(r.adjusted(-6, -6, 6, 6))
        now = now.intersected(self.glw.viewport().rect())
        dirty = now.united(self._hover_painted)
        if not dirty.isEmpty():
            self.glw.viewport().update(dirty)
        self._hover_painted = now

    def _clear_name_hover(self) -> None:
        for c in self._name_items:
            if c.hover:
                c.hover = False
                c.update()

    def _hover_off(self) -> None:
        """Pointer left the plot: nothing stays highlighted."""
        self._last_hover = None
        self._hover_gen += 1
        self._hover_line.hide()
        self._hide_pulse()
        self._clear_word_hover()
        self._flush_hover_paint()

    def _rehover(self) -> None:
        """The view moved under a still pointer (keys, wheel, Fit ...): hover again."""
        if self._last_hover is not None and self._rd is None:
            self._pulse_key = None
            self._clear_word_hover()
            self._on_move(self._last_hover)

    # --- hovering a pulse: it lights up on the trace with its size written across it ---------
    def _style_pulse_items(self) -> None:
        self._pulse_dim.setPen(pg.mkPen(G.ORANGE, width=1.4))
        self._pulse_tag.set_colors(G.RAISED, G.ORANGE)

    def _show_pulse(self, i: int, pr: dict) -> None:
        base, rate = self._base(i), self.rate
        a, b = pr["start"] / rate, pr["end"] / rate
        hi, lo = base + 1.0, base
        on, off = (hi, lo) if pr["level"] else (lo, hi)
        col = QtGui.QColor(G.theme_color(self.rows[i].color))
        self._pulse_trace.setPen(pg.mkPen(col, width=3))
        glow = QtGui.QColor(col)
        glow.setAlpha(70)
        self._pulse_trace.setShadowPen(pg.mkPen(glow, width=8))
        self._pulse_trace.prepareGeometryChange()          # its glow is wider than its data
        self._pulse_trace.setData([a, a, b, b], [off, on, on, off])
        mid, k = base + 0.5, 0.14                       # dimension line with end ticks
        self._pulse_dim.setData([a, b, a, a, b, b], [mid, mid, mid - k, mid + k, mid - k, mid + k])
        txt = f"<b>{format_time_res(pr['width'], self.res)}</b>"
        if pr["frequency"]:
            txt += (f'&nbsp;&nbsp;<span style="color:{G.INK_DIM}">{_fmt_hz(pr["frequency"])}'
                    f'&nbsp;&nbsp;{pr["duty"] * 100:.1f}%</span>')
        tag = self._pulse_tag
        tag.setHtml(f'<span style="color:{G.INK}">{txt}</span>', G.ui_font(8.5, tabular=True))
        w_px = (b - a) / max(1e-30, self._px())
        if w_px > tag.boundingRect().width() + 16:      # fits: written across the pulse
            tag.setAnchor((0.5, 0.5))
            tag.setPos((a + b) / 2, mid)
        else:                                           # narrow pulse: above it
            tag.setAnchor((0.5, 1.0))
            tag.setPos((a + b) / 2, base + 1.1)
        if not self._pulse_shown:
            self._pulse_shown = True
            self._pulse_pop.to(1.0)

    def _hide_pulse(self, keep_key: bool = False) -> None:
        if not keep_key:
            self._pulse_key = None
        if self._pulse_shown:
            self._pulse_shown = False
            self._pulse_pop.to(0.0, 90)

    def _set_pulse_opacity(self) -> None:
        o = self._pulse_alpha
        for it in (self._pulse_trace, self._pulse_dim, self._pulse_tag):
            it.setOpacity(o)
            it.setVisible(o > 0.01)
        self._flush_hover_paint()

    def _snap(self, i: int, t: float) -> float:
        ch = self.rows[i].ch
        s = int(min(max(t * self.rate, 0), self.cap.num_samples - 1))
        lim = int(8 * self._px() * self.rate) + 2
        cands = [e for e in (self.cap.prev_edge(ch, s, lim), self.cap.next_edge(ch, s, lim)) if e is not None]
        best = min(cands, key=lambda e: abs(e - s), default=None)
        return best / self.rate if best is not None else t

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

    def _measure_drag(self, i, xa, xb, finish):
        a, b = sorted((self._snap(i, xa), self._snap(i, xb)))
        txt = self._measure_text(i, a, b)
        self.mstatus.setText(txt)
        if finish:
            self._pulse_curve.setData([], [])
            self.pins.append(dict(key=self.rows[i].key, t0=a, t1=b, label=txt.split("  ", 1)[1],
                                  color=G.ORANGE))
            self._draw_pins()
        else:
            yb = self._base(i) + 1.12
            self._pulse_curve.setData([a, b], [yb, yb])

    def _fire_pin(self):
        if self._pending_pin is None:
            return
        x, y = self._pending_pin
        self._pending_pin = None
        i = self._ch_row_at(y)
        if i is None:
            return
        p = self.cap.pulse_at(self.rows[i].ch, int(x * self.rate), max_span=self._search_span())
        if p["width"] is None:
            return
        label = f"{'H' if p['level'] else 'L'} {format_time(p['width'])}"
        if p["frequency"]:
            label += f"  {_fmt_hz(p['frequency'])}  {p['duty'] * 100:.1f}%"
        self.pins.append(dict(key=self.rows[i].key, t0=p["start"] / self.rate, t1=p["end"] / self.rate,
                              label=label, color=G.AMBER))
        self._draw_pins()

    def _draw_pins(self):
        for it in self._pin_items:
            self.plot.removeItem(it)
        self._pin_items = []
        index = {r.key: i for i, r in enumerate(self.rows)}
        for p in self.pins:
            i = index.get(p["key"])
            if i is None:
                continue
            yb = self._base(i) + 1.12
            c = pg.PlotCurveItem([p["t0"], p["t1"]], [yb, yb], pen=pg.mkPen(p["color"], width=2))
            t = pg.TextItem(p["label"], color=p["color"], anchor=(0.5, 1.0),
                            fill=pg.mkBrush(G.RAISED), border=pg.mkPen(G.line(30)))
            t.setFont(G.ui_font(8.5, tabular=True))
            t.setPos((p["t0"] + p["t1"]) / 2, yb + 0.05)
            for it in (c, t):
                self.plot.addItem(it, ignoreBounds=True)
                self._pin_items.append(it)

    # --- cursors -------------------------------------------------------------------------
    def add_cursor(self, x: float) -> None:
        c = _Cursor(self, x, CURSOR_COLORS[len(self.cursors) % len(CURSOR_COLORS)], len(self.cursors) + 1)
        self.plot.addItem(c, ignoreBounds=True)
        self.cursors.append(c)
        c.label.setPosition(getattr(self, "_label_frac", 0.05))
        self._cursor_status()

    def _remove_cursor(self, c):
        self.plot.removeItem(c)
        self.cursors.remove(c)
        self._cursor_status()

    def _cursor_status(self):
        xs = sorted(c.value() for c in self.cursors)
        parts = [format_time_res(b - a, self.res) + (f" ({_fmt_hz(1 / (b - a))})" if b > a else "")
                 for a, b in list(zip(xs, xs[1:]))[-3:]]
        self.cstatus.setText(("Cursor gaps  " + "   ".join(parts)) if parts else "")
        self._schedule_glass()

    # --- right-click menus ----------------------------------------------------------------
    def _menu(self):
        m = QtWidgets.QMenu(self.win)
        m.setWindowFlag(QtCore.Qt.WindowType.FramelessWindowHint)
        m.setWindowFlag(QtCore.Qt.WindowType.NoDropShadowWindowHint)
        m.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        m.setFont(G.ui_font(9.5))
        m.setStyleSheet(f"QMenu{{background:{G.RAISED};color:{G.INK};border:1px solid {G.MENU_BORDER};"
                        "border-radius:8px;padding:5px}"
                        "QMenu::item{padding:5px 26px 5px 12px;border-radius:5px}"
                        f"QMenu::item:selected{{background:{G.ACCENT};color:white}}"
                        f"QMenu::item:disabled{{color:{G.INK_FAINT}}}"
                        f"QMenu::separator{{height:1px;background:{G.MENU_BORDER};margin:4px 8px}}")
        return m

    @staticmethod
    def _popup(menu, screen_pos, show):
        if show and not menu.isEmpty():
            menu.popup(QtCore.QPoint(int(screen_pos.x()), int(screen_pos.y())))
        return menu

    def context_menu(self, screen_pos, x: float, y: float, show: bool = True):
        """Right-click menu on the waveform area."""
        m = self._menu()
        i = self._ch_row_at(y)
        m.addAction("Fit all", self.fit)
        a = m.addAction("Go to trigger", lambda: self._center(self.cap.trigger_pos / self.rate))
        a.setEnabled(self.cap.triggered)
        m.addAction("Zoom in here", lambda: self.zoom(0.5, x))
        m.addAction("Zoom out here", lambda: self.zoom(2.0, x))
        m.addSeparator()
        m.addAction(f"Add cursor at {format_time_res(x, self.res)}", lambda: self.add_cursor(x))
        a = m.addAction("Pin pulse measurement here", lambda: self._pin_here(x, y))
        a.setEnabled(i is not None)
        m.addAction("Copy time", lambda: QtWidgets.QApplication.clipboard().setText(format_time_res(x, self.res)))
        j = self._row_at(y)
        if j is not None and self.rows[j].kind == "dec":
            self._format_menu(m, j)
        m.addSeparator()
        a = m.addAction("Clear cursors", self.clear_cursors)
        a.setEnabled(bool(self.cursors))
        a = m.addAction("Clear measurements", self.clear_pins)
        a.setEnabled(bool(self.pins))
        m.addSeparator()
        a = m.addAction(f"Show hidden rows ({len(self._hidden)})", self.show_hidden_rows)
        a.setEnabled(bool(self._hidden))
        m.addAction("Save screenshot...", self._save_screenshot_dialog)
        m.addAction("Save visible range as...", self._save_range_dialog)
        m.addSeparator()
        m.addAction("About dslogic-tool...", self.show_about)
        return self._popup(m, screen_pos, show)

    def cursor_menu(self, cursor, screen_pos, show: bool = True):
        m = self._menu()
        n = self.cursors.index(cursor) + 1 if cursor in self.cursors else "?"
        m.addAction(f"Delete cursor {n}", lambda: self._remove_cursor(cursor))
        m.addAction("Delete all cursors", self.clear_cursors)
        return self._popup(m, screen_pos, show)

    def row_menu(self, i, screen_pos, show: bool = True):
        m = self._menu()
        if i is not None:
            name = self.rows[i].name
            m.addAction("Rename...", lambda: self.start_rename(i))
            if self.rows[i].kind == "dec":
                self._format_menu(m, i)
            m.addAction(f"Hide '{name}'", lambda: self.hide_row(i))
            a = m.addAction("Move to top", lambda: self.move_row(i, 0))
            a.setEnabled(i > 0)
            a = m.addAction("Move to bottom", lambda: self.move_row(i, len(self.rows) - 1))
            a.setEnabled(i < len(self.rows) - 1)
            m.addSeparator()
        a = m.addAction(f"Show hidden rows ({len(self._hidden)})", self.show_hidden_rows)
        a.setEnabled(bool(self._hidden))
        return self._popup(m, screen_pos, show)

    def show_about(self, tab: int = 0, show: bool = True) -> "_AboutSheet":
        """The About sheet: author (tab 0) and open-source credits (tab 1)."""
        sheet = _AboutSheet(self.win)
        sheet.tabs.set_index(tab, emit=False)
        sheet.tabs._pos_a.to(float(tab), 0)          # open on the tab, no slide
        sheet.tabs._lead_a.to(float(tab), 0)
        sheet.pages.setCurrentIndex(tab)
        sheet.adjustSize()
        g = self.win.geometry()
        sheet.move(g.center().x() - sheet.width() // 2, g.center().y() - sheet.height() // 2)
        if show:
            sheet.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
            sheet.show()
        self._about = sheet
        return sheet

    # --- renaming ---------------------------------------------------------------------------
    def rename_row(self, i: int, name: str) -> None:
        """Rename row i. For a channel this renames the channel in the capture (so a saved
        range keeps it), keeps decoders that refer to it by name working, and updates the
        decoder rows that show its name."""
        name = name.strip()
        r = self.rows[i]
        if not name or name == r.name:
            return
        if r.kind == "ch":
            old = self.cap.channels[r.ch].name
            self.cap.set_name(r.ch, name)
            for d in self.cap.decoders:
                for k, v in list(d.channels.items()):
                    if v == old:
                        d.channels[k] = name
            for x in self.rows + [h[0] for h in self._hidden]:
                if x.kind == "dec" and getattr(x, "src", None) == r.ch and x.sub.startswith(old):
                    x.sub = name + x.sub[len(old):]
        r.name = name
        self._layout_rows()

    def start_rename(self, i: int) -> None:
        """Edit row i's name in place: Enter (or clicking anywhere else) keeps it, Esc cancels."""
        self.finish_rename(True)
        self._clear_name_hover()
        cell = self._name_items[i]
        col = self.lvb.sceneBoundingRect()                  # the name column, in scene pixels
        y = self.lvb.mapViewToScene(QtCore.QPointF(0, self._base(i) + 0.5)).y()
        tl = self.glw.mapTo(self.central, self.glw.mapFromScene(QtCore.QPointF(col.left(), y)))
        br = self.glw.mapTo(self.central, self.glw.mapFromScene(QtCore.QPointF(col.right(), y)))
        x0 = tl.x() + 8 + cell.badge_w + 12
        ed = QtWidgets.QLineEdit(self.rows[i].name, self.central)
        ed.setFont(G.ui_font(10, QtGui.QFont.Weight.Medium))
        ed.setStyleSheet(f"QLineEdit{{background:{G.RAISED};color:{G.INK};border:1.5px solid {G.ACCENT};"
                         f"border-radius:6px;padding:1px 5px;selection-background-color:{G.ACCENT};"
                         "selection-color:white}")
        h = 26
        ed.setGeometry(int(x0), int(tl.y() - h / 2), int(br.x() - 12 - x0), h)
        ed.selectAll()
        ed.show()
        ed.setFocus()
        ed.returnPressed.connect(lambda: self.finish_rename(True))
        ed.editingFinished.connect(lambda: self.finish_rename(True))
        ed.keyPressEvent = lambda ev, ed=ed: (self.finish_rename(False) if ev.key() == QtCore.Qt.Key.Key_Escape
                                               else QtWidgets.QLineEdit.keyPressEvent(ed, ev))
        self._rename = (i, ed)
        self._click_away = _ClickAway(ed, lambda: self.finish_rename(True))
        self.app.installEventFilter(self._click_away)

    def finish_rename(self, keep: bool) -> None:
        cur = getattr(self, "_rename", None)
        if cur is None:
            return
        self._rename = None
        i, ed = cur
        ca = getattr(self, "_click_away", None)
        if ca is not None:
            self.app.removeEventFilter(ca)
            self._click_away = None
        if keep and i < len(self.rows):
            self.rename_row(i, ed.text())
        ed.hide()
        ed.deleteLater()
        self.win.setFocus()

    def _center(self, t):
        x0, x1 = self.view()
        self.set_view(t - (x1 - x0) / 2, t + (x1 - x0) / 2)

    def _pin_here(self, x, y):
        self._pending_pin = (x, y)
        self._fire_pin()

    def clear_cursors(self):
        for c in list(self.cursors):
            self._remove_cursor(c)

    def clear_pins(self):
        self.pins = []
        self._draw_pins()

    def hide_row(self, i: int) -> None:
        if len(self.rows) <= 1:
            return
        r, t = self.rows.pop(i), self._name_items.pop(i)
        for it in list(r.items) + self._texts(r):
            it.hide()
        t.hide()
        self._hidden.append((r, t))
        self._layout_rows()

    def show_hidden_rows(self) -> None:
        for r, t in self._hidden:
            for it in r.items:
                it.show()
            t.show()
            self.rows.append(r)
            self._name_items.append(t)
        self._hidden = []
        self._layout_rows()
        self._request()

    def set_mode(self, mode: str, from_switch: bool = False) -> None:
        """What a left-drag does: "zoom", "pan" or "measure"."""
        self.mode = mode
        if not from_switch:
            self.mode_switch.set_index(("zoom", "pan", "measure").index(mode), emit=False)
        self.vb.setCursor(QtCore.Qt.CursorShape.OpenHandCursor if mode == "pan" else
                          QtCore.Qt.CursorShape.CrossCursor)

    def _save_screenshot_dialog(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self.win, "Save screenshot", "view.png",
                                                        "PNG image (*.png)")
        if not path:
            return
        image = self.win.grab().toImage()        # grab on the GUI thread, write in the background

        def write():
            if not image.save(path):
                raise OSError(f"cannot write {path}")

        self.btn_shot.run_busy(write, label="Saving", done_label="Saved",
                               on_error=lambda e: self.mstatus.setText(f"Screenshot not saved: {e}"))

    def save_range(self, path) -> None:
        """Save the visible time range (visible channel rows and their decoders)."""
        x0, x1 = self.view()
        self._save_range(path, x0, x1, [r.ch for r in self.rows if r.kind == "ch"])

    def _save_range(self, path, x0, x1, chans) -> None:
        s0 = int(max(0, x0 * self.rate))
        s1 = int(min(self.cap.num_samples, x1 * self.rate + 1))
        part = self.cap.slice(s0, s1, channels=chans)
        part.decoders = [d for d in self.cap.decoders]
        part.save(path)

    def _save_range_dialog(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.win, "Save visible range", "range.dsl",
            "DSView (*.dsl);;sigrok (*.sr);;VCD (*.vcd);;CSV (*.csv);;numpy (*.npz)")
        if not path:
            return
        x0, x1 = self.view()
        chans = [r.ch for r in self.rows if r.kind == "ch"]
        self.btn_save.run_busy(lambda: self._save_range(path, x0, x1, chans), label="Saving",
                               done_label="Saved",
                               on_error=lambda e: self.mstatus.setText(f"Range not saved: {e}"))

    # --- keys / lifecycle ----------------------------------------------------------------
    def _on_key(self, ev):
        k = ev.key()
        K = QtCore.Qt.Key
        x0, x1 = self.view()
        span = x1 - x0
        if k in (K.Key_F, K.Key_Home):
            self.fit()
        elif k == K.Key_T and self.cap.triggered:
            c = self.cap.trigger_pos / self.rate
            self.set_view(c - span / 2, c + span / 2)
        elif k in (K.Key_Plus, K.Key_Equal):
            self.zoom(0.5)
        elif k in (K.Key_Minus, K.Key_Underscore):
            self.zoom(2.0)
        elif k == K.Key_Right:
            self.pan(span / 4)
        elif k == K.Key_Left:
            self.pan(-span / 4)
        elif k == K.Key_C:
            self.clear_cursors()
            self.clear_pins()
        elif k == K.Key_Z:
            self.set_mode("zoom")
        elif k == K.Key_P:
            self.set_mode("pan")
        elif k == K.Key_M:
            self.set_mode("measure")
        elif k == K.Key_Escape and self.fab.opened:
            self.fab.set_open(False)
        elif k in (K.Key_Q, K.Key_Escape):
            self.win.close()

    def _on_close(self, ev):
        self._ov.cancel()
        self._dec_worker.shutdown(wait=False, cancel_futures=True)
        self._worker.shutdown(wait=False, cancel_futures=True)
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._hover_pool.shutdown(wait=False, cancel_futures=True)
        ev.accept()

    def process(self, seconds: float = 0.0):
        """Run the Qt event loop for a while (tests / scripted use)."""
        import time
        end = time.monotonic() + seconds
        while True:
            self.app.processEvents()
            if time.monotonic() >= end:
                break
            time.sleep(0.002)

    def save(self, path) -> None:
        """Screenshot of the window (works offscreen too)."""
        self.win.grab().save(str(path))

    def show(self):
        self.win.show()
        self.app.exec()


def show(cap: Capture, **kw) -> QtViewer:
    v = QtViewer(cap, **kw)
    v.show()
    return v
