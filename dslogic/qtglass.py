"""Widgets and theme for the Qt viewer.

Quiet, flat surfaces in the VS Code "Dark Modern" greys with iOS dark system colours,
and one piece of liquid glass: the floating action button, which refracts the
waveforms under it and fans its actions out when the pointer reaches it. Motion only
answers the user: buttons squish when pressed, the mode switch has a pill that stretches
as it slides, and busy actions spin, then draw a check mark when done. Animations are
off when the OS has animations disabled (or DSLOGIC_REDUCED_MOTION=1).
"""
from __future__ import annotations

import math
import os
import sys

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

Qt = QtCore.Qt

# --- tokens: VS Code "Modern" greys with iOS system colours, dark and light -----------
THEMES = {
    "dark": dict(
        SURFACE="#1F1F1F",       # plot (VS Code editor)
        SIDEBAR="#181818",       # name column, top bar, status bar
        RAISED="#2A2A2C",        # menus, labels on the plot
        BORDER="#2B2B2B",
        MENU_BORDER="#3F3F42",
        INK="#D4D4D4",           # primary text
        INK_DIM="#9D9D9D",       # secondary text
        INK_FAINT="#6E6E6E",
        ACCENT="#0A84FF",        # iOS blue: selection, focus, primary action
        AMBER="#FFD60A",         # cursors
        ORANGE="#FF9F0A",        # measurements
        MINT="#30D158",          # done
        CORAL="#FF453A",         # trigger / error
        TRACK=(118, 118, 128, 60),          # segmented control track
        PILL=("#4A4A4E", "#3A3A3D"),        # its selected pill (top, bottom)
        PILL_INK="#FFFFFF",
        PILL_SHADOW=70,
        GLASS=(44, 44, 48),                 # tint of the floating actions
        GLASS_INK="#FFFFFF",
    ),
    "light": dict(
        SURFACE="#FFFFFF",
        SIDEBAR="#F2F2F7",       # iOS grouped background
        RAISED="#FFFFFF",
        BORDER="#E5E5EA",
        MENU_BORDER="#D1D1D6",
        INK="#1D1D1F",
        INK_DIM="#6E6E73",
        INK_FAINT="#AEAEB2",
        ACCENT="#007AFF",
        AMBER="#B98900",
        ORANGE="#F28500",
        MINT="#34C759",
        CORAL="#FF3B30",
        TRACK=(118, 118, 128, 30),
        PILL=("#FFFFFF", "#FFFFFF"),
        PILL_INK="#1D1D1F",
        PILL_SHADOW=30,
        GLASS=(252, 252, 254),
        GLASS_INK="#1D1D1F",
    ),
}
THEME = "dark"
globals().update(THEMES[THEME])


def set_theme(name: str) -> None:
    """Switch every colour token ("dark" / "light"); widgets pick it up on their next paint."""
    global THEME
    THEME = "light" if name == "light" else "dark"
    globals().update(THEMES[THEME])


def is_dark() -> bool:
    return THEME == "dark"


def system_theme() -> str:
    """'light' or 'dark' as the OS is set (DSLOGIC_THEME=light|dark overrides)."""
    env = os.environ.get("DSLOGIC_THEME", "").lower()
    if env in ("light", "dark"):
        return env
    try:
        cs = QtGui.QGuiApplication.styleHints().colorScheme()
        if cs == Qt.ColorScheme.Light:
            return "light"
        if cs == Qt.ColorScheme.Dark:
            return "dark"
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize") as k:
                return "light" if winreg.QueryValueEx(k, "AppsUseLightTheme")[0] else "dark"
        except OSError:
            pass
    return "dark"


def line(alpha: int) -> QtGui.QColor:
    """A faint rule or highlight: white on dark, black on light."""
    return QtGui.QColor(255, 255, 255, alpha) if THEME == "dark" else QtGui.QColor(0, 0, 0, int(alpha * 0.85))


def theme_color(color: str) -> str:
    """A trace colour made readable on the current background (bright yellows and
    pastels are darkened on the light theme)."""
    c = QtGui.QColor(color)
    if THEME == "light":
        for _ in range(6):
            lum = 0.2126 * c.redF() + 0.7152 * c.greenF() + 0.0722 * c.blueF()
            if lum <= 0.55:
                break
            c = c.darker(118)
    return c.name()


def _pick_family(cands):
    have = set(QtGui.QFontDatabase.families())
    for c in cands:
        if c in have:
            return c
    return QtWidgets.QApplication.font().family()


def ui_font(size: float = 9.5, weight=QtGui.QFont.Weight.Normal, tabular: bool = False) -> QtGui.QFont:
    fam = _pick_family(["Segoe UI Variable Text", "Segoe UI Variable", "Inter", "Segoe UI",
                        "Noto Sans", "Cantarell", "DejaVu Sans"])
    f = QtGui.QFont(fam)
    f.setPointSizeF(size)
    f.setWeight(weight)
    if tabular and hasattr(f, "setFeature"):
        try:
            f.setFeature(QtGui.QFont.Tag("tnum"), 1)
        except Exception:
            pass
    f.setHintingPreference(QtGui.QFont.HintingPreference.PreferNoHinting)
    return f


def reduced_motion() -> bool:
    if os.environ.get("DSLOGIC_REDUCED_MOTION") == "1":
        return True
    if sys.platform == "win32":
        try:
            import ctypes
            val = ctypes.c_int(1)
            ctypes.windll.user32.SystemParametersInfoW(0x1042, 0, ctypes.byref(val), 0)  # SPI_GETCLIENTAREAANIMATION
            return val.value == 0
        except Exception:
            return False
    return False


def reduced_transparency() -> bool:
    """Windows 'Transparency effects' off (or DSLOGIC_REDUCED_TRANSPARENCY=1): solid panels."""
    if os.environ.get("DSLOGIC_REDUCED_TRANSPARENCY") == "1":
        return True
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize") as k:
                return winreg.QueryValueEx(k, "EnableTransparency")[0] == 0
        except OSError:
            return False
    return False


def _ms(ms: int) -> int:
    return 0 if reduced_motion() else ms


def _c(hexstr: str, alpha: float = 1.0) -> QtGui.QColor:
    c = QtGui.QColor(hexstr)
    c.setAlphaF(alpha)
    return c


def paint_glass(p: QtGui.QPainter, rect: QtCore.QRectF, radius: float, lift: float = 0.0,
                tint: QtGui.QColor | None = None) -> None:
    """Frosted capsule: translucent body, brighter top, specular rim, soft inner shade."""
    path = QtGui.QPainterPath()
    path.addRoundedRect(rect, radius, radius)
    body = QtGui.QLinearGradient(rect.topLeft(), rect.bottomLeft())
    body.setColorAt(0.0, QtGui.QColor(255, 255, 255, int(34 + 30 * lift)))
    body.setColorAt(0.5, QtGui.QColor(255, 255, 255, int(14 + 18 * lift)))
    body.setColorAt(1.0, QtGui.QColor(255, 255, 255, int(8 + 12 * lift)))
    p.fillPath(path, body)
    if tint is not None:
        p.fillPath(path, tint)
    # specular rim: bright along the top, fading down the sides
    rim = QtGui.QLinearGradient(rect.topLeft(), rect.bottomLeft())
    rim.setColorAt(0.0, QtGui.QColor(255, 255, 255, int(120 + 60 * lift)))
    rim.setColorAt(0.45, QtGui.QColor(255, 255, 255, 30))
    rim.setColorAt(1.0, QtGui.QColor(255, 255, 255, 55))
    p.setPen(QtGui.QPen(QtGui.QBrush(rim), 1.0))
    p.drawPath(path)
    # inner highlight band (the "liquid" sheen)
    sheen = QtCore.QRectF(rect.adjusted(radius * 0.6, 1.5, -radius * 0.6, -rect.height() * 0.55))
    if sheen.width() > 4 and sheen.height() > 2:
        g = QtGui.QLinearGradient(sheen.topLeft(), sheen.bottomLeft())
        g.setColorAt(0.0, QtGui.QColor(255, 255, 255, int(45 + 30 * lift)))
        g.setColorAt(1.0, QtGui.QColor(255, 255, 255, 0))
        sp = QtGui.QPainterPath()
        sp.addRoundedRect(sheen, sheen.height() / 2, sheen.height() / 2)
        p.fillPath(sp, g)


class _PaintCache:
    """Render a widget once into a pixmap and reuse it while its look is unchanged.
    The glass widgets float over the plot, so every plot repaint repaints them too;
    blitting a cached image instead of re-drawing keeps zoom fast."""

    def __init__(self):
        self.key = None
        self.pm: QtGui.QPixmap | None = None

    def paint(self, widget, key, draw) -> None:
        dpr = widget.devicePixelRatioF()
        full = (key, THEME, widget.width(), widget.height(), dpr)
        if self.pm is None or self.key != full:
            pm = QtGui.QPixmap(int(widget.width() * dpr), int(widget.height() * dpr))
            pm.setDevicePixelRatio(dpr)
            pm.fill(QtCore.Qt.GlobalColor.transparent)
            p = QtGui.QPainter(pm)
            draw(p)
            p.end()
            self.pm, self.key = pm, full
        QtGui.QPainter(widget).drawPixmap(0, 0, self.pm)


# --- icons (drawn, no image assets) ----------------------------------------------------
def draw_icon(p: QtGui.QPainter, name: str, r: QtCore.QRectF, color: QtGui.QColor) -> None:
    pen = QtGui.QPen(color, 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    x, y, w, h = r.x(), r.y(), r.width(), r.height()
    P = lambda fx, fy: QtCore.QPointF(x + fx * w, y + fy * h)  # noqa: E731
    if name == "fit":            # four corner brackets
        for (ax, ay, dx, dy) in ((0.1, 0.1, 1, 1), (0.9, 0.1, -1, 1), (0.1, 0.9, 1, -1), (0.9, 0.9, -1, -1)):
            p.drawLine(P(ax, ay), P(ax + 0.28 * dx, ay))
            p.drawLine(P(ax, ay), P(ax, ay + 0.28 * dy))
    elif name == "trigger":      # a rising edge with a flag
        path = QtGui.QPainterPath(P(0.05, 0.85))
        for pt in ((0.45, 0.85), (0.45, 0.2), (0.95, 0.2)):
            path.lineTo(P(*pt))
        p.drawPath(path)
        p.drawLine(P(0.45, 0.2), P(0.45, 0.05))
    elif name == "cursor":       # vertical rule with a plus
        p.drawLine(P(0.35, 0.05), P(0.35, 0.95))
        p.drawLine(P(0.62, 0.5), P(0.95, 0.5))
        p.drawLine(P(0.785, 0.33), P(0.785, 0.67))
    elif name == "clear":        # cross
        p.drawLine(P(0.2, 0.2), P(0.8, 0.8))
        p.drawLine(P(0.8, 0.2), P(0.2, 0.8))
    elif name == "camera":
        p.drawRoundedRect(QtCore.QRectF(P(0.05, 0.3), P(0.95, 0.88)), 2.5, 2.5)
        p.drawEllipse(QtCore.QRectF(P(0.36, 0.42), P(0.64, 0.74)))
        p.drawLine(P(0.32, 0.3), P(0.4, 0.15))
        p.drawLine(P(0.4, 0.15), P(0.6, 0.15))
        p.drawLine(P(0.6, 0.15), P(0.68, 0.3))
    elif name == "save":         # arrow into a tray
        p.drawLine(P(0.5, 0.08), P(0.5, 0.62))
        p.drawLine(P(0.3, 0.42), P(0.5, 0.62))
        p.drawLine(P(0.7, 0.42), P(0.5, 0.62))
        path = QtGui.QPainterPath(P(0.08, 0.62))
        for pt in ((0.08, 0.9), (0.92, 0.9), (0.92, 0.62)):
            path.lineTo(P(*pt))
        p.drawPath(path)
    elif name == "zoom":
        p.drawEllipse(QtCore.QRectF(P(0.08, 0.08), P(0.68, 0.68)))
        p.drawLine(P(0.62, 0.62), P(0.92, 0.92))
    elif name == "pan":          # left-right arrows
        p.drawLine(P(0.08, 0.5), P(0.92, 0.5))
        p.drawLine(P(0.08, 0.5), P(0.28, 0.3))
        p.drawLine(P(0.08, 0.5), P(0.28, 0.7))
        p.drawLine(P(0.92, 0.5), P(0.72, 0.3))
        p.drawLine(P(0.92, 0.5), P(0.72, 0.7))
    elif name == "info":         # circled i
        p.drawEllipse(QtCore.QRectF(P(0.06, 0.06), P(0.94, 0.94)))
        p.drawLine(P(0.5, 0.45), P(0.5, 0.74))
        p.drawPoint(P(0.5, 0.29))
    elif name == "sun":
        p.drawEllipse(QtCore.QRectF(P(0.3, 0.3), P(0.7, 0.7)))
        for k in range(8):
            a = k * math.pi / 4
            p.drawLine(P(0.5 + 0.32 * math.cos(a), 0.5 + 0.32 * math.sin(a)),
                       P(0.5 + 0.46 * math.cos(a), 0.5 + 0.46 * math.sin(a)))
    elif name == "moon":         # crescent
        path = QtGui.QPainterPath()
        path.addEllipse(QtCore.QRectF(P(0.1, 0.1), P(0.9, 0.9)))
        bite = QtGui.QPainterPath()
        bite.addEllipse(QtCore.QRectF(P(0.38, -0.05), P(1.1, 0.67)))
        p.drawPath(path.subtracted(bite))
    elif name == "measure":      # caliper: two ticks and a span
        p.drawLine(P(0.1, 0.2), P(0.1, 0.8))
        p.drawLine(P(0.9, 0.2), P(0.9, 0.8))
        p.drawLine(P(0.1, 0.5), P(0.9, 0.5))


def draw_spinner(p: QtGui.QPainter, r: QtCore.QRectF, angle: float, color: QtGui.QColor) -> None:
    pen = QtGui.QPen(color, 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    track = QtGui.QPen(line(40), 2.0)
    p.setPen(track)
    p.drawEllipse(r)
    p.setPen(pen)
    p.drawArc(r, int(-angle * 16), int(110 * 16))


def draw_check(p: QtGui.QPainter, r: QtCore.QRectF, progress: float, color: QtGui.QColor) -> None:
    """A check mark drawn progressively (progress 0..1)."""
    pts = [QtCore.QPointF(r.x() + 0.12 * r.width(), r.y() + 0.55 * r.height()),
           QtCore.QPointF(r.x() + 0.42 * r.width(), r.y() + 0.82 * r.height()),
           QtCore.QPointF(r.x() + 0.9 * r.width(), r.y() + 0.2 * r.height())]
    l1 = math.dist((pts[0].x(), pts[0].y()), (pts[1].x(), pts[1].y()))
    l2 = math.dist((pts[1].x(), pts[1].y()), (pts[2].x(), pts[2].y()))
    d = progress * (l1 + l2)
    pen = QtGui.QPen(color, 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    path = QtGui.QPainterPath(pts[0])
    if d <= l1:
        t = d / l1 if l1 else 1
        path.lineTo(pts[0] + (pts[1] - pts[0]) * t)
    else:
        path.lineTo(pts[1])
        t = (d - l1) / l2 if l2 else 1
        path.lineTo(pts[1] + (pts[2] - pts[1]) * t)
    p.drawPath(path)


def _track_focus(cls):
    """Show focus rings only for keyboard focus (Tab), not after a click or at startup."""
    orig_in = cls.focusInEvent
    orig_out = cls.focusOutEvent

    def focusInEvent(self, ev):
        self._kbd_focus = ev.reason() in (Qt.FocusReason.TabFocusReason, Qt.FocusReason.BacktabFocusReason)
        self.update()
        orig_in(self, ev)

    def focusOutEvent(self, ev):
        self._kbd_focus = False
        self.update()
        orig_out(self, ev)

    cls.focusInEvent, cls.focusOutEvent = focusInEvent, focusOutEvent
    return cls


class _Anim(QtCore.QObject):
    """Tiny helper: animate a float attribute on a widget, repainting as it goes."""

    def __init__(self, widget, attr, duration, curve=QtCore.QEasingCurve.Type.OutCubic,
                 target=None, on_change=None):
        """Animates `attr` of `target` (default: the widget, which also owns the animation);
        on_change() replaces the widget repaint."""
        super().__init__(widget)
        self.w, self.attr, self.on_change = (widget if target is None else target), attr, on_change
        self.a = QtCore.QVariantAnimation(self)
        self.a.setDuration(duration)
        self.a.setEasingCurve(curve)
        self.a.valueChanged.connect(self._set)

    def _set(self, v):
        setattr(self.w, self.attr, float(v))
        if self.on_change is not None:
            self.on_change()
        else:
            self.w.update()

    def to(self, target: float, duration: int | None = None):
        d = _ms(duration if duration is not None else self.a.duration())
        if d == 0:
            self.a.stop()
            self._set(target)
            return
        self.a.stop()
        self.a.setDuration(d)
        self.a.setStartValue(float(getattr(self.w, self.attr)))
        self.a.setEndValue(float(target))
        self.a.start()


@_track_focus
class GlassButton(QtWidgets.QAbstractButton):
    """Capsule button: lifts on hover, squishes on press, stays lit when checked (sticky),
    spins while busy and draws a check mark when done (`run_busy()` / `done()` / `failed()`)."""
    stateChanged = QtCore.Signal(str)

    def __init__(self, text: str, icon: str | None = None, checkable: bool = False, parent=None,
                 tip: str = "", variant: str = "glass", busy_labels: tuple = ()):
        super().__init__(parent)
        self.busy_labels = busy_labels   # labels it may switch to, so its width never jumps
        self.variant = variant           # "glass" | "plain" (toolbar item) | "prominent" (tinted)
        self.setText(text)
        self.icon_name = icon
        self.setCheckable(checkable)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setToolTip(tip)
        self.setFont(ui_font(9.5, QtGui.QFont.Weight.Medium))
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        self.hover = 0.0
        self.press = 0.0
        self.lit = 0.0
        self.spin = 0.0
        self.check = 0.0
        self.state = "idle"           # idle | busy | done | error
        self._label_override = None
        self._ha = _Anim(self, "hover", 160)
        self._pa = _Anim(self, "press", 90, QtCore.QEasingCurve.Type.OutQuad)
        self._la = _Anim(self, "lit", 220)
        self._ca = _Anim(self, "check", 420, QtCore.QEasingCurve.Type.OutCubic)
        self._spin = QtCore.QVariantAnimation(self)
        self._spin.setStartValue(0.0)
        self._spin.setEndValue(360.0)
        self._spin.setDuration(900)
        self._spin.setLoopCount(-1)
        self._spin.valueChanged.connect(lambda v: (setattr(self, "spin", float(v)), self.update()))
        self._reset = QtCore.QTimer(self)
        self._reset.setSingleShot(True)
        self._reset.timeout.connect(self._back_to_idle)
        self.toggled.connect(lambda on: self._la.to(1.0 if on else 0.0))

    # states -----------------------------------------------------------------------
    def set_busy(self, label: str | None = None):
        self.state = "busy"
        self._label_override = label
        self._reset.stop()
        if reduced_motion():
            self.spin = 45.0
        else:
            self._spin.start()
        self.setEnabled(False)
        self.update()
        self.stateChanged.emit(self.state)

    def done(self, label: str = "Saved", hold_ms: int = 1600):
        self._finish("done", label, hold_ms)

    def failed(self, label: str = "Failed", hold_ms: int = 2200):
        self._finish("error", label, hold_ms)

    def _finish(self, state, label, hold_ms):
        self._spin.stop()
        self.state = state
        self._label_override = label
        self.check = 0.0
        self._ca.to(1.0)
        self.setEnabled(True)
        self._reset.start(hold_ms)
        self.update()
        self.stateChanged.emit(self.state)

    def _back_to_idle(self):
        self.state = "idle"
        self._label_override = None
        self.update()
        self.stateChanged.emit(self.state)

    def run_busy(self, work, label="Saving", done_label="Saved", on_error=None):
        """Run `work()` in a thread with spinner, then check mark (or error)."""
        import threading
        self.set_busy(label)
        bridge = _Signal(self)

        def runner():
            try:
                res = work()
                bridge.sig.emit(("ok", res))
            except Exception as e:     # reported on the button and to on_error
                bridge.sig.emit(("err", e))

        def finished(msg):
            kind, val = msg
            if kind == "ok":
                self.done(done_label)
            else:
                self.failed("Failed")
                if on_error:
                    on_error(val)

        bridge.sig.connect(finished)
        threading.Thread(target=runner, daemon=True).start()

    # interaction ------------------------------------------------------------------
    def enterEvent(self, ev):
        self._ha.to(1.0)
        super().enterEvent(ev)

    def leaveEvent(self, ev):
        self._ha.to(0.0)
        self._pa.to(0.0)
        super().leaveEvent(ev)

    def mousePressEvent(self, ev):
        self._pa.to(1.0)
        super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev):
        self._pa.to(0.0, 260)      # spring back slower than the press: a soft "sticky" feel
        super().mouseReleaseEvent(ev)

    def sizeHint(self):
        fm = QtGui.QFontMetrics(self.font())
        text = max((self.text(), *self.busy_labels), key=fm.horizontalAdvance) if self.text() else ""
        w = fm.horizontalAdvance(text) + (48 if self.icon_name else 24)
        return QtCore.QSize(max(w, 36), 32)

    def minimumSizeHint(self):
        return self.sizeHint()

    def paintEvent(self, ev):
        key = (round(self.hover, 3), round(self.press, 3), round(self.lit, 3), round(self.check, 3),
               int(self.spin) if self.state == "busy" else 0, self.state, self._label_override, self.text(),
               self.isEnabled(), self.hasFocus() and getattr(self, "_kbd_focus", False), self.variant,
               self.icon_name)
        if not hasattr(self, "_pc"):
            self._pc = _PaintCache()
        self._pc.paint(self, key, self._draw)

    def _draw(self, p):
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        r = QtCore.QRectF(self.rect()).adjusted(2, 2, -2, -2)
        s = 1.0 - 0.05 * self.press + 0.02 * self.hover
        c = r.center()
        p.translate(c)
        p.scale(s, s)
        p.translate(-c)
        tint = None
        if self.state == "done":
            tint = _c(MINT, 0.22 * self.check)
        elif self.state == "error":
            tint = _c(CORAL, 0.26 * self.check)
        elif self.lit > 0:
            tint = _c(ACCENT, 0.25 * self.lit)
        elif self.variant == "prominent":
            tint = _c(ACCENT, 0.30 + 0.12 * self.hover)
        if self.variant == "plain" and tint is None:
            # toolbar item on a glass capsule: no second glass layer, just a soft highlight
            a = 26 * self.hover + 34 * self.press
            if a > 0.5:
                path = QtGui.QPainterPath()
                path.addRoundedRect(r, r.height() / 2, r.height() / 2)
                p.fillPath(path, line(int(a)))
        elif self.variant == "plain":
            path = QtGui.QPainterPath()
            path.addRoundedRect(r, r.height() / 2, r.height() / 2)
            p.fillPath(path, tint)
        else:
            paint_glass(p, r, r.height() / 2, lift=0.6 * self.hover + 0.5 * self.lit, tint=tint)
        if self.hasFocus() and getattr(self, "_kbd_focus", False):
            p.setPen(QtGui.QPen(_c(ACCENT, 0.9), 1.4))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(r.adjusted(-1, -1, 1, 1), r.height() / 2 + 1, r.height() / 2 + 1)
        ink = QtGui.QColor(INK) if self.isEnabled() or self.state == "busy" else _c(INK_DIM, 0.6)
        if self.lit > 0.5:
            ink = QtGui.QColor("#FFFFFF")
        ir = QtCore.QRectF(r.x() + 11, r.center().y() - 7, 14, 14)
        if self.state == "busy":
            draw_spinner(p, ir, self.spin, _c(ACCENT))
        elif self.state == "done":
            draw_check(p, ir, self.check, _c(MINT))
        elif self.state == "error":
            draw_icon(p, "clear", ir, _c(CORAL))
        elif self.icon_name:
            draw_icon(p, self.icon_name, ir, ink)
        text = self._label_override or self.text()
        if text:
            p.setPen(ink if self.state == "idle" else QtGui.QColor(INK))
            p.setFont(self.font())
            tx = r.adjusted(32 if self.icon_name or self.state != "idle" else 10, 0, -10, 0)
            p.drawText(tx, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)


class _Signal(QtCore.QObject):
    sig = QtCore.Signal(object)


@_track_focus
class LiquidSwitch(QtWidgets.QWidget):
    """Segmented control with a liquid pill: it stretches toward the new option while
    sliding and settles with a small overshoot. Emits `changed(index)`."""
    changed = QtCore.Signal(int)

    def __init__(self, options: list[tuple[str, str]], parent=None, framed: bool = True):
        super().__init__(parent)
        self.framed = framed                 # False when it sits on a glass capsule already
        self.options = options               # (label, icon)
        self.index = 0
        self.pos = 0.0                       # animated pill position (in option units)
        self.lead = 0.0                      # animated leading edge for the stretch
        self.hover_i = -1
        self.setFont(ui_font(9.5, QtGui.QFont.Weight.Medium))
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        self._pos_a = _Anim(self, "pos", 420, QtCore.QEasingCurve.Type.OutBack)
        self._lead_a = _Anim(self, "lead", 220, QtCore.QEasingCurve.Type.OutCubic)

    def _seg_w(self):
        fm = QtGui.QFontMetrics(self.font())
        return max(fm.horizontalAdvance(t) for t, _ in self.options) + 40

    def sizeHint(self):
        return QtCore.QSize(self._seg_w() * len(self.options) + 6, 30)

    def minimumSizeHint(self):
        return self.sizeHint()

    def set_index(self, i: int, emit: bool = True):
        i = max(0, min(len(self.options) - 1, i))
        if i == self.index:
            return
        self.index = i
        self._lead_a.to(float(i))         # the leading edge races ahead...
        self._pos_a.to(float(i))          # ...the body follows with overshoot
        if emit:
            self.changed.emit(i)

    def _hit(self, x):
        return int(min(len(self.options) - 1, max(0, (x - 3) // self._seg_w())))

    def mousePressEvent(self, ev):
        self.set_index(self._hit(ev.position().x()))

    def mouseMoveEvent(self, ev):
        h = self._hit(ev.position().x())
        if h != self.hover_i:
            self.hover_i = h
            self.update()

    def leaveEvent(self, ev):
        self.hover_i = -1
        self.update()

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key.Key_Left, Qt.Key.Key_Up):
            self.set_index(self.index - 1)
        elif ev.key() in (Qt.Key.Key_Right, Qt.Key.Key_Down):
            self.set_index(self.index + 1)
        else:
            super().keyPressEvent(ev)

    def paintEvent(self, ev):
        if not hasattr(self, "_pc"):
            self._pc = _PaintCache()
        self._pc.paint(self, (round(self.pos, 4), round(self.lead, 4), self.index, self.hover_i,
                              self.hasFocus() and getattr(self, "_kbd_focus", False), self.framed), self._draw)

    def _draw(self, p):
        """iOS segmented control: a recessed track and a raised pill under the choice."""
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        outer = QtCore.QRectF(self.rect()).adjusted(1, 1, -1, -1)
        track = QtGui.QPainterPath()
        track.addRoundedRect(outer, 9, 9)
        p.fillPath(track, QtGui.QColor(*TRACK))
        sw = self._seg_w()
        a, b = sorted((self.pos, self.lead))
        pill = QtCore.QRectF(3 + a * sw, 3, sw + (b - a) * sw, outer.height() - 4)
        path = QtGui.QPainterPath()
        path.addRoundedRect(pill, 7, 7)
        p.fillPath(path.translated(0, 1), QtGui.QColor(0, 0, 0, PILL_SHADOW))  # contact shadow
        g = QtGui.QLinearGradient(pill.topLeft(), pill.bottomLeft())
        g.setColorAt(0.0, QtGui.QColor(PILL[0]))
        g.setColorAt(1.0, QtGui.QColor(PILL[1]))
        p.fillPath(path, g)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 28 if THEME == "dark" else 0), 1.0))
        p.drawLine(QtCore.QPointF(pill.left() + 6, pill.top() + 0.5), QtCore.QPointF(pill.right() - 6, pill.top() + 0.5))
        p.setFont(self.font())
        for i, (label, icon) in enumerate(self.options):
            seg = QtCore.QRectF(3 + i * sw, 0, sw, outer.height() + 2)
            on = i == self.index
            col = QtGui.QColor(PILL_INK) if on else (QtGui.QColor(INK) if i == self.hover_i else QtGui.QColor(INK_DIM))
            fm = QtGui.QFontMetrics(self.font())
            gap = 20 if icon else 0
            content = gap + fm.horizontalAdvance(label)            # icon + gap + label, centred
            x = seg.center().x() - content / 2
            if icon:
                draw_icon(p, icon, QtCore.QRectF(x, seg.center().y() - 6.5, 13, 13), col)
            p.setPen(col)
            p.drawText(QtCore.QRectF(x + gap, seg.y(), fm.horizontalAdvance(label) + 4, seg.height()),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, label)
        if self.hasFocus() and getattr(self, "_kbd_focus", False):
            p.setPen(QtGui.QPen(_c(ACCENT, 0.9), 1.4))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(outer, 9, 9)


class StatusChip(QtWidgets.QWidget):
    """Background-work indicator: spinner while busy, then a check and 'Ready'."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFont(ui_font(9.0))
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        self.text = "Ready"
        self.state = "idle"
        self.spin = 0.0
        self.check = 1.0
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._spin = QtCore.QVariantAnimation(self)
        self._spin.setStartValue(0.0)
        self._spin.setEndValue(360.0)
        self._spin.setDuration(900)
        self._spin.setLoopCount(-1)
        self._spin.valueChanged.connect(lambda v: (setattr(self, "spin", float(v)), self.update()))
        self._ca = _Anim(self, "check", 380)
        self._delay = QtCore.QTimer(self)
        self._delay.setSingleShot(True)
        self._delay.timeout.connect(self._show_busy)
        self._pending = ""

    def sizeHint(self):
        fm = QtGui.QFontMetrics(self.font())
        return QtCore.QSize(fm.horizontalAdvance(self.text) + 44, 32)

    def minimumSizeHint(self):
        return self.sizeHint()

    def busy(self, text="Rendering", delay_ms=120):
        """Show the spinner only if the work takes longer than delay_ms (no flicker)."""
        self._pending = text
        if self.state != "busy" and not self._delay.isActive():
            self._delay.start(delay_ms)

    def _set_text(self, text):
        if text != self.text:
            self.text = text
            self.updateGeometry()
            cb = getattr(self, "resized", None)
            if cb:
                cb()

    def _show_busy(self):
        self.state = "busy"
        self._set_text(self._pending)
        if not reduced_motion():
            self._spin.start()
        self.update()

    def ready(self, text="Ready"):
        self._delay.stop()
        if self.state != "busy" and self.text == text:
            return                                   # nothing visible changes: no repaint
        was_busy = self.state == "busy"
        self._spin.stop()
        self.state = "idle"
        self._set_text(text)
        if was_busy:
            self.check = 0.0
            self._ca.to(1.0)
        self.update()

    def paintEvent(self, ev):
        if not hasattr(self, "_pc"):
            self._pc = _PaintCache()
        self._pc.paint(self, (self.state, self.text, int(self.spin) if self.state == "busy" else 0,
                              round(self.check, 3)), self._draw)

    def _draw(self, p):
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        r = QtCore.QRectF(self.rect()).adjusted(2, 0, -2, 0)
        ir = QtCore.QRectF(r.x() + 10, r.center().y() - 6.5, 13, 13)
        if self.state == "busy":
            draw_spinner(p, ir, self.spin, _c(ACCENT))
            p.setPen(QtGui.QColor(INK))
        else:
            draw_check(p, ir, self.check, _c(MINT, 0.9))
            p.setPen(QtGui.QColor(INK_DIM))
        p.setFont(self.font())
        p.drawText(r.adjusted(30, 0, 0, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.text)


class Hairline(QtWidgets.QWidget):
    """A thin vertical divider in a bar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(17)

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        x = self.width() // 2
        p.setPen(line(26))
        p.drawLine(x, 6, x, self.height() - 6)


class Bar(QtWidgets.QWidget):
    """A docked bar (top bar, status bar): the sidebar grey with a hairline border on
    the edge that meets the plot."""

    def __init__(self, edge: str = "bottom", parent=None):
        super().__init__(parent)
        self.edge = edge
        self.opacity = 1.0

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), _c(SIDEBAR, self.opacity))
        p.setPen(QtGui.QColor(BORDER))
        y = self.height() - 1 if self.edge == "bottom" else 0
        p.drawLine(0, y, self.width(), y)


def paint_surface(p: QtGui.QPainter, full: QtCore.QRectF, sidebar: float = 0.0, opacity: float = 1.0) -> None:
    """Window body: the plot grey, with the darker name column on the left. `full` is
    the whole surface in the painter's coordinates (so a crop can be painted for glass)."""
    p.fillRect(full, _c(SURFACE, opacity))
    if sidebar > 0:
        side = QtCore.QRectF(full.x(), full.y(), sidebar, full.height())
        p.fillRect(side, _c(SIDEBAR, opacity))
        p.setPen(QtGui.QColor(BORDER))
        p.drawLine(QtCore.QPointF(side.right() + 0.5, full.top()), QtCore.QPointF(side.right() + 0.5, full.bottom()))


class Surface(QtWidgets.QWidget):
    """Window background (see paint_surface). With Windows 11 window glass on, it is drawn
    slightly translucent so the system acrylic tints through."""
    opacity = 1.0
    sidebar = 0.0

    def paintEvent(self, ev):
        if not hasattr(self, "_pc"):
            self._pc = _PaintCache()
        self._pc.paint(self, ("surface", self.opacity, self.sidebar),
                       lambda p: paint_surface(p, QtCore.QRectF(self.rect()), self.sidebar, self.opacity))


def blur_pixmap(pm: QtGui.QPixmap, radius: float) -> QtGui.QPixmap:
    """Gaussian-blurred copy (QGraphicsBlurEffect on a tiny scene; cheap on small pixmaps)."""
    if pm.isNull() or radius <= 0:
        return pm
    scene = QtWidgets.QGraphicsScene()
    item = QtWidgets.QGraphicsPixmapItem(pm)
    eff = QtWidgets.QGraphicsBlurEffect()
    eff.setBlurRadius(radius)
    eff.setBlurHints(QtWidgets.QGraphicsBlurEffect.BlurHint.QualityHint)
    item.setGraphicsEffect(eff)
    scene.addItem(item)
    out = QtGui.QPixmap(pm.size())
    out.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(out)
    scene.render(p, QtCore.QRectF(out.rect()), QtCore.QRectF(pm.rect()))
    p.end()
    return out


def paint_liquid(p: QtGui.QPainter, r: QtCore.QRectF, rad: float, tint: QtGui.QColor,
                 backdrop: QtGui.QPixmap | None = None, src: QtCore.QRectF | None = None,
                 shadow: float = 1.0) -> None:
    """Liquid glass: the blurred backdrop (src part of `backdrop`), magnified along the rim
    like a lens edge, a tint, inner light and a specular rim, over a soft shadow."""
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
    if shadow > 0:
        p.setPen(Qt.PenStyle.NoPen)
        for k in range(7):
            p.setBrush(QtGui.QColor(0, 0, 0, int((30 - 4 * k) * shadow)))
            p.drawRoundedRect(r.adjusted(-k, k * 0.7 + 2, k, k + 2.5), rad + k, rad + k)
    path = QtGui.QPainterPath()
    path.addRoundedRect(r, rad, rad)
    p.save()
    p.setClipPath(path)
    if backdrop is not None and src is not None and not backdrop.isNull() and not reduced_transparency():
        p.drawPixmap(r, backdrop, src)
        inner = QtGui.QPainterPath()
        e = min(7.0, r.height() * 0.2)
        inner.addRoundedRect(r.adjusted(e, e, -e, -e), max(0.0, rad - e), max(0.0, rad - e))
        p.setClipPath(path.subtracted(inner))
        mag = src.adjusted(src.width() * 0.06, src.height() * 0.16, -src.width() * 0.06, -src.height() * 0.16)
        p.setOpacity(0.8)
        p.drawPixmap(r, backdrop, mag)
        p.setOpacity(1.0)
        p.setClipPath(path)
    else:
        p.fillPath(path, QtGui.QColor(RAISED))
    p.fillPath(path, tint)
    g = QtGui.QLinearGradient(r.topLeft(), r.bottomLeft())
    g.setColorAt(0.0, QtGui.QColor(255, 255, 255, 40))
    g.setColorAt(0.4, QtGui.QColor(255, 255, 255, 6))
    g.setColorAt(1.0, QtGui.QColor(0, 0, 0, 30))
    p.fillPath(path, g)
    p.restore()
    rim = QtGui.QLinearGradient(r.topLeft(), r.bottomRight())
    rim.setColorAt(0.0, QtGui.QColor(255, 255, 255, 150))
    rim.setColorAt(0.35, QtGui.QColor(255, 255, 255, 40))
    rim.setColorAt(0.7, QtGui.QColor(255, 255, 255, 18))
    rim.setColorAt(1.0, QtGui.QColor(255, 255, 255, 70))
    p.setPen(QtGui.QPen(QtGui.QBrush(rim), 1.0))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(path)


class FabAction(GlassButton):
    """One action of the floating action button: a glass capsule with its label and a
    round icon well on the right, lined up under the button. `appear` (0..1) is driven by
    the menu while it fans out."""
    H = 36
    M = 8                      # room for the shadow

    def __init__(self, parent, menu, text, icon, tip="", prominent=False, busy_labels=()):
        super().__init__(text, icon, parent=parent, tip=tip, variant="prominent" if prominent else "glass",
                         busy_labels=busy_labels)
        self.menu = menu
        self.appear = 0.0
        self.setFont(ui_font(9.5, QtGui.QFont.Weight.Medium))

    def sizeHint(self):
        fm = QtGui.QFontMetrics(self.font())
        text = max((self.text(), *self.busy_labels), key=fm.horizontalAdvance)
        return QtCore.QSize(fm.horizontalAdvance(text) + 18 + self.H + 2 * self.M, self.H + 2 * self.M)

    def paintEvent(self, ev):
        if not hasattr(self, "_pc"):
            self._pc = _PaintCache()
        bd = self.menu.backdrop.cacheKey() if self.menu.backdrop is not None else 0
        key = (round(self.hover, 3), round(self.press, 3), round(self.check, 3), round(self.appear, 3),
               int(self.spin) if self.state == "busy" else 0, self.state, self._label_override, self.text(),
               self.isEnabled(), self.hasFocus() and getattr(self, "_kbd_focus", False), bd,
               self.x(), self.y())
        self._pc.paint(self, key, self._draw)

    def _draw(self, p):
        body = QtCore.QRectF(self.rect()).adjusted(self.M, self.M, -self.M, -self.M)
        well = QtCore.QRectF(body.right() - self.H, body.top(), self.H, self.H)
        p.setOpacity(self.appear)
        s = (0.55 + 0.45 * self.appear) * (1.0 - 0.05 * self.press + 0.02 * self.hover)
        c = well.center()                       # grows out of the button's axis
        p.translate(c)
        p.scale(s, s)
        p.translate(-c)
        if self.state == "done":
            tint = _c(MINT, 0.30 + 0.2 * self.check)
        elif self.state == "error":
            tint = _c(CORAL, 0.45)
        elif self.variant == "prominent":
            tint = _c(ACCENT, 0.72 + 0.12 * self.hover)
        else:
            tint = QtGui.QColor(*GLASS, int((150 if THEME == "dark" else 170) + 40 * self.hover))
        pix, src = self.menu.glass_src(self, body)
        paint_liquid(p, body, body.height() / 2, tint, pix, src, shadow=self.appear)
        if self.hover > 0.01 and self.variant != "prominent":
            hp = QtGui.QPainterPath()
            hp.addRoundedRect(body, body.height() / 2, body.height() / 2)
            p.fillPath(hp, line(int(18 * self.hover)))
        if self.hasFocus() and getattr(self, "_kbd_focus", False):
            p.setPen(QtGui.QPen(_c(ACCENT, 0.95), 1.6))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(body.adjusted(-2, -2, 2, 2), body.height() / 2 + 2, body.height() / 2 + 2)
        on_tint = self.variant == "prominent" or self.state in ("done", "error")
        full = QtGui.QColor("#FFFFFF" if on_tint else GLASS_INK)
        ink = full if self.isEnabled() or self.state == "busy" else _c(INK_DIM, 0.6)
        ir = QtCore.QRectF(well.center().x() - 7.5, well.center().y() - 7.5, 15, 15)
        if self.state == "busy":
            draw_spinner(p, ir, self.spin, full)
        elif self.state == "done":
            draw_check(p, ir, self.check, full)
        elif self.state == "error":
            draw_icon(p, "clear", ir, full)
        elif self.icon_name:
            draw_icon(p, self.icon_name, ir, ink)
        p.setPen(ink)
        p.setFont(self.font())
        p.drawText(body.adjusted(16, 0, -self.H, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                   self._label_override or self.text())


@_track_focus
class Fab(QtWidgets.QAbstractButton):
    """The floating action button: a round, blue liquid-glass lens. Its plus turns into a
    close mark while the actions are out; it spins while an action works in the
    background and shows a check (or a cross) when that finishes."""
    SIZE = 52
    M = 10

    def __init__(self, parent, menu):
        super().__init__(parent)
        self.menu = menu
        self.hover = self.press = self.turn = self.check = self.spin = 0.0
        self.state = "idle"
        self.setFixedSize(self.SIZE + 2 * self.M, self.SIZE + 2 * self.M)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setToolTip("Actions")
        self._ha = _Anim(self, "hover", 160)
        self._pa = _Anim(self, "press", 90, QtCore.QEasingCurve.Type.OutQuad)
        self._ta = _Anim(self, "turn", 300, QtCore.QEasingCurve.Type.OutBack)
        self._ca = _Anim(self, "check", 420)
        self._spin = QtCore.QVariantAnimation(self)
        self._spin.setStartValue(0.0)
        self._spin.setEndValue(360.0)
        self._spin.setDuration(900)
        self._spin.setLoopCount(-1)
        self._spin.valueChanged.connect(lambda v: (setattr(self, "spin", float(v)), self.update()))
        self._reset = QtCore.QTimer(self)
        self._reset.setSingleShot(True)
        self._reset.timeout.connect(lambda: self.set_state("idle"))

    def set_state(self, state: str) -> None:
        self.state = state
        self._spin.stop()
        if state == "busy":
            if not reduced_motion():
                self._spin.start()
        elif state in ("done", "error"):
            self.check = 0.0
            self._ca.to(1.0)
            self._reset.start(1500)
        self.update()

    def enterEvent(self, ev):
        self._ha.to(1.0)
        super().enterEvent(ev)

    def leaveEvent(self, ev):
        self._ha.to(0.0)
        self._pa.to(0.0)
        super().leaveEvent(ev)

    def mousePressEvent(self, ev):
        self._pa.to(1.0)
        super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev):
        self._pa.to(0.0, 260)
        super().mouseReleaseEvent(ev)

    def paintEvent(self, ev):
        if not hasattr(self, "_pc"):
            self._pc = _PaintCache()
        bd = self.menu.backdrop.cacheKey() if self.menu.backdrop is not None else 0
        key = (round(self.hover, 3), round(self.press, 3), round(self.turn, 3), round(self.check, 3),
               int(self.spin) if self.state == "busy" else 0, self.state, bd, self.x(), self.y(),
               self.hasFocus() and getattr(self, "_kbd_focus", False))
        self._pc.paint(self, key, self._draw)

    def _draw(self, p):
        body = QtCore.QRectF(self.rect()).adjusted(self.M, self.M, -self.M, -self.M)
        c = body.center()
        s = 1.0 + 0.05 * self.hover - 0.07 * self.press
        p.translate(c)
        p.scale(s, s)
        p.translate(-c)
        pix, src = self.menu.glass_src(self, body)
        tint = _c(CORAL, 0.75) if self.state == "error" else _c(ACCENT, 0.66 + 0.12 * self.hover)
        paint_liquid(p, body, body.height() / 2, tint, pix, src)
        if self.hasFocus() and getattr(self, "_kbd_focus", False):
            p.setPen(QtGui.QPen(QtGui.QColor("#FFFFFF"), 1.6))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(body.adjusted(-3, -3, 3, 3))
        white = QtGui.QColor("#FFFFFF")
        if self.state == "busy":
            draw_spinner(p, body.adjusted(5, 5, -5, -5), self.spin, white)
        if self.state in ("done", "error"):
            ir = QtCore.QRectF(c.x() - 10, c.y() - 10, 20, 20)
            if self.state == "done":
                draw_check(p, ir, self.check, white)
            else:
                draw_icon(p, "clear", ir.adjusted(2, 2, -2, -2), white)
            return
        p.save()                                 # plus, turning into a close mark
        p.translate(c)
        p.rotate(45 * self.turn)
        p.setPen(QtGui.QPen(white, 2.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        k = 8.0 if self.state != "busy" else 6.0
        p.drawLine(QtCore.QPointF(-k, 0), QtCore.QPointF(k, 0))
        p.drawLine(QtCore.QPointF(0, -k), QtCore.QPointF(0, k))
        p.restore()


class FabMenu(QtCore.QObject):
    """Floating action button whose actions fan out above it while the pointer is on it
    (or after a click, until the next click). Actions spring out one after another and
    tuck back in reverse order when the pointer leaves."""
    GAP = 6
    GROUP_GAP = 16
    STAGGER = 0.08

    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(parent)
        self.parent_w = parent
        self.items: list[FabAction] = []
        self._group: list[bool] = []
        self.fab = Fab(parent, self)
        self.fab.installEventFilter(self)
        self.fab.clicked.connect(self._clicked)
        self.t = 0.0                              # 0 closed .. 1 fully out
        self.opened = False
        self.pinned = False
        self.backdrop: QtGui.QPixmap | None = None
        self.origin = QtCore.QRect()
        self.request_backdrop = None              # called when the menu opens
        self._center = QtCore.QPointF(0, 0)
        self._anim = QtCore.QVariantAnimation(self)
        self._anim.valueChanged.connect(lambda v: self._set_t(float(v)))
        self._close = QtCore.QTimer(self)
        self._close.setSingleShot(True)
        self._close.timeout.connect(self._maybe_close)

    def add(self, text, icon, tip="", prominent=False, busy_labels=(), new_group=False) -> FabAction:
        """Add an action; the first one added sits closest to the button."""
        it = FabAction(self.parent_w, self, text, icon, tip, prominent, busy_labels)
        it.hide()
        it.installEventFilter(self)
        it.stateChanged.connect(self._on_state)
        self.items.append(it)
        self._group.append(new_group)
        return it

    def place(self, center: QtCore.QPointF) -> None:
        """Put the button's centre at `center` (parent coordinates)."""
        self._center = QtCore.QPointF(center)
        self.fab.move(int(center.x() - self.fab.width() / 2), int(center.y() - self.fab.height() / 2))
        self._layout()

    def targets(self) -> list[QtCore.QRect]:
        out = []
        y = self._center.y() - Fab.SIZE / 2 - self.GAP           # bottom of the next action's body
        for it, group in zip(self.items, self._group):
            if group:
                y -= self.GROUP_GAP - self.GAP
            sz = it.sizeHint()
            right = self._center.x() + FabAction.H / 2 + FabAction.M
            out.append(QtCore.QRect(int(right - sz.width()), int(y - FabAction.H - FabAction.M),
                                    sz.width(), sz.height()))
            y -= FabAction.H + self.GAP
        return out

    def region(self) -> QtCore.QRect:
        """Where the glass needs a backdrop: the button, plus the actions while open."""
        r = self.fab.geometry()
        if self.opened or self.t > 0:
            for t in self.targets():
                r = r.united(t)
        return r

    def glass_src(self, widget, body: QtCore.QRectF):
        pix = self.backdrop
        if pix is None or pix.isNull() or self.origin.width() <= 0:
            return None, None
        k = pix.width() / self.origin.width()
        x = widget.x() + body.x() - self.origin.x()
        y = widget.y() + body.y() - self.origin.y()
        return pix, QtCore.QRectF(x * k, y * k, body.width() * k, body.height() * k)

    def set_backdrop(self, pix: QtGui.QPixmap, origin: QtCore.QRect) -> None:
        self.backdrop, self.origin = pix, QtCore.QRect(origin)
        for w in [self.fab] + self.items:
            if w.isVisible():
                w.update()

    def set_open(self, on: bool, animate: bool = True) -> None:
        if on == self.opened:
            return
        self.opened = on
        if not on:
            self.pinned = False
        if on and self.request_backdrop:
            self.request_backdrop()
        self.fab._ta.to(1.0 if on else 0.0)
        target = 1.0 if on else 0.0
        d = _ms(int((420 if on else 240) * max(0.25, abs(target - self.t)))) if animate else 0
        self._anim.stop()
        if d == 0:
            self._set_t(target)
            return
        self._anim.setDuration(d)
        self._anim.setStartValue(self.t)
        self._anim.setEndValue(target)
        self._anim.start()

    def _set_t(self, v: float) -> None:
        self.t = v
        self._layout()

    def _layout(self) -> None:
        n = len(self.items)
        span = max(0.2, 1.0 - (n - 1) * self.STAGGER)
        back, fade = QtCore.QEasingCurve(QtCore.QEasingCurve.Type.OutBack), QtCore.QEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        for i, (it, rect) in enumerate(zip(self.items, self.targets())):
            local = min(1.0, max(0.0, (self.t - i * self.STAGGER) / span))
            if local <= 0.0:
                it.appear = 0.0
                it.hide()
                continue
            y0 = self._center.y() - rect.height() / 2               # starts tucked behind the button
            y = y0 + (rect.y() - y0) * back.valueForProgress(local)
            it.appear = fade.valueForProgress(local)
            it.setGeometry(rect.x(), int(round(y)), rect.width(), rect.height())
            if not it.isVisible():
                it.show()
            it.update()
        for it in reversed(self.items):          # nearer actions over farther ones while they slide
            it.raise_()
        self.fab.raise_()

    def eventFilter(self, obj, ev):
        t = ev.type()
        if t == QtCore.QEvent.Type.Enter:
            self._close.stop()
            if not self.opened and obj is self.fab:
                self.set_open(True)
        elif t == QtCore.QEvent.Type.Leave and not self.pinned:
            self._close.start(280)
        return False

    def _maybe_close(self) -> None:
        if self.pinned:
            return
        pos = self.parent_w.mapFromGlobal(QtGui.QCursor.pos())
        if any(w.isVisible() and w.geometry().contains(pos) for w in [self.fab] + self.items):
            return
        self.set_open(False)

    def _clicked(self) -> None:
        if self.opened and (self.pinned or self.t >= 1.0 or self._anim.state() == QtCore.QAbstractAnimation.State.Running):
            self.set_open(False)
        else:
            self.set_open(True)
            self.pinned = True

    def _on_state(self, state: str) -> None:
        if any(it.state == "busy" for it in self.items):
            self.fab.set_state("busy")
        elif state in ("done", "error"):
            self.fab.set_state(state)
        elif self.fab.state == "busy":
            self.fab.set_state("idle")


# --- window-level glass (Windows 11 acrylic backdrop) ------------------------------------
def window_glass_supported() -> bool:
    """Windows 11 22H2+ with transparency effects on (and not disabled by DSLOGIC_WINDOW_GLASS=0)."""
    if sys.platform != "win32" or os.environ.get("DSLOGIC_WINDOW_GLASS") == "0" or reduced_transparency():
        return False
    try:
        return sys.getwindowsversion().build >= 22621
    except Exception:
        return False


def enable_window_glass(widget: QtWidgets.QWidget) -> bool:
    """Give a top-level window the system acrylic backdrop (blurred desktop behind it).
    The widget must have WA_TranslucentBackground set before it is first shown."""
    if not window_glass_supported():
        return False
    try:
        import ctypes

        class MARGINS(ctypes.Structure):
            _fields_ = [("l", ctypes.c_int), ("r", ctypes.c_int), ("t", ctypes.c_int), ("b", ctypes.c_int)]

        hwnd = ctypes.c_void_p(int(widget.winId()))
        dwm = ctypes.windll.dwmapi
        dwm.DwmExtendFrameIntoClientArea(hwnd, ctypes.byref(MARGINS(-1, -1, -1, -1)))
        dark = ctypes.c_int(1 if THEME == "dark" else 0)
        dwm.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(dark), ctypes.sizeof(dark))      # title bar
        acrylic = ctypes.c_int(3)                                                          # DWMSBT_TRANSIENTWINDOW
        return dwm.DwmSetWindowAttribute(hwnd, 38, ctypes.byref(acrylic), ctypes.sizeof(acrylic)) == 0
    except Exception:
        return False


def set_window_dark(widget: QtWidgets.QWidget, dark: bool) -> None:
    """Dark or light Windows title bar (no-op elsewhere)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        v = ctypes.c_int(1 if dark else 0)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(ctypes.c_void_p(int(widget.winId())), 20,
                                                   ctypes.byref(v), ctypes.sizeof(v))
    except Exception:
        pass


class CrossFade(QtWidgets.QWidget):
    """A snapshot of the window laid over it that fades away: the theme change dissolves
    instead of flashing."""

    def __init__(self, parent: QtWidgets.QWidget, snapshot: QtGui.QPixmap, ms: int = 320):
        super().__init__(parent)
        self.snap, self.alpha = snapshot, 1.0
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setGeometry(parent.rect())
        self._a = _Anim(self, "alpha", ms, QtCore.QEasingCurve.Type.InOutQuad)
        self._a.a.finished.connect(self.deleteLater)
        self.show()
        self.raise_()
        if _ms(ms) == 0:
            self.deleteLater()
        else:
            self._a.to(0.0)

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.setOpacity(self.alpha)
        p.drawPixmap(self.rect(), self.snap)


class DragCard(QtWidgets.QWidget):
    """A row lifted off the list: a glass card holding a snapshot of the row. `lift`
    (0..1) animates the pick-up (scale and shadow) and the drop."""

    PAD = 14                                   # room for the shadow around the card

    def __init__(self, parent, snapshot: QtGui.QPixmap):
        super().__init__(parent)
        self.snap = snapshot
        self.lift = 0.0
        self._la = _Anim(self, "lift", 180, QtCore.QEasingCurve.Type.OutBack)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        dpr = snapshot.devicePixelRatioF() or 1.0
        self.resize(int(snapshot.width() / dpr) + 2 * self.PAD, int(snapshot.height() / dpr) + 2 * self.PAD)

    def pick_up(self):
        self.show()
        self.raise_()
        self._la.to(1.0)

    def drop(self, target: QtCore.QPoint, done):
        """Glide to `target` (top-left) while settling, then call done()."""
        a = QtCore.QPropertyAnimation(self, b"pos", self)
        a.setDuration(_ms(170))
        a.setEndValue(target)
        a.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        a.finished.connect(done)
        self._la.to(0.0, 170)
        if _ms(170) == 0:
            self.move(target)
            done()
        else:
            a.start()

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        body = QtCore.QRectF(self.rect()).adjusted(self.PAD, self.PAD, -self.PAD, -self.PAD)
        c = body.center()
        s = 1.0 + 0.012 * self.lift
        p.translate(c)
        p.scale(s, s)
        p.translate(-c)
        p.setPen(Qt.PenStyle.NoPen)
        for k in range(8):                                   # shadow deepens as it lifts
            a = int((10 + 22 * self.lift) * (1 - k / 8))
            p.setBrush(QtGui.QColor(0, 0, 0, a))
            p.drawRoundedRect(body.adjusted(-k, k * 0.8 * (0.4 + self.lift), k, k * (0.6 + self.lift)),
                              12 + k, 12 + k)
        path = QtGui.QPainterPath()
        path.addRoundedRect(body, 12, 12)
        p.fillPath(path, _c(RAISED, 0.94))
        p.save()
        p.setClipPath(path)
        p.drawPixmap(body, self.snap, QtCore.QRectF(self.snap.rect()))
        g = QtGui.QLinearGradient(body.topLeft(), body.bottomLeft())
        g.setColorAt(0, QtGui.QColor(255, 255, 255, int(28 + 22 * self.lift)))
        g.setColorAt(0.4, QtGui.QColor(255, 255, 255, 0))
        p.fillPath(path, g)
        p.restore()
        rim = QtGui.QLinearGradient(body.topLeft(), body.bottomRight())
        rim.setColorAt(0, QtGui.QColor(255, 255, 255, 200))
        rim.setColorAt(0.5, _c(ACCENT, 0.5))
        rim.setColorAt(1, QtGui.QColor(255, 255, 255, 60))
        p.setPen(QtGui.QPen(QtGui.QBrush(rim), 1.3))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
