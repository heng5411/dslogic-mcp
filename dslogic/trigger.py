"""Trigger configuration and its encoding into the FPGA trigger words.

Mirrors libsigrok4DSL trigger.c (struct ds_trigger) and the way DSView's GUI fills it
(pv/view/logicsignal.cpp for the simple trigger, pv/dock/triggerdock.cpp for the
advanced and serial triggers).

Per-channel condition codes:
    'X' don't care   '0' low   '1' high   'R' rising   'F' falling   'C' any edge
Friendly aliases are accepted: "rising", "falling", "high", "low", "edge"/"change", "any".
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import constants as C

MAX_PROBES = 32

_ALIASES = {
    "x": "X", "any": "X", "none": "X", "dontcare": "X", "-": "X",
    "0": "0", "low": "0", "l": "0",
    "1": "1", "high": "1", "h": "1",
    "r": "R", "rise": "R", "rising": "R", "posedge": "R", "pos": "R",
    "f": "F", "fall": "F", "falling": "F", "negedge": "F", "neg": "F",
    "c": "C", "edge": "C", "change": "C", "toggle": "C", "both": "C",
}


COND_TEXT = {"X": "don't care", "0": "low", "1": "high", "R": "rising edge",
             "F": "falling edge", "C": "any edge (rising or falling)"}


def describe_condition(code: str) -> str:
    return COND_TEXT.get(cond_code(code), code)


def cond_code(value: str) -> str:
    try:
        return _ALIASES[str(value).strip().lower()]
    except KeyError:
        raise ValueError(f"unknown trigger condition {value!r} "
                         "(use X/0/1/R/F/C or low/high/rising/falling/edge)") from None


def _conditions(spec: dict[int, str] | str | None) -> list[str]:
    """dict {channel: cond} or DSView-style pattern string (first char = highest channel)."""
    row = ["X"] * MAX_PROBES
    if spec is None:
        return row
    if isinstance(spec, str):
        chars = spec.replace(" ", "")
        if len(chars) > MAX_PROBES:
            raise ValueError("pattern longer than 32 channels")
        for j, ch in enumerate(chars):
            row[len(chars) - 1 - j] = cond_code(ch)
        return row
    for ch, c in spec.items():
        ch = int(ch)
        if not 0 <= ch < MAX_PROBES:
            raise ValueError(f"trigger channel {ch} out of range")
        row[ch] = cond_code(c)
    return row


@dataclass
class Stage:
    """One advanced-trigger stage (DSView "Stage N" tab).

    The stage matches when (cond0 [!=] true) <op> (cond1 [!=] true) holds `count` times
    (consecutively if `contiguous`).  cond0/cond1 are {channel: condition}.
    """
    cond0: dict[int, str] | str | None = None
    cond1: dict[int, str] | str | None = None
    op: str = "and"                 # "and" | "or"
    inv0: bool = False              # True = "!=" for cond0
    inv1: bool = False
    count: int = 1
    contiguous: bool = False

    def logic(self) -> int:
        if self.op.lower() not in ("and", "or"):
            raise ValueError("Stage.op must be 'and' or 'or'")
        return (int(self.contiguous) << 1) + (1 if self.op.lower() == "and" else 0)


@dataclass
class Trigger:
    """Trigger setup for one capture.

    simple:   {channel: condition}; all listed conditions must hold together (AND).
    stages:   advanced trigger (1..16 stages, matched in order). Overrides `simple`.
    serial:   SerialTrigger. Overrides both.
    position: pre-trigger amount in percent of the capture (0..100). The hardware caps
              it at 90 % in buffer mode and 10 % of the per-channel depth in stream mode.
    """
    simple: dict[int, str] = field(default_factory=dict)
    stages: list[Stage] = field(default_factory=list)
    serial: "SerialTrigger | None" = None
    position: int = 0

    @property
    def mode(self) -> int:
        if self.serial is not None:
            return C.SERIAL_TRIGGER
        if self.stages:
            return C.ADV_TRIGGER
        return C.SIMPLE_TRIGGER

    @property
    def enabled(self) -> bool:
        if self.mode != C.SIMPLE_TRIGGER:
            return True
        return any(cond_code(c) != "X" for c in self.simple.values())

    def table(self) -> "TriggerTable":
        t = TriggerTable()
        t.pos = max(0, min(100, int(self.position)))
        t.mode = self.mode
        t.en = self.enabled
        if self.serial is not None:
            self.serial.fill(t)
        elif self.stages:
            if len(self.stages) > C.NUM_TRIGGER_STAGES:
                raise ValueError("at most 16 trigger stages")
            t.stages = len(self.stages) - 1
            for i, st in enumerate(self.stages):
                if st.count < 1:
                    raise ValueError("Stage.count must be >= 1")
                t.trigger0[i] = _conditions(st.cond0)
                t.trigger1[i] = _conditions(st.cond1)
                t.logic[i] = st.logic()
                t.inv0[i] = int(st.inv0)
                t.inv1[i] = int(st.inv1)
                t.count0[i] = int(st.count)
        else:
            t.trigger0[C.NUM_TRIGGER_STAGES] = _conditions(self.simple)
        return t


@dataclass
class SerialTrigger:
    """DSView "Serial Trigger" tab (experimental: encoded exactly like DSView, not yet
    verified on hardware).  The trigger fires when, between the `start` and `stop`
    conditions, `bits` bits shifted in from `data_channel` on each `clock` condition
    equal `value`.
    """
    start: dict[int, str]
    stop: dict[int, str]
    clock: dict[int, str]
    data_channel: int
    value: int | str
    bits: int = 8

    def fill(self, t: "TriggerTable") -> None:
        if not 1 <= self.bits <= 16:
            raise ValueError("serial trigger bits must be 1..16")
        if not 0 <= self.data_channel < MAX_PROBES:
            raise ValueError("serial data channel out of range")
        t.stages = 3
        t.trigger0[0] = _conditions(self.start)
        t.trigger1[0] = _conditions(self.stop)
        t.trigger0[1] = _conditions(self.clock)
        # stage 2: DSView passes the low-16 channel string as trigger0 and the
        # high-16 ("ext32") string as trigger1, both through a 16-probe call.
        lo, hi = ["X"] * MAX_PROBES, ["X"] * MAX_PROBES
        if self.data_channel < 16:
            lo[self.data_channel] = "0"
        else:
            hi[self.data_channel - 16] = "0"
        t.trigger0[2], t.trigger1[2] = lo, hi
        if isinstance(self.value, str):
            pattern = self.value.replace(" ", "")
        else:
            pattern = "".join("1" if (self.value >> b) & 1 else "0"
                              for b in reversed(range(self.bits)))
        pattern = pattern.rjust(16, "X")
        t.trigger0[C.S_TRIGGER_DATA_STAGE] = _conditions(pattern)
        t.count0[1] = 1
        t.count0[C.S_TRIGGER_DATA_STAGE] = self.bits - 1
        for i in range(4):
            t.logic[i] = 1


class TriggerTable:
    """struct ds_trigger: rows 0..15 are advanced stages, row 16 is the simple trigger."""

    def __init__(self):
        n = C.NUM_TRIGGER_STAGES + 1
        self.en = False
        self.mode = C.SIMPLE_TRIGGER
        self.pos = 0
        self.stages = 0
        self.trigger0 = [["X"] * MAX_PROBES for _ in range(n)]
        self.trigger1 = [["X"] * MAX_PROBES for _ in range(n)]
        self.count0 = [0] * n
        self.inv0 = [0] * n
        self.inv1 = [0] * n
        self.logic = [1] * n

    # ds_trigger_get_{mask,value,edge}{0,1}
    def _word(self, row, msc, lsc, ones, qutr, half) -> int:
        w = 0
        for i in range(msc, lsc - 1, -1):
            w = ((w << 1) + (row[i] in ones)) & 0xFFFF
        if qutr:
            q = w & 0x000F
            w = (q << 12) + (q << 8) + (q << 4) + q
        elif half:
            h = w & 0x00FF
            w = (h << 8) + h
        return w & 0xFFFF

    def words(self, stage, msc, lsc, qutr=False, half=False):
        """(mask0, mask1, value0, value1, edge0, edge1) for one stage/channel half."""
        r0, r1 = self.trigger0[stage], self.trigger1[stage]
        return (self._word(r0, msc, lsc, "XC", qutr, half),
                self._word(r1, msc, lsc, "XC", qutr, half),
                self._word(r0, msc, lsc, "1R", qutr, half),
                self._word(r1, msc, lsc, "1R", qutr, half),
                self._word(r0, msc, lsc, "RFC", qutr, half),
                self._word(r1, msc, lsc, "RFC", qutr, half))
