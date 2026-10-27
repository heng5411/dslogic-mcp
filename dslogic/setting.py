"""Packing of the FPGA arm block (struct DSL_setting / DSL_setting_ext32 in dsl.h).

`build_arm_blocks()` is a direct port of the computation in dsl_fpga_arm() (dsl.c).
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from . import constants as C
from .trigger import TriggerTable

SETTING_SIZE = 372          # sizeof(struct DSL_setting)
SETTING_EXT32_SIZE = 204    # sizeof(struct DSL_setting_ext32)


@dataclass
class ArmParams:
    profile: C.Profile
    ch_mode: C.ChannelMode
    samplerate: int
    limit_samples: int
    actual_samples: int
    enabled: list[int]              # enabled channel indices
    trigger: TriggerTable
    stream: bool
    clock_external: bool = False
    clock_negedge: bool = False
    rle: bool = False
    filter_1t: bool = False
    internal_test: bool = False

    @property
    def ch_num(self) -> int:
        return max(1, len(self.enabled))   # dsl_en_ch_num()

    def channel_depth(self) -> int:
        return (self.profile.hw_depth // self.ch_num) & ~C.SAMPLES_ALIGN

    def bytes_per_ms(self) -> int:
        return math.ceil(self.samplerate / 1000.0 * self.ch_num / 8)


def build_arm_blocks(p: ArmParams) -> tuple[bytes, bytes, dict]:
    """Return (setting, setting_ext32, fields) ready for the bulk-OUT endpoint.
    `fields` holds the computed scalar values (for logging and tests)."""
    t = p.trigger
    mode = ((int(t.en) << C.TRIG_EN_BIT)
            | (int(p.clock_external) << C.CLK_TYPE_BIT)
            | (int(p.clock_negedge) << C.CLK_EDGE_BIT)
            | (int(p.rle) << C.RLE_MODE_BIT)
            | (int(p.samplerate == p.profile.half_samplerate) << C.HALF_MODE_BIT)
            | (int(p.samplerate == p.profile.quarter_samplerate) << C.QUAR_MODE_BIT)
            | (int(p.filter_1t) << C.FILTER_BIT)
            | (int(p.bytes_per_ms() < 1024) << C.SLOW_ACQ_BIT)
            | (int(t.mode == C.SERIAL_TRIGGER) << C.STRIG_MODE_BIT)
            | (int(p.stream) << C.STREAM_MODE_BIT)
            | (int(p.internal_test) << C.INT_TEST_BIT))

    # sample rate divider (logic mode branch)
    m = p.ch_mode
    div = math.ceil(m.hw_max_samplerate * 1.0 / p.samplerate)
    div_h = ((m.pre_div - 1) if div >= m.pre_div else (div - 1)) << 8
    div = math.ceil(div * 1.0 / m.pre_div)
    div_l = div & 0xFFFF
    div_h = (div_h + (div >> 16)) & 0xFFFF

    cnt = p.actual_samples >> 4
    cnt_l, cnt_h = cnt & 0xFFFF, (cnt >> 16) & 0xFFFF
    dso_cnt_l, dso_cnt_h = p.actual_samples & 0xFFFF, (p.actual_samples >> 16) & 0xFFFF

    tpos = max(int(t.pos / 100.0 * p.limit_samples) & 0xFFFFFFFF, C.ATOMIC_SAMPLES)
    if p.stream:
        tpos = min(tpos, p.channel_depth() * 10 // 100)
    else:
        tpos = min(tpos, p.channel_depth() * C.DS_MAX_TRIG_PERCENT // 100)
    tpos_l = tpos & C.ATOMIC_MASK
    tpos_h = (tpos >> 16) & 0xFFFF

    trig_glb = ((p.ch_num & 0x1F) << 8) + (t.stages & 0xFF)

    ch_en_l = sum(1 << i for i in p.enabled if i < 16)
    ch_en_h = sum(1 << (i - 16) for i in p.enabled if i >= 16)

    adf = bool(p.profile.feature_caps & C.CAPS_FEATURE_ADF4360)
    quar = bool(mode & (1 << C.QUAR_MODE_BIT))
    half = bool(mode & (1 << C.HALF_MODE_BIT))
    qutr_trig = (not adf) and quar
    half_trig = ((not adf) and half) or (adf and quar)

    n = C.NUM_TRIGGER_STAGES
    lo = {k: [0] * n for k in ("m0", "m1", "v0", "v1", "e0", "e1")}
    hi = {k: [0] * n for k in ("m0", "m1", "v0", "v1", "e0", "e1")}
    logic0, logic1, count = [0] * n, [0] * n, [0] * n

    def put(i, src_stage, q, h):
        for d, (msc, lsc) in ((lo, (C.TRIGGER_PROBES - 1, 0)),
                              (hi, (2 * C.TRIGGER_PROBES - 1, C.TRIGGER_PROBES))):
            w = t.words(src_stage, msc, lsc, q, h)
            for k, v in zip(("m0", "m1", "v0", "v1", "e0", "e1"), w):
                d[k][i] = v

    if t.mode == C.SIMPLE_TRIGGER:
        s = C.NUM_TRIGGER_STAGES
        put(0, s, qutr_trig, half_trig)
        logic0[0] = (t.logic[s] << 1) + t.inv0[s]
        logic1[0] = (t.logic[s] << 1) + t.inv1[s]
        count[0] = t.count0[s]
        for i in range(1, n):
            for d in (lo, hi):
                d["m0"][i] = d["m1"][i] = 0xFFFF
            logic0[i] = logic1[i] = 2
    else:
        for i in range(n):
            if (mode & (1 << C.STRIG_MODE_BIT)) and i == C.S_TRIGGER_DATA_STAGE:
                put(i, i, False, False)
            else:
                put(i, i, qutr_trig, half_trig)
            logic0[i] = (t.logic[i] << 1) + t.inv0[i]
            logic1[i] = (t.logic[i] << 1) + t.inv1[i]
            count[i] = t.count0[i]

    h16 = "<" + "H" * n
    setting = b"".join((
        struct.pack("<I", 0xF5A5F5A5),
        struct.pack("<2H", 0x0001, mode & 0xFFFF),
        struct.pack("<3H", 0x0102, div_l, div_h),
        struct.pack("<3H", 0x0302, cnt_l, cnt_h),
        struct.pack("<3H", 0x0502, tpos_l, tpos_h),
        struct.pack("<2H", 0x0701, trig_glb & 0xFFFF),
        struct.pack("<3H", 0x0802, dso_cnt_l, dso_cnt_h),
        struct.pack("<3H", 0x0A02, ch_en_l, ch_en_h),
        struct.pack("<2H", 0x0C01, 0),          # fgain: digi_fgain of probe 0 (0 for LA)
        struct.pack("<H", 0x40A0),
        *(struct.pack(h16, *lo[k]) for k in ("m0", "m1", "v0", "v1", "e0", "e1")),
        struct.pack(h16, *[v & 0xFFFF for v in logic0]),
        struct.pack(h16, *[v & 0xFFFF for v in logic1]),
        struct.pack("<" + "I" * n, *[v & 0xFFFFFFFF for v in count]),
        struct.pack("<I", 0xFA5AFA5A),
    ))
    ext32 = b"".join((
        struct.pack("<I", 0xF5A5F5A5),
        struct.pack("<H", 0x6060),
        *(struct.pack(h16, *hi[k]) for k in ("m0", "m1", "v0", "v1", "e0", "e1")),
        struct.pack("<H", 0xFFFF),
        struct.pack("<I", 0xFA5AFA5A),
    ))
    assert len(setting) == SETTING_SIZE and len(ext32) == SETTING_EXT32_SIZE
    fields = dict(mode=mode, div_l=div_l, div_h=div_h, cnt=cnt, tpos=tpos, tpos_l=tpos_l,
                  tpos_h=tpos_h, trig_glb=trig_glb, ch_en_l=ch_en_l, ch_en_h=ch_en_h)
    return setting, ext32, fields
