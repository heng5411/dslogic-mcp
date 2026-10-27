import struct

from dslogic import constants as C
from dslogic.setting import ArmParams, build_arm_blocks
from dslogic.trigger import SerialTrigger, Stage, Trigger

U3PRO32_SS = C.find_profile(0x002C, True)


def params(**kw):
    d = dict(profile=U3PRO32_SS, ch_mode=C.CHANNEL_MODES[23], samplerate=C.MHZ(250),
             limit_samples=1_000_000, actual_samples=(1_000_000 + 1023) & ~1023,
             enabled=[0, 1, 2, 3], trigger=Trigger().table(), stream=False)
    d.update(kw)
    return ArmParams(**d)


def u16(b, off):
    return struct.unpack_from("<H", b, off)[0]


def test_sizes_and_sync_words():
    s, e, _ = build_arm_blocks(params())
    assert len(s) == 372 and len(e) == 204
    assert struct.unpack_from("<I", s, 0)[0] == 0xF5A5F5A5
    assert struct.unpack_from("<I", s, 368)[0] == 0xFA5AFA5A
    assert u16(s, 4) == 0x0001 and u16(s, 46) == 0x40A0
    assert u16(e, 4) == 0x6060 and u16(e, 198) == 0xFFFF


def test_basic_fields_buffer_250m():
    s, _, f = build_arm_blocks(params())
    # divider: ceil(500M/250M)=2 -> div_h=(2-1)<<8, then ceil(2/5)=1
    assert (f["div_l"], f["div_h"]) == (1, 0x100)
    assert f["cnt"] == 1_000_448 >> 4
    assert f["ch_en_l"] == 0b1111 and f["ch_en_h"] == 0
    assert f["trig_glb"] == (4 << 8)
    assert f["mode"] == 0                      # no trigger, buffer, internal clock
    assert f["tpos"] == 64                     # minimum one atomic word


def test_mode_bits():
    t = Trigger(simple={0: "F"}).table()
    _, _, f = build_arm_blocks(params(trigger=t, stream=True, ch_mode=C.CHANNEL_MODES[13],
                                      samplerate=C.MHZ(1), clock_external=True, clock_negedge=True,
                                      filter_1t=True))
    m = f["mode"]
    assert m & 1 and m >> C.CLK_TYPE_BIT & 1 and m >> C.CLK_EDGE_BIT & 1
    assert m >> C.FILTER_BIT & 1 and m >> C.STREAM_MODE_BIT & 1
    assert m >> C.SLOW_ACQ_BIT & 1 == 1        # ceil(1e6/1000*4/8) = 500 B/ms < 1024
    _, _, f2 = build_arm_blocks(params())      # 250 MHz x 4 ch = 125000 B/ms
    assert f2["mode"] >> C.SLOW_ACQ_BIT & 1 == 0


def test_half_and_quarter_rate_bits():
    _, _, f = build_arm_blocks(params(samplerate=C.MHZ(500), ch_mode=C.CHANNEL_MODES[24]))
    assert f["mode"] >> C.HALF_MODE_BIT & 1
    _, _, f = build_arm_blocks(params(samplerate=C.MHZ(1000), ch_mode=C.CHANNEL_MODES[25]))
    assert f["mode"] >> C.QUAR_MODE_BIT & 1
    assert (f["div_l"], f["div_h"]) == (1, 0)


def test_divider_low_rate():
    # 10 kHz in buffer mode: ceil(500M/10k)=50000 -> div_h=(5-1)<<8, ceil(50000/5)=10000
    _, _, f = build_arm_blocks(params(samplerate=C.KHZ(10)))
    assert (f["div_l"], f["div_h"]) == (10000, 0x400)
    # 100 Hz: 5e6 -> /5 = 1e6 = 0xF4240 -> div_l=0x4240, div_h=0x400+0xF
    _, _, f = build_arm_blocks(params(samplerate=100))
    assert (f["div_l"], f["div_h"]) == (0x4240, 0x40F)


def test_trigger_position_clamped_and_aligned():
    t = Trigger(simple={0: "R"}, position=50).table()
    _, _, f = build_arm_blocks(params(trigger=t))
    assert f["tpos"] == 500_000 and f["tpos_l"] == 500_000 & 0xFFC0 and f["tpos_h"] == 500_000 >> 16
    # stream mode: at most 10 % of the per-channel depth
    t = Trigger(simple={0: "R"}, position=100).table()
    _, _, f = build_arm_blocks(params(trigger=t, stream=True, limit_samples=10**10,
                                      actual_samples=10**10, ch_mode=C.CHANNEL_MODES[13], samplerate=C.MHZ(50)))
    depth = (2 * 1024 ** 3 // 4) & ~1023
    assert f["tpos"] == depth * 10 // 100


def test_simple_trigger_words():
    t = Trigger(simple={0: "F", 3: "1", 17: "R"}).table()
    s, e, f = build_arm_blocks(params(trigger=t, enabled=[0, 3, 17]))
    assert f["mode"] & 1
    m0, m1, v0, v1, e0, e1 = (u16(s, 48 + 32 * k) for k in range(6))
    assert m0 == 0xFFFF & ~0b1001          # ch0 and ch3 are cared about
    assert v0 == 0b1000                    # ch3 high
    assert e0 == 0b0001                    # ch0 edge
    assert m1 == 0xFFFF and v1 == 0 and e1 == 0
    assert u16(s, 48 + 32 * 6) == 2        # trig_logic0[0] = (1 << 1) + 0
    assert u16(s, 48 + 32 * 0 + 2) == 0xFFFF   # stage 1 mask fully don't-care
    hm0, _, hv0, _, he0, _ = (u16(e, 6 + 32 * k) for k in range(6))
    assert hm0 == 0xFFFF & ~(1 << 1) and hv0 == 1 << 1 and he0 == 1 << 1   # ch17 rising


def test_simple_trigger_half_mode_duplicates_low_byte():
    t = Trigger(simple={2: "R"}).table()
    s, _, _ = build_arm_blocks(params(trigger=t, samplerate=C.MHZ(1000), ch_mode=C.CHANNEL_MODES[25],
                                      enabled=[0, 1, 2]))
    e0 = u16(s, 48 + 32 * 4)
    assert e0 == (0x04 << 8) | 0x04


def test_advanced_trigger_stages():
    trig = Trigger(stages=[Stage(cond0={0: "F"}, count=1),
                           Stage(cond0={1: "R"}, cond1={2: "1"}, op="or", inv1=True, count=8, contiguous=True)])
    s, _, f = build_arm_blocks(params(trigger=trig.table()))
    assert f["trig_glb"] & 0xFF == 1                     # stages - 1
    logic0 = struct.unpack_from("<16H", s, 48 + 32 * 6)
    logic1 = struct.unpack_from("<16H", s, 48 + 32 * 7)
    count = struct.unpack_from("<16I", s, 48 + 32 * 8)
    assert logic0[0] == (1 << 1) and logic1[0] == (1 << 1)
    assert logic0[1] == (0b10 << 1) and logic1[1] == (0b10 << 1) + 1   # contiguous, or, inv1
    assert count[:2] == (1, 8)


def test_serial_trigger_sets_strig_bit():
    st = SerialTrigger(start={0: "0"}, stop={0: "1"}, clock={1: "R"}, data_channel=2, value=0xA5, bits=8)
    s, _, f = build_arm_blocks(params(trigger=Trigger(serial=st).table()))
    assert f["mode"] >> C.STRIG_MODE_BIT & 1
    count = struct.unpack_from("<16I", s, 48 + 32 * 8)
    assert count[1] == 1 and count[3] == 7
