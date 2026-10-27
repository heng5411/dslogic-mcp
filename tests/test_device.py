import struct

import numpy as np
import pytest

from dslogic import constants as C
from dslogic import SPI, CaptureError, DSLogic, Stage
from dslogic.device import OverflowError_

from fake_device import FakeTransport


def counter_wave(n):
    """ch k = bit k of the sample index (ch0 toggles every sample, ch1 every 2, ...)."""
    idx = np.arange(n, dtype=np.uint64)
    return {k: ((idx >> np.uint64(k)) & np.uint64(1)).astype(np.uint8) for k in range(32)}


def make(waveform=counter_wave, firmware_dir=None, **kw):
    t = FakeTransport(waveform, **kw)
    la = DSLogic(t, C.find_profile(0x002C, True), firmware_dir)
    la._bring_up()
    return la, t


def test_bring_up_loads_fpga_and_sets_threshold(tmp_path):
    # a stand-in bitstream: the real one comes from DSView and is not in this repo
    (tmp_path / C.find_profile(0x2C, True).fpga_bitstream).write_bytes(bytes(465304))
    la, t = make(fpga_done=False, firmware_dir=tmp_path)
    assert t.claimed
    assert t.fpga_bytes == 465304
    assert t.regs[C.VTH_ADDR] == int(1.0 / 3.3 * 0.6 * 255)
    la.threshold = 1.65
    assert t.regs[C.VTH_ADDR] == int(1.65 / 3.3 * 0.6 * 255)
    with pytest.raises(ValueError):
        la.threshold = 6
    assert la.fw_version == (2, 0) and la.hdl_version == C.DSL_HDL_VERSION


def test_bring_up_skips_fpga_when_configured():
    la, t = make(fpga_done=True)
    assert t.fpga_bytes == 0


def test_capture_buffer_mode_roundtrip():
    la, t = make(fpga_done=True)
    la.samplerate = "250M"
    la.sample_count = 100_000
    la.enable_channels([0, 1, 2, 5, 31])
    la.set_names({0: "A", 31: "Z"})
    la.set_color("A", "red")
    cap = la.capture(timeout=5)
    assert cap.enabled == [0, 1, 2, 5, 31]
    assert cap.num_samples == 100_000
    ref = counter_wave(cap.num_samples)
    for ch in cap.enabled:
        assert np.array_equal(cap[ch], ref[ch]), ch
    assert cap.channels[0].name == "A" and cap.channels[0].color == "#ff0000"
    assert cap["Z"].shape == (100_000,)
    # the arm block went out right after the BULK_WR command announcing 186 words
    bulk_cmds = [e for e in t.log if e[0] == "w" and e[1] == C.DSL_CTL_BULK_WR]
    assert bulk_cmds[-1][3] == bytes([186, 0, 0])
    assert len(t.bulk[-2]) == 372 and len(t.bulk[-1]) == 204
    assert ("w", C.DSL_CTL_START, 0, b"") in t.log
    assert t.log[-2][:2] == ("w", C.DSL_CTL_STOP)


def test_capture_with_trigger_reports_position():
    la, t = make(fpga_done=True, trigger_at=12_345)
    la.enable_channels(range(4))
    la.set_trigger(0, "rising")
    la.trigger_position = 20
    cap = la.capture(timeout=5)
    assert cap.triggered and cap.trigger_pos == 12_345
    assert t.decoded_setting()["mode"] & 1


def test_buffer_mode_early_stop_uploads_partial():
    la, t = make(fpga_done=True, never_trigger=True)
    la.enable_channels([0, 1])
    la.sample_count = 1_000_000
    la.set_trigger(0, "rising")
    cap = la.capture(timeout=0.3)
    assert t.regs[C.CTR0_ADDR] == C.bmFORCE_STOP
    assert 0 < cap.num_samples < 1_000_000
    assert cap.meta["stopped_early"]


def test_stream_mode_and_overflow():
    la, t = make(fpga_done=True)
    la.operation_mode = "stream"
    la.samplerate = "50M"
    la.sample_count = 50_000
    la.enable_channels(range(8))
    cap = la.capture(timeout=5)
    assert t.decoded_setting()["mode"] >> C.STREAM_MODE_BIT & 1
    assert np.array_equal(cap[7], counter_wave(cap.num_samples)[7])

    la2, t2 = make(fpga_done=True, overflow=True)
    la2.operation_mode = "stream"
    la2.samplerate = "50M"
    la2.sample_count = 10_000_000
    la2.enable_channels(range(32))
    with pytest.raises(OverflowError_):
        la2.capture(timeout=5)


def test_channel_mode_selection_and_validation():
    la, _ = make(fpga_done=True)
    la.enable_channels(range(32))
    la.samplerate = "250M"
    assert la.channel_mode.id == 23
    la.samplerate = "500M"
    with pytest.raises(ValueError, match="no buffer-mode channel mode"):
        la.channel_mode
    la.enable_channels(range(16))
    assert la.channel_mode.id == 24
    la.samplerate = "1G"
    la.enable_channels(range(8))
    assert la.channel_mode.id == 25
    la.operation_mode = "stream"
    la.samplerate = "100M"
    la.enable_channels(range(30))
    assert la.channel_mode.id == 14
    with pytest.raises(ValueError):
        la.samplerate = "123M"
    la.operation_mode = "buffer"
    la.samplerate = "250M"
    la.enable_channels(range(32))
    la.sample_count = la.max_samples() + 1024
    with pytest.raises(ValueError, match="buffer mode holds at most"):
        la.capture()


def test_settings_roundtrip(tmp_path):
    la, _ = make(fpga_done=True)
    la.operation_mode = "stream"
    la.samplerate = "20M"
    la.duration = "5ms"
    la.threshold = 0.9
    la.filter = True
    la.enable_channels([3, 4])
    la.set_names({3: "SDA", 4: "SCL"})
    la.set_color(3, (0, 128, 255))
    la.set_trigger("SDA", "falling")
    la.trigger_position = 30
    la.add_decoder(SPI(clk=4, mosi=3))
    p = tmp_path / "s.json"
    la.save_settings(p)

    lb, _ = make(fpga_done=True)
    lb.load_settings(p)
    assert lb.settings() == la.settings()
    assert lb.sample_count == 100_000 and lb.channel(3).color == "#0080ff"


def test_advanced_trigger_reaches_device():
    la, t = make(fpga_done=True, trigger_at=99)
    la.enable_channels(range(4))
    la.set_advanced_trigger([Stage(cond0={0: "F"}), Stage(cond0={1: "R"}, count=3)])
    cap = la.capture(timeout=5)
    s = t.armed
    assert struct.unpack_from("<16I", s, 48 + 32 * 8)[:2] == (1, 3)
    assert cap.trigger_pos == 99


def test_no_data_raises():
    la, t = make(fpga_done=True, never_trigger=True)
    la.enable_channels([0])
    la.stop_option = "stop"
    la.set_trigger(0, "R")
    with pytest.raises(CaptureError):
        la.capture(timeout=0.2)


def test_describe_spells_out_trigger_and_capture_has_timing():
    la, t = make(fpga_done=True, trigger_at=2000)
    la.enable_channels(range(4))
    la.set_name(1, "SCLK")
    la.set_trigger("SCLK", "edge")
    assert "SCLK any edge (rising or falling)" in la.describe()
    cap = la.capture(timeout=5)
    assert set(cap.meta["timing"]) == {"armed_to_data_s", "transfer_s", "total_s"}
    assert "SCLK" in cap.trigger_cause()


DSVIEW_SESSION = __import__("pathlib").Path(__file__).parent / "data" / "dsview_1.3.2_stream_session.json"


def test_load_settings_from_dsview_session(tmp_path):
    """Session JSON taken from a real DSView 1.3.2 stream capture of two SPI buses."""
    import zipfile
    dsl = tmp_path / "dsview.dsl"
    with zipfile.ZipFile(dsl, "w") as z:
        z.writestr("header", "[version]\nversion = 3\n[header]\n")
        z.writestr("session", DSVIEW_SESSION.read_text())
    la, _ = make(fpga_done=True)
    la.enable_channels(range(32))                 # must be narrowed to the file's channels
    assert la.load_settings(dsl) == []
    assert la.operation_mode == "stream" and la.channel_mode.id == 15
    assert la.samplerate == 250_000_000 and la.sample_count == 16 << 30
    assert la.threshold == 2.5 and la.trigger_position == 1
    assert la.enabled_channels() == list(range(8))
    assert [c.name for c in la.channels[:8]] == ["spi1_cs0", "spi1_sclk", "spi1_miso", "spi1_mosi",
                                                 "gpio7", "spi2_sclk", "spi2_miso", "spi2_mosi"]
    assert la.trigger.simple == {0: "F"}
    assert [d.channels for d in la.decoders] == [{"clk": 1, "miso": 2, "mosi": 3, "cs": 0},
                                                 {"clk": 5, "miso": 6, "mosi": 7, "cs": None}]
    # the same file name works for DSView's own settings files (plain JSON)
    lb, _ = make(fpga_done=True)
    lb.load_settings(DSVIEW_SESSION)
    assert lb.settings() == la.settings()


def test_dsview_advanced_trigger_import():
    import json
    from dslogic.dsview_session import settings_from_dsview
    sess = json.loads(DSVIEW_SESSION.read_text())
    tr = sess["trigger"]
    tr.update(advTriggerMode=True, triggerTab=0, triggerStages=1,
              stageTriggerValue00="X X X X X X X X X X X X X X X F",
              stageTriggerValue01="X X X X X X X X X X X X X X R X",
              stageTriggerCount1=4, stageTriggerContiguous1=True, stageTriggerLogic1=0)
    s, _ = settings_from_dsview(sess)
    st = s["advanced_trigger"]
    assert len(st) == 2 and st[1]["count"] == 4 and st[1]["contiguous"] and st[1]["op"] == "or"
    la, t = make(fpga_done=True, trigger_at=10)
    la.apply_settings(s)
    tab = la.trigger.table()
    assert tab.trigger0[0][0] == "F" and tab.trigger0[1][1] == "R" and tab.stages == 1


def test_large_capture_spools_to_disk(tmp_path):
    la, t = make(fpga_done=True)
    la.spool, la.spool_dir = "always", str(tmp_path)
    la.operation_mode = "stream"
    la.samplerate = "50M"
    la.sample_count = 200_000
    la.enable_channels([0, 3, 6])
    cap = la.capture(timeout=5)
    import numpy as np
    assert cap.meta["spool_dir"] and cap.meta["spool_dir"].startswith(str(tmp_path))
    assert isinstance(cap.data[3].base, np.memmap) or isinstance(cap.data[3], np.memmap)
    assert np.array_equal(cap[6], counter_wave(cap.num_samples)[6])
    cap.release()


def test_ctrl_c_cancels_capture_without_data():
    import signal, threading
    from dslogic import CaptureCancelled
    la, t = make(fpga_done=True, never_trigger=True)
    la.enable_channels([0, 1])
    la.sample_count = 1_000_000
    la.set_trigger(0, "rising")
    threading.Timer(0.3, lambda: signal.raise_signal(signal.SIGINT)).start()
    with pytest.raises(CaptureCancelled):
        la.capture()
    assert t.regs[C.CTR0_ADDR] == C.bmFORCE_RDY            # analyser stopped cleanly
    assert t.log[-2][:2] == ("w", C.DSL_CTL_STOP)
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_stop_key_keeps_data():
    import threading
    from dslogic.keys import CTRL_X, KeyWatcher
    la, t = make(fpga_done=True, never_trigger=True)
    la.enable_channels([0, 1])
    la.sample_count = 1_000_000
    la.set_trigger(0, "rising")
    kw = KeyWatcher({CTRL_X: la.stop})
    threading.Timer(0.3, lambda: kw.feed(CTRL_X)).start()   # as if Ctrl-X was typed
    cap = la.capture()
    assert cap.meta["stopped_early"] and cap.num_samples > 0
