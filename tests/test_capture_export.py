import json
import re
import zipfile

import numpy as np
import pytest

from dslogic import I2C, SPI, UART, Capture, DSViewDecoder
from dslogic.capture import parse_count, parse_rate, parse_time

from signals import i2c, spi, uart


def rand_capture(n=5000, chans=(0, 3, 9), rate=100_000_000, seed=1):
    rng = np.random.default_rng(seed)
    return Capture.from_arrays(rate, {c: rng.integers(0, 2, n) for c in chans})


def test_from_cross_splits_words():
    # 3 channels, 2 words each: word k of channel c = (c+1) * 0x0101010101010101 * (k+1)
    words = []
    for k in range(2):
        for c in range(3):
            words.append(np.uint64(0x0101010101010101) * np.uint64((c + 1) * (k + 1)))
    raw = np.array(words, dtype="<u8").tobytes()
    cap = Capture.from_cross(raw, [0, 5, 7], 1_000_000)
    assert cap.num_samples == 128
    assert cap.data[5][:8].tolist() == [2] * 8 and cap.data[5][8:].tolist() == [4] * 8
    assert cap[7][:8].tolist() == [1, 1, 0, 0, 0, 0, 0, 0]


def test_samples_window_and_edges():
    cap = Capture.from_arrays(1000, {0: [0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1]})
    assert cap.samples(0, 3, 9).tolist() == [1, 0, 1, 0, 0, 1]
    assert cap.edges(0, "rising").tolist() == [2, 5, 8]
    assert cap.edges(0, "falling").tolist() == [4, 6]


def test_dsl_roundtrip_with_names_colors_decoders(tmp_path):
    cap = rand_capture(n=70_000)
    cap.set_name(0, "CS")
    cap.set_color(3, "#00ff00")
    cap.trigger_pos, cap.triggered = 1234, True
    cap.add_decoder(SPI(clk=3, mosi=9, cs="CS", cpol=1, cpha=1))
    cap.add_decoder(DSViewDecoder("can", {"can_rx": 9}, {"bitrate": 500000}, label="CAN"))
    p = cap.save(tmp_path / "x.dsl")
    with zipfile.ZipFile(p) as z:
        names = z.namelist()
        assert names[:3] == ["header", "decoders", "session"]
        assert {"L-0/0", "L-3/0", "L-9/0"} <= set(names)
        header = z.read("header").decode()
        assert "samplerate = 100 MHz" in header and "probe0 = CS" in header
        assert "total probes = 3" in header and "trigger pos = 1234" in header
        sess = json.loads(z.read("session"))
        assert sess["channel"][1]["colour"] == "#00ff00"
        dec = json.loads(z.read("decoders"))
        assert dec[0]["id"] == "0:spi" and {"cs": 0} in dec[0]["channel"]
        assert dec[0]["options"]["cpol"] == 1
    back = Capture.load(p)
    assert back.samplerate == cap.samplerate and back.num_samples == cap.num_samples
    for c in cap.enabled:
        assert np.array_equal(back[c], cap[c])
    assert back.channels[0].name == "CS" and back.channels[3].color == "#00ff00"
    assert back.trigger_pos == 1234
    assert [d.id for d in back.decoders] == ["0:spi", "can"]
    assert back.decoders[0].options["cpha"] == 1


def test_dsl_multi_block(tmp_path):
    n = (1 << 24) + 4096                      # spills into a second 2 MiB block
    cap = Capture(samplerate=1_000_000, num_samples=n,
                  data={1: np.full((n + 7) // 8, 0xA5, np.uint8)})
    p = cap.save(tmp_path / "big.dsl")
    with zipfile.ZipFile(p) as z:
        assert z.getinfo("L-1/0").file_size == 1 << 21
        assert z.getinfo("L-1/1").file_size == 512
        assert "total blocks = 2" in z.read("header").decode()
    assert np.array_equal(Capture.load(p).data[1], cap.data[1])


def test_dsl_readable_by_minimal_loader(tmp_path):
    """A minimal third-party reader (header regex + L-<ch>/<block>) can parse the file."""
    cap = rand_capture(n=8192, chans=range(8), rate=250_000_000)
    p = cap.save(tmp_path / "x.dsl")
    d = tmp_path / "unz"
    zipfile.ZipFile(p).extractall(d)
    header = (d / "header").read_text()
    nb = int(re.search(r"total blocks = (\d+)", header).group(1))
    m = re.search(r"samplerate = ([\d.]+) (MHz|KHz|kHz|GHz)", header)
    assert float(m.group(1)) * 1e6 == 250e6
    for c in range(8):
        raw = b"".join((d / f"L-{c}" / str(i)).read_bytes() for i in range(nb))
        assert raw == cap.data[c].tobytes()


def test_npz_vcd_csv_sr(tmp_path):
    cap = Capture.from_arrays(1_000_000, {0: [0, 1, 1, 0, 0, 1], 2: [1, 1, 0, 0, 1, 1]})
    cap.set_name(2, "B")
    back = Capture.load(cap.save(tmp_path / "a.npz"))
    assert back[0].tolist() == [0, 1, 1, 0, 0, 1] and back.channels[2].name == "B"

    vcd = cap.save(tmp_path / "a.vcd").read_text()
    assert "$timescale 1 us $end" in vcd and "$var wire 1 \" B $end" in vcd
    assert "#1\n1!\n" in vcd and "#2\n0\"\n" in vcd

    csv = cap.save(tmp_path / "a.csv").read_text().splitlines()
    assert csv[0] == "sample,time_s,0,B"
    assert csv[1:] == ["0,0,0,1", "1,1e-06,1,1", "2,2e-06,1,0", "3,3e-06,0,0", "4,4e-06,0,1", "5,5e-06,1,1"]

    with zipfile.ZipFile(cap.save(tmp_path / "a.sr")) as z:
        assert z.read("version") == b"2"
        assert "unitsize=1" in z.read("metadata").decode()
        assert list(z.read("logic-1-1")) == [2, 3, 1, 0, 2, 3]


@pytest.mark.parametrize("cpol,cpha", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_spi_decoder_all_modes(cpol, cpha):
    mosi_bytes, miso_bytes = [0x9F, 0x00, 0xA5, 0x3C], [0xFF, 0xC2, 0x20, 0x16]
    cs, clk, mosi, miso = spi(mosi_bytes, miso_bytes, cpol=cpol, cpha=cpha)
    cap = Capture.from_arrays(100_000_000, {0: cs, 1: clk, 2: miso, 3: mosi})
    cap.set_name(0, "CS")
    dec = SPI(clk=1, mosi=3, miso=2, cs="CS", cpol=cpol, cpha=cpha)
    tr = dec.transfers(cap)
    assert len(tr) == 1
    assert tr[0].data["mosi"] == mosi_bytes and tr[0].data["miso"] == miso_bytes
    words = [a for a in dec.decode(cap) if a.kind == "word"]
    assert [w.data["mosi"] for w in words] == mosi_bytes


def test_i2c_decoder():
    scl, sda = i2c(0x50, 0, [0x12, 0xAB])
    cap = Capture.from_arrays(10_000_000, {4: scl, 5: sda})
    anns = I2C(scl=4, sda=5).decode(cap)
    kinds = [a.kind for a in anns]
    assert kinds == ["start", "address", "data", "data", "stop"]
    assert anns[1].data == {"address": 0x50, "rw": "write", "ack": True}
    assert [a.data["value"] for a in anns if a.kind == "data"] == [0x12, 0xAB]


def test_uart_decoder_with_parity():
    line = uart(b"Hi!\x00\xff", spb=16, parity="even")
    cap = Capture.from_arrays(16 * 115200, {6: line})
    dec = UART(rx=6, baudrate=115200, parity_type="even")
    anns = dec.decode(cap)
    assert [a.data["value"] for a in anns] == list(b"Hi!\x00\xff")
    assert not any(a.data["framing_error"] or a.data["parity_error"] for a in anns)
    assert UART(rx=6, baudrate=115200, parity_type="odd").decode(cap)[0].data["parity_error"]


def test_slice_keeps_names_and_shifts_trigger():
    cap = rand_capture()
    cap.set_name(3, "X")
    cap.trigger_pos = 1000
    s = cap.slice(500, 1500)
    assert s.num_samples == 1000 and s.trigger_pos == 500 and s.channels[3].name == "X"
    assert np.array_equal(s[3], cap[3][500:1500])


def test_parsers():
    assert parse_rate("250M") == 250_000_000 and parse_rate("1GHz") == 10**9 and parse_rate("100k") == 100_000
    assert parse_count("10M") == 10**7 and parse_count("64Mi") == 64 << 20
    assert parse_time("20ms") == pytest.approx(0.02) and parse_time("500us") == pytest.approx(5e-4)


def test_spi_reports_cs_window_without_clock():
    import numpy as np
    n = 10_000
    cs = np.ones(n, np.uint8); cs[1000:] = 0          # CS falls, never any clock (seen on hardware)
    cap = Capture.from_arrays(250_000_000, {4: cs, 5: np.zeros(n), 7: np.zeros(n)})
    anns = SPI(clk=5, mosi=7, cs=4).decode(cap)
    assert len(anns) == 1 and anns[0].data["bits"] == 0 and anns[0].start == 1000


def test_spi_merges_cs_glitches_like_hardware():
    """A CS line can rise for 1 sample inside a frame (probe crosstalk)."""
    cs, clk, mosi, miso = spi([0x08, 0x50, 0x04], [0x00, 0x09, 0x00], spb=8)
    cs = cs.copy()
    cs[60] = 1                                  # 1-sample glitch mid-frame
    cs[5] = 0                                   # 1-sample clockless CS pulse before the frame
    cap = Capture.from_arrays(250_000_000, {0: cs, 1: clk, 2: miso, 3: mosi})
    tr = SPI(clk=1, mosi=3, miso=2, cs=0).transfers(cap)
    assert len(tr) == 1 and tr[0].data["mosi"] == [0x08, 0x50, 0x04]


def test_trigger_cause_and_timing_text():
    import numpy as np
    s = np.zeros(1000, np.uint8); s[100:] = 1
    cap = Capture.from_arrays(1_000_000, {5: s})
    cap.set_name(5, "SCLK")
    cap.channels[5].trigger = "C"
    cap.trigger_pos, cap.triggered = 100, True
    assert cap.trigger_cause() == "SCLK rising edge"
    assert "trigger @ 0.1 ms (sample 100): SCLK rising edge" in cap.summary()
    cap.meta = {"operation_mode": "buffer",
                "timing": {"armed_to_data_s": 3.5, "transfer_s": 0.25, "total_s": 3.75}}
    t = cap.timing_text()
    assert "100.000 us before trigger, 900.000 us after" in t and "uploaded in 0.25 s" in t
