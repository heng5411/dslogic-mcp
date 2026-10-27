import numpy as np
import pytest

from dslogic import Capture
from dslogic.overview import UNITS, Overview, get_overview


def busy_cap(n=(1 << 25) + 12345, seed=2):
    """Mostly idle line with bursts of toggling, like real bus traffic."""
    rng = np.random.default_rng(seed)
    a = np.zeros(n, np.uint8)
    for start in rng.integers(0, n - 50_000, 40):
        a[start:start + rng.integers(10, 40_000)] = rng.integers(0, 2, 1)[0]
        a[start: start + 2000: 7] ^= 1
    a[-5:] = 1
    return Capture.from_arrays(250_000_000, {0: a, 3: 1 - a})


def raw_env(cap, ch, x):
    """Reference: each bucket from the raw samples."""
    s = cap.samples(ch)
    edges = list(x) + [cap.num_samples]
    lo = [int(s[a:b].all()) for a, b in zip(edges, edges[1:])]
    hi = [int(s[a:b].any()) for a, b in zip(edges, edges[1:])]
    return np.array(lo), np.array(hi)


@pytest.mark.parametrize("lvl", [0, 1, 2])
def test_levels_match_raw(lvl):
    cap = busy_cap()
    ov = Overview(cap)
    ov.build([0, 3], background=False)
    assert ov.complete(0) and ov.complete(3)
    us = UNITS[lvl] * 8
    width = max(4, min(300, cap.num_samples // us // 2))
    x, lo, hi = ov.buckets(0, lvl, 0, cap.num_samples, width)
    rlo, rhi = raw_env(cap, 0, x)
    assert np.array_equal(lo.astype(int), rlo) and np.array_equal(hi.astype(int), rhi)


def test_unbuilt_blocks_are_nan_and_fill_in():
    cap = busy_cap()
    ov = Overview(cap)
    ov._alloc(0)
    ov._build_block(0, 0)                    # only the first 16 Mi samples
    x, lo, hi = ov.buckets(0, 2, 0, cap.num_samples, 10)
    assert not np.isnan(lo[0]) and np.isnan(lo[-1])


def test_disk_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "cache"))
    cap = busy_cap()
    p = cap.save(tmp_path / "c.dsl")
    a = Capture.load(p, lazy=True)
    ov = get_overview(a)
    ov.build([0, 3], background=False)
    files = list((tmp_path / "cache").rglob("*.npz"))
    assert len(files) == 1
    b = Capture.load(p, lazy=True)
    ob = get_overview(b)
    ob._alloc(0)
    ob._alloc(3)
    assert ob.load_cache([0, 3]) and ob.complete(0)
    for lvl in range(3):
        assert np.array_equal(ob.mn[0][lvl], ov.mn[0][lvl]) and np.array_equal(ob.mx[3][lvl], ov.mx[3][lvl])


def _strip(anns):
    return [(a.start, a.end, a.kind, a.text) for a in anns]


def test_decoders_same_with_and_without_edge_index():
    from dslogic import I2C, SPI, UART
    from signals import i2c, spi, uart
    cs, clk, mosi, miso = spi([0x9F, 0x01, 0xA5], [0x00, 0xC2, 0x3C], spb=8)
    scl, sda = i2c(0x50, 1, [0x12, 0xAB])
    line = uart(b"Hi!", spb=16, parity="even")
    pad = 70_000
    def place(x, fill):
        out = np.full(pad * 3, fill, np.uint8)
        out[pad:pad + len(x)] = x
        return out
    arrays = {0: place(cs, 1), 1: place(clk, 0), 2: place(miso, 0), 3: place(mosi, 0),
              4: place(scl, 1), 5: place(sda, 1), 6: place(line, 1), 7: 1 - place(line, 1)}
    decs = [SPI(clk=1, mosi=3, miso=2, cs=0), I2C(scl=4, sda=5),
            UART(rx=6, baudrate=115200, parity_type="even"),
            UART(rx=7, baudrate=115200, parity_type="even", invert=True)]
    plain = Capture.from_arrays(16 * 115200, arrays)
    want = [_strip(d.decode(plain)) for d in decs]
    assert want[2] == want[3] and len(want[0]) > 3 and len(want[1]) == 5
    indexed = Capture.from_arrays(16 * 115200, arrays)
    get_overview(indexed).build(list(arrays), background=False)
    assert all(get_overview(indexed).edges_of(c) is not None for c in arrays)
    assert [_strip(d.decode(indexed)) for d in decs] == want
    # windows give the same answers too
    w0, w1 = pad - 10, pad + 3000
    assert _strip(decs[0].decode(indexed, w0, w1)) == _strip(decs[0].decode(plain, w0, w1))


def test_dense_blocks_fall_back_to_scanning():
    from dslogic.overview import BLOCK
    n = BLOCK * 8 * 2
    a = np.zeros(n, np.uint8)
    a[: BLOCK * 8: 2] = 1                        # first block toggles every sample: too dense
    cap = Capture.from_arrays(1e9, {0: a})
    ov = get_overview(cap)
    ov.build([0], background=False)
    assert ov.blk_edges[0][0] is None and ov.edges_of(0) is None
    ref = np.flatnonzero(np.diff(a)) + 1
    assert np.array_equal(cap.edges(0), ref)     # still exact via the scan path


@pytest.mark.parametrize("flip_at_unit", [1, 2, 7])
def test_edge_exactly_at_start_of_idle_stretch(flip_at_unit):
    """A busy 4 Ki-sample stretch followed by a flat one, with the level changing on the
    flat stretch's first sample: that edge must not be lost (it inverted every level after it)."""
    us = UNITS[0] * 8                                   # samples per overview unit
    n = 16 * us
    x = np.zeros(n, np.uint8)
    b = flip_at_unit * us                               # first sample of the idle stretch
    x[b - 300:b - 200] = 1                              # activity in the stretch before it
    x[b - 50:b] = 1                                     # high until the boundary...
    x[b + 3 * us:] = 1                                  # ...low from b, high again later
    x[b + 3 * us + 100] = 0                             # (a busy unit again)
    cap = Capture.from_arrays(1_000_000, {0: x})
    truth = np.flatnonzero(np.diff(x.astype(np.int8))) + 1
    ov = get_overview(cap)
    ov.build([0], background=False)
    assert np.array_equal(ov.edges_of(0), truth)
    probe = np.arange(0, n, 97)
    assert np.array_equal(cap.levels_at(0, probe), x[probe])
    cap.meta["_overview"]._edges_all[0] = None          # no index: the overview-guided scan
    assert np.array_equal(cap.edges(0), truth)
