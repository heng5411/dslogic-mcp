"""Randomised end-to-end checks: every sample must survive capture (with USB data arriving
in odd-sized pieces), .dsl save/lazy load and the export formats, bit for bit."""
import numpy as np
import pytest

from dslogic import constants as C
from dslogic import Capture, DSLogic
from dslogic.overview import get_overview

from export_parsers import PARSE
from fake_device import FakeTransport


class ChoppyTransport(FakeTransport):
    """Hands the data over in random-sized pieces (short packets, sizes that split the
    64-bit words of a channel group)."""

    def __init__(self, *a, rng, **kw):
        super().__init__(*a, **kw)
        self.rng = rng

    def handle_events(self, timeout_s):
        if self.queue and self.pos >= 1024 and self.queue[0].size != 1024:
            r = self.queue[0]
            full = r.size
            r.size = max(1, min(full, int(self.rng.choice([1, 7, 23, 24, 511, 4095, full,
                                                           int(self.rng.integers(1, full + 1))]))))
            try:
                return super().handle_events(timeout_s)
            finally:
                r.size = full
        return super().handle_events(timeout_s)


def random_signal(rng, n):
    kind = rng.integers(0, 4)
    if kind == 0:
        return rng.integers(0, 2, n).astype(np.uint8)
    if kind == 1:
        e = np.unique(rng.integers(1, max(2, n), int(rng.integers(0, 40))))
        return (np.searchsorted(e, np.arange(n), side="right") & 1).astype(np.uint8)
    if kind == 2:
        return ((np.arange(n) // int(rng.integers(1, 300))) & 1).astype(np.uint8)
    return np.full(n, rng.integers(0, 2), np.uint8)


@pytest.mark.parametrize("seed", range(6))
def test_capture_survives_choppy_usb(seed, tmp_path):
    rng = np.random.default_rng(seed)
    store = {}

    def wave(n):
        store.update({ch: random_signal(rng, n) for ch in range(32)})
        return store

    for _ in range(8):
        t = ChoppyTransport(wave, fpga_done=True, rng=rng)
        la = DSLogic(t, C.find_profile(0x002C, True))
        la._bring_up()
        stream = bool(rng.integers(0, 2))
        la.operation_mode = "stream" if stream else "buffer"
        la.samplerate = "50M" if stream else "100M"
        chans = sorted(rng.choice(16, int(rng.integers(1, 17)), replace=False).tolist())
        la.enable_channels(chans)
        la.sample_count = int(rng.integers(1024, 200_000))
        if rng.integers(0, 3) == 0:
            la.spool, la.spool_dir = "always", str(tmp_path)
        cap = la.capture(timeout=10)
        assert cap.num_samples == la.sample_count and cap.enabled == chans
        for ch in chans:
            assert np.array_equal(cap[ch], store[ch][:cap.num_samples]), (stream, chans, ch)
        cap.release()


@pytest.mark.parametrize("seed", range(4))
def test_files_roundtrip(seed, tmp_path):
    rng = np.random.default_rng(100 + seed)
    for it in range(6):
        n = int(rng.choice([int(rng.integers(1, 3000)), int(rng.integers(3000, 200_000))]))
        chans = sorted(rng.choice(32, int(rng.integers(1, 5)), replace=False).tolist())
        src = {c: random_signal(rng, n) for c in chans}
        cap = Capture.from_arrays(int(rng.choice([1_000_000, 250_000_000, 3_000_000, 300_000_000])), src)
        back = Capture.load(cap.save(tmp_path / f"r{it}.dsl"))
        get_overview(back).build(chans, background=False)
        for ch in chans:
            x = src[ch]
            truth = np.flatnonzero(np.diff(x.astype(np.int8))) + 1
            idx = rng.integers(0, n, 300)
            assert np.array_equal(back[ch], x)
            assert np.array_equal(back.levels_at(ch, idx), x[idx])
            assert np.array_equal(back.edges(ch), truth)
        for ext in (".sr", ".vcd", ".csv"):
            got = PARSE[ext](cap.save(tmp_path / f"r{it}{ext}"), cap)
            for ch in chans:
                assert np.array_equal(got[ch], src[ch]), (ext, ch, cap.samplerate)


def test_vcd_timescale_is_legal_and_exact(tmp_path):
    for rate, scale in ((250_000_000, "1 ns"), (1_000_000_000, "1 ns"), (100_000_000, "10 ns"),
                        (1_000_000, "1 us"), (300_000_000, "1 fs")):
        x = np.zeros(10_000_000 // max(1, rate // 1_000_000) + 1000, np.uint8)
        x[-1] = 1                                     # one edge at the very end: no drift allowed
        cap = Capture.from_arrays(rate, {0: x})
        txt = cap.save(tmp_path / "t.vcd").read_text()
        assert f"$timescale {scale} $end" in txt
        got = PARSE[".vcd"](tmp_path / "t.vcd", cap)
        assert np.array_equal(got[0], x)
