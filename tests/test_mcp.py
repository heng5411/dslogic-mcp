import json

import anyio
import numpy as np
import pytest

pytest.importorskip("mcp")

from mcp import Client  # noqa: E402

from dslogic import Capture, DSLogic  # noqa: E402
from dslogic import constants as C  # noqa: E402
from dslogic import mcp_server as M  # noqa: E402

from fake_device import FakeTransport  # noqa: E402
from signals import spi  # noqa: E402


def spi_cap():
    cs, clk, mosi, miso = spi([0x9F, 0x12], [0x00, 0xC2], spb=8)
    pad = np.ones(4_000, np.uint8)
    z = np.zeros(4_000, np.uint8)
    cap = Capture.from_arrays(100_000_000, {0: np.concatenate([pad, cs, pad]),
                                            1: np.concatenate([z, clk, z]),
                                            2: np.concatenate([z, miso, z]),
                                            3: np.concatenate([z, mosi, z])})
    for i, n in enumerate(["CS", "SCLK", "MISO", "MOSI"]):
        cap.set_name(i, n)
    return cap


def call(name, args=None):
    """Call one tool through a real MCP client session (in-process)."""
    async def go():
        async with Client(M.server) as c:
            return await c.call_tool(name, args or {})
    return anyio.run(go)


def result(r):
    assert not r.is_error, r.content
    return r.structured_content.get("result", r.structured_content) \
        if r.structured_content is not None else json.loads(r.content[0].text)


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(M, "_last", None)
    monkeypatch.setattr(M, "_cache", type(M._cache)())
    monkeypatch.setattr(M, "_open_device", None)


@pytest.fixture
def spi_file(tmp_path):
    cap = spi_cap()
    cap.add_decoder(M._decoder("spi:clk=SCLK,mosi=MOSI,miso=MISO,cs=CS"))
    return str(cap.save(tmp_path / "spi.dsl"))


def test_lists_all_tools():
    async def go():
        async with Client(M.server) as c:
            return {t.name for t in (await c.list_tools()).tools}
    assert anyio.run(go) == {"scan", "info", "capture", "stop_capture", "load", "decode",
                             "measure", "pulse", "edges", "render", "convert", "view"}


def test_load_and_decode_saved_decoder(spi_file):
    info = result(call("load", {"file": spi_file}))
    assert info["samples"] == spi_cap().num_samples and info["channels"]["0"] == "CS"
    assert "SPI" in info["decoders"][0]
    out = result(call("decode", {"kind": "transfer"}))     # no file: the loaded one
    assert out[0]["count"] == 1
    assert out[0]["annotations"][0]["data"] == {"mosi": [0x9F, 0x12], "miso": [0x00, 0xC2],
                                                "bits": 16}


def test_decode_with_other_decoder_window_and_limit(spi_file):
    out = result(call("decode", {"file": spi_file, "kind": "word", "max_lines": 1,
                                 "decoders": ["spi:clk=1,mosi=3,miso=2,cs=0,label=X"]}))
    assert out[0]["decoder"] == "X" and out[0]["count"] == 2 and out[0]["shown"] == 1
    assert out[0]["annotations"][0]["data"] == {"mosi": 0x9F, "miso": 0x00}
    empty = result(call("decode", {"file": spi_file, "end": "10us", "kind": "word"}))
    assert empty[0]["count"] == 0          # the transfer starts after 40 us


def test_errors_reach_the_model(spi_file, tmp_path):
    r = call("measure", {"channel": "SCLK"})
    assert r.is_error and "no capture yet" in r.content[0].text
    r = call("decode", {"file": spi_file, "decoders": ["can:rx=1"]})
    assert r.is_error and "unknown decoder" in r.content[0].text
    r = call("load", {"file": str(tmp_path / "missing.dsl")})
    assert r.is_error and "missing.dsl" in r.content[0].text


def test_measure_pulse_edges(spi_file):
    m = result(call("measure", {"file": spi_file, "channel": "SCLK"}))
    assert m["rising"] == 16 and m["frequency"] == pytest.approx(100e6 / 8)
    p = result(call("pulse", {"channel": "CS", "at": "41us"}))
    assert p["level"] == 0 and p["width"] == pytest.approx((16 * 8 + 8) / 100e6)
    e = result(call("edges", {"channel": "0", "kind": "falling"}))
    assert e["count"] == 1 and e["times_s"][0] == pytest.approx((4000 + 36) / 100e6)


def test_render_returns_png(spi_file, tmp_path):
    pytest.importorskip("matplotlib")
    r = call("render", {"file": spi_file, "start": "39us", "end": "43us", "channels": "CS,SCLK",
                        "png": str(tmp_path / "w.png")})
    assert not r.is_error and r.content[0].type == "image"
    assert (tmp_path / "w.png").read_bytes()[:4] == b"\x89PNG"


def test_convert_crop_and_rename_leaves_cache_alone(spi_file, tmp_path):
    msg = result(call("convert", {"file": spi_file, "output": str(tmp_path / "x.vcd"),
                                  "start": "40us", "end": "42us", "channels": "0-1"}))
    assert "saved" in msg and (tmp_path / "x.vcd").exists()
    call("convert", {"output": str(tmp_path / "r.dsl"), "names": {"0": "NCS"}})
    assert Capture.load(tmp_path / "r.dsl").channels[0].name == "NCS"
    assert result(call("load", {"file": spi_file}))["channels"]["0"] == "CS"


def test_cache_reloads_a_changed_file(spi_file):
    a = M._get(spi_file)
    assert M._get(spi_file) is a
    cap = spi_cap()
    cap.set_name(0, "CHANGED")
    cap.save(spi_file)
    assert M._get(spi_file) is not a and M._get(spi_file).channels[0].name == "CHANGED"


def test_host_path_conversion(monkeypatch):
    monkeypatch.setattr(M.os, "name", "nt")
    assert M.host_path("/mnt/d/captures/x.dsl") == r"D:\captures\x.dsl"
    assert M.host_path(r"C:\a\b.dsl") == r"C:\a\b.dsl"
    assert M._shown(r"D:\captures\x.dsl").endswith("(WSL: /mnt/d/captures/x.dsl)")


def test_capture_through_fake_analyser(monkeypatch, tmp_path):
    ref = spi_cap()

    def wave(n):                # the SPI frame repeated over the whole capture
        return {i: np.resize(ref.samples(i), n) for i in range(4)}

    def opener(index):
        la = DSLogic(FakeTransport(wave, fpga_done=True), C.find_profile(0x002C, True))
        la._bring_up()
        return la

    monkeypatch.setattr(M, "_open_device", opener)
    out = result(call("capture", {
        "rate": "100M", "samples": "8k", "channels": "0-3", "threshold": 1.65,
        "names": {"0": "CS", "1": "SCLK", "2": "MISO", "3": "MOSI"},
        "decoders": ["spi:clk=SCLK,mosi=MOSI,miso=MISO,cs=CS"], "timeout": 5,
        "output": [str(tmp_path / "c.dsl")], "max_lines": 3}))
    assert out["summary"].startswith("8000 samples @ 100 MHz")
    assert out["saved"][0].startswith(str(tmp_path / "c.dsl"))
    words = [a["data"] for a in out["decode"][0]["annotations"] if a["kind"] == "word"]
    assert words[0] == {"mosi": 0x9F, "miso": 0x00}
    # the capture stays loaded for the analysis tools
    assert result(call("measure", {"channel": "SCLK"}))["rising"] == 16


def test_capture_rejects_bad_settings_before_opening(monkeypatch):
    monkeypatch.setattr(M, "_open_device", lambda i: pytest.fail("analyser opened"))
    r = call("capture", {"output": ["x.ds"]})
    assert r.is_error and "unknown file type" in r.content[0].text
    r = call("capture", {"decoders": ["spi:mosi=3"]})
    assert r.is_error and "missing" in r.content[0].text
    assert result(call("stop_capture")) == "no capture running"
