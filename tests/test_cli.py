from dslogic import SPI, UART, DSViewDecoder
from dslogic.cli import _channels, main, parse_decoder


def test_channel_ranges():
    assert _channels("0-3,8,10-11") == [0, 1, 2, 3, 8, 10, 11]


def test_parse_decoders():
    d = parse_decoder("spi:clk=1,mosi=3,miso=2,cs=CS,cpol=1,cpha=1,label=Flash")
    assert isinstance(d, SPI) and d.channels == {"clk": 1, "miso": 2, "mosi": 3, "cs": "CS"}
    assert d.options["cpol"] == 1 and d.label == "Flash"
    u = parse_decoder("uart:rx=6,baud=9600,parity=even")
    assert isinstance(u, UART) and u.options["baudrate"] == 9600 and u.options["parity_type"] == "even"
    c = parse_decoder("dsview:can:can_rx=5;bitrate=500000,sample_point=75.0")
    assert isinstance(c, DSViewDecoder) and c.channels == {"can_rx": 5}
    assert c.options == {"bitrate": 500000, "sample_point": 75.0}


def test_convert_adds_names(tmp_path):
    import numpy as np
    from dslogic import Capture
    cap = Capture.from_arrays(1_000_000, {0: np.zeros(2048), 1: np.ones(2048)})
    src = cap.save(tmp_path / "a.dsl")
    assert main(["convert", str(src), str(tmp_path / "b.dsl"), "--name", "0=CLK,1=DAT",
                 "--color", "DAT=blue", "--decoder", "i2c:scl=0,sda=1"]) == 0
    b = Capture.load(tmp_path / "b.dsl")
    assert b.channels[1].name == "DAT" and b.channels[1].color == "#0000ff"
    assert b.decoders[0].id == "1:i2c"


def test_capture_rejects_bad_output_before_opening_device(capsys):
    # would need hardware if it got as far as DSLogic.open()
    assert main(["capture", "-o", "t1.ds"]) == 1
    assert "unknown file type '.ds'" in capsys.readouterr().err


def test_progress_bar_renders_phases():
    import io
    from dslogic.cli import ProgressBar

    class TTY(io.StringIO):
        encoding = "utf-8"
        def isatty(self):
            return True

    out = TTY()
    bar = ProgressBar(stream=out)
    bar.update("waiting", 0.3)
    bar.update("capturing", 0.5)
    bar.update("uploading", 0.25)
    bar.close()
    lines = out.getvalue().split("\n")
    finals = [l.split("\r")[-1] for l in lines if l]
    assert finals[0].startswith("waiting for trigger") and " 30.0 %" in finals[0]
    assert finals[1].startswith("capturing") and "100.0 %" in finals[1]
    assert finals[2].startswith("uploading") and " 25.0 %" in finals[2]
    assert "█" in finals[1]
    plain = io.StringIO()            # not a terminal: only one line per phase, ASCII-safe
    b2 = ProgressBar(stream=plain)
    b2.update("capturing", 0.5); b2.close()
    assert plain.getvalue().count("\n") == 1
