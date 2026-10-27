import subprocess
import shutil
import zipfile

import pytest

from dslogic.fastzip import write_zip


@pytest.mark.parametrize("limit", [0xFFFFFFFF, 100])   # 100: force the zip64 records
def test_fastzip_readable(tmp_path, limit):
    items = {f"L-{c}/{b}": bytes([c, b]) * (1000 + 7 * b) for c in range(3) for b in range(5)}
    items["header"] = b"[version]\nversion = 3\n"
    p = tmp_path / "x.zip"
    seen = []
    write_zip(p, [(k, (lambda v=v: v)) for k, v in items.items()], workers=4,
              progress=seen.append, _zip64_limit=limit)
    assert seen[-1] == len(items)
    with zipfile.ZipFile(p) as z:
        assert z.namelist() == list(items)
        assert z.testzip() is None
        assert all(z.read(k) == v for k, v in items.items())
    if shutil.which("unzip"):
        r = subprocess.run(["unzip", "-tq", str(p)], capture_output=True, text=True)
        assert "No errors" in r.stdout, r.stdout + r.stderr
