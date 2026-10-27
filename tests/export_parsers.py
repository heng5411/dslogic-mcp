"""Independent readers for the export formats (for checking the writers)."""
import re
import zipfile
from fractions import Fraction

import numpy as np


def parse_sr(path, cap):
    z = zipfile.ZipFile(path)
    meta = z.read("metadata").decode()
    unit = int(re.search(r"unitsize=(\d+)", meta).group(1))
    parts = sorted((n for n in z.namelist() if n.startswith("logic-1-")), key=lambda n: int(n.rsplit("-", 1)[1]))
    raw = np.frombuffer(b"".join(z.read(n) for n in parts), np.uint8).reshape(-1, unit)
    return {ch: (raw[:, k // 8] >> (k % 8)) & 1 for k, ch in enumerate(cap.enabled)}


def parse_vcd(path, cap):
    txt = open(path).read()
    m = re.search(r"\$timescale\s+(\d+)\s*(\w+)\s+\$end", txt)
    assert m.group(1) in ("1", "10", "100"), f"illegal VCD timescale {m.group(1)} {m.group(2)}"
    mult = {"s": 10**15, "ms": 10**12, "us": 10**9, "ns": 10**6, "ps": 10**3, "fs": 1}[m.group(2)]
    unit_fs = int(m.group(1)) * mult
    ids = re.findall(r"\$var wire 1 (\S+) (\S+) \$end", txt)
    order = [i for i, _ in ids]
    n = cap.num_samples
    period_fs = Fraction(10**15, cap.samplerate)
    out = {ch: np.zeros(n, np.uint8) for ch in cap.enabled}
    cur_t = 0
    body = txt.split("$enddefinitions $end", 1)[1]
    for tok in body.split():
        if tok.startswith("#"):
            cur_t = int(tok[1:])
        elif tok[0] in "01" and tok[1:] in order:
            s = Fraction(cur_t * unit_fs) / period_fs
            assert s.denominator == 1 or abs(float(s) - round(s)) < 0.01, f"time {cur_t} is between samples"
            ch = cap.enabled[order.index(tok[1:])]
            out[ch][round(s):] = int(tok[0])
    return out


def parse_csv(path, cap):
    rows = open(path).read().strip().split("\n")[1:]
    n = cap.num_samples
    out = {ch: np.zeros(n, np.uint8) for ch in cap.enabled}
    for r in rows:
        f = r.split(",")
        s = int(f[0])
        for k, ch in enumerate(cap.enabled):
            out[ch][s:] = int(f[2 + k])
    return out


PARSE = {".sr": parse_sr, ".vcd": parse_vcd, ".csv": parse_csv}
