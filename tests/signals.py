"""Synthetic bus waveforms for decoder tests (1 = high)."""
import numpy as np


def spi(data_mosi, data_miso, spb=8, idle=40, cpol=0, cpha=0, gap=0):
    """CS, SCLK, MOSI, MISO arrays for one CS-low transfer of the given bytes, MSB first.
    `spb` samples per bit. Data changes on the launch edge, sampled on the other."""
    n = idle * 2 + len(data_mosi) * 8 * spb + gap
    cs = np.ones(n, np.uint8)
    clk = np.full(n, cpol, np.uint8)
    mosi = np.zeros(n, np.uint8)
    miso = np.zeros(n, np.uint8)
    t0 = idle
    cs[t0 - 4: t0 + len(data_mosi) * 8 * spb + 4] = 0
    half = spb // 2
    for k, (bm, bs) in enumerate(zip(data_mosi, data_miso)):
        for b in range(8):
            i = t0 + (k * 8 + b) * spb
            vm, vs = (bm >> (7 - b)) & 1, (bs >> (7 - b)) & 1
            if cpha == 0:
                mosi[i - 2: i + spb - 2] = vm      # valid before the first (sampling) edge
                miso[i - 2: i + spb - 2] = vs
                clk[i: i + half] = 1 - cpol
            else:
                mosi[i: i + spb] = vm
                miso[i: i + spb] = vs
                clk[i: i + half] = 1 - cpol
    return cs, clk, mosi, miso


def i2c(addr, rw, payload, spb=10, idle=30):
    """SCL, SDA for START, address+R/W, data bytes (ACKed), STOP."""
    bits = []
    for byte in [(addr << 1) | rw] + list(payload):
        bits += [(byte >> (7 - i)) & 1 for i in range(8)] + [0]   # ACK = 0
    n = idle * 2 + (len(bits) + 2) * spb
    scl = np.ones(n, np.uint8)
    sda = np.ones(n, np.uint8)
    t = idle
    sda[t:] = 0                       # START: SDA falls while SCL high
    t += spb // 2
    scl[t:] = 0
    for b in bits:
        sda[t + 1: t + spb] = b      # change while SCL low
        scl[t + spb // 2: t + spb] = 1
        scl[t + spb:] = 0
        t += spb
    sda[t + 1:] = 0
    scl[t + spb // 2:] = 1
    sda[t + spb:] = 1                 # STOP: SDA rises while SCL high
    return scl, sda


def uart(data, spb=16, idle=50, bits=8, parity=None, stop=1):
    out = [np.ones(idle, np.uint8)]
    for byte in data:
        frame = [0] + [(byte >> i) & 1 for i in range(bits)]
        if parity == "even":
            frame.append(sum(frame[1:]) % 2)
        elif parity == "odd":
            frame.append(1 - sum(frame[1:]) % 2)
        frame += [1] * stop
        out.append(np.repeat(np.array(frame, np.uint8), spb))
        out.append(np.ones(spb, np.uint8))
    out.append(np.ones(idle, np.uint8))
    return np.concatenate(out)
