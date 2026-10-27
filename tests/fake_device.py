"""A simulated DSLogic U3Pro32 that speaks the transport API of dslogic.usbio.

It checks the command order DSView's driver uses and answers like the hardware:
the trigger header (1024 bytes) followed by the channel data in 64-bit words per
enabled channel.  `waveform(n)` supplies {channel: 0/1 array} for the simulated inputs.
"""
from __future__ import annotations

import struct

import numpy as np

from dslogic import constants as C


class FakeRead:
    def __init__(self, size, cb):
        self.size, self.cb = size, cb
        self.done = False
        self.cancelled = False

    def cancel(self):
        if not self.done:
            self.cancelled = True


class FakeTransport:
    def __init__(self, waveform, fpga_done=False, trigger_at=0, remain=0, overflow=False,
                 never_trigger=False):
        self.waveform = waveform
        self.fpga_done = fpga_done
        self.trigger_at = trigger_at
        self.remain = remain
        self.overflow = overflow
        self.never_trigger = never_trigger
        self.log: list = []
        self.regs: dict[int, int] = {}
        self.bulk: list[bytes] = []
        self.queue: list[FakeRead] = []
        self.stream = b""
        self.pos = 0
        self.claimed = False
        self.closed = False
        self.fpga_bytes = 0
        self.armed = None

    # --- transport API ------------------------------------------------------
    def strings(self):
        return "DreamSourceLab", "USB-based DSL Instrument v2"

    def claim(self):
        self.claimed = True

    def close(self):
        self.closed = True

    def ctl_write(self, dest, offset=0, data=b""):
        self.log.append(("w", dest, offset, bytes(data)))
        if dest == C.DSL_CTL_I2C_REG:
            self.regs[offset] = data[0]
            if offset == C.CTR0_ADDR and data[0] == C.bmFORCE_STOP and self.never_trigger:
                self.never_trigger = False
                self.remain = -1          # upload what we have: build stream now
                self._make_stream(forced=True)
        elif dest == C.DSL_CTL_START:
            assert self.armed is not None, "START before arm"
            if not self.never_trigger:
                self._make_stream()

    def ctl_read(self, dest, offset=0, size=1):
        self.log.append(("r", dest, offset, size))
        if dest == C.DSL_CTL_FW_VERSION:
            return bytes([2, 0])
        if dest == C.DSL_CTL_HW_STATUS:
            v = C.bmFPGA_INIT_B | C.bmSYS_CLR | C.bmGPIF_DONE
            if self.fpga_done:
                v |= C.bmFPGA_DONE
            if self.overflow and self.stream:
                v |= C.bmSYS_OVERFLOW
            return bytes([v])
        if dest == C.DSL_CTL_I2C_STATUS:
            if offset == 0 and size == C.HDL_VERSION_ADDR + 1:
                return bytes([0, 0, 0, 0, C.DSL_HDL_VERSION])
            if offset == 0 and size == 4:
                return bytes([0, 0, 0, 0])
            return bytes(size)
        return bytes(size)

    def bulk_write(self, data, timeout_ms=1000):
        data = bytes(data)
        self.bulk.append(data)
        last_bulk_cmd = [e for e in self.log if e[0] == "w" and e[1] == C.DSL_CTL_BULK_WR][-1]
        n = int.from_bytes(last_bulk_cmd[3], "little")
        if len(data) == n and n > 10000:          # FPGA bitstream
            self.fpga_bytes = n
            self.fpga_done = True
        elif len(data) == 372:
            self.armed = data
        return len(data)

    def submit_read(self, size, cb, timeout_ms=0):
        r = FakeRead(size, cb)
        self.queue.append(r)
        return r

    def handle_events(self, timeout_s):
        if self.overflow:
            import time
            time.sleep(0.05)
        if not self.queue:
            return
        r = self.queue.pop(0)
        if r.cancelled:
            r.cb(memoryview(b""), "cancelled")
            r.done = True
            return
        if self.pos >= len(self.stream):
            self.queue.insert(0, r)            # nothing to send yet (waiting for trigger)
            return
        chunk = self.stream[self.pos:self.pos + r.size]
        self.pos += len(chunk)
        if r.cb(memoryview(chunk), "ok"):
            self.queue.append(r)
        else:
            r.done = True

    # --- simulation -----------------------------------------------------------
    def decoded_setting(self):
        s = self.armed
        f = {}
        f["mode"] = struct.unpack_from("<H", s, 6)[0]
        f["cnt"] = struct.unpack_from("<H", s, 16)[0] | (struct.unpack_from("<H", s, 18)[0] << 16)
        f["ch_en"] = struct.unpack_from("<H", s, 38)[0] | (struct.unpack_from("<H", s, 40)[0] << 16)
        return f

    def _make_stream(self, forced=False):
        f = self.decoded_setting()
        actual_samples = f["cnt"] << 4
        en = [i for i in range(32) if f["ch_en"] >> i & 1]
        remain = actual_samples // 2 if forced else self.remain
        samples = actual_samples - max(0, remain)
        wave = self.waveform(actual_samples)
        cols = []
        for ch in en:
            bits = np.asarray(wave.get(ch, np.zeros(actual_samples)), dtype=np.uint8)[:actual_samples]
            cols.append(np.packbits(bits, bitorder="little").view("<u8"))
        words = np.stack(cols, axis=1).reshape(-1)
        header = struct.pack("<6I", C.TRIG_CHECKID, self.trigger_at, 0,
                             max(0, remain) & 0xFFFFFFFF, max(0, remain) >> 32,
                             1 if f["mode"] & 1 else 0).ljust(1024, b"\0")
        nbytes = samples // 64 * len(en) * 8
        self.stream = header + words.tobytes()[:nbytes] + b"\0" * 4096   # device keeps streaming
