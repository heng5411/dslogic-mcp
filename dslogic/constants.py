"""Protocol constants and hardware profiles.

Ported from DreamSourceLab's libsigrok4DSL (DSView, hardware/DSL/{command.h,dsl.h,
dslogic.c} at commit 2e9e2c8e, Nov 2024).  Names follow the C source so the two can
be compared line by line.
"""
from dataclasses import dataclass

DS_VENDOR_ID = 0x2A0E

# USB
USB_INTERFACE = 0
EP_BULK_OUT = 0x02          # 2 | LIBUSB_ENDPOINT_OUT
EP_BULK_IN = 0x86           # 6 | LIBUSB_ENDPOINT_IN
CTL_TIMEOUT_MS = 3000

# Vendor control requests
CMD_CTL_WR = 0xB0
CMD_CTL_RD_PRE = 0xB1
CMD_CTL_RD = 0xB2

# Control "dest" codes (enum in command.h)
DSL_CTL_FW_VERSION = 0
DSL_CTL_REVID_VERSION = 1
DSL_CTL_HW_STATUS = 2
DSL_CTL_PROG_B = 3
DSL_CTL_SYS = 4
DSL_CTL_LED = 5
DSL_CTL_INTRDY = 6
DSL_CTL_WORDWIDE = 7
DSL_CTL_START = 8
DSL_CTL_STOP = 9
DSL_CTL_BULK_WR = 10
DSL_CTL_REG = 11
DSL_CTL_NVM = 12
DSL_CTL_I2C_DSO = 13
DSL_CTL_I2C_REG = 14
DSL_CTL_I2C_STATUS = 15

# HW_STATUS bits
bmGPIF_DONE = 1 << 7
bmFPGA_DONE = 1 << 6
bmFPGA_INIT_B = 1 << 5
bmSYS_OVERFLOW = 1 << 4
bmSYS_CLR = 1 << 3
bmLED_RED = 1 << 1
bmLED_GREEN = 1 << 0
bmWR_PROG_B = 1 << 2
bmWR_INTRDY = 1 << 7
bmWR_WORDWIDE = 1 << 0

# FPGA registers (via DSL_CTL_I2C_REG / DSL_CTL_I2C_STATUS)
VTH_ADDR = 0x78
CTR0_ADDR = 0x70
ADCC_ADDR = 0x48
HW_STATUS_ADDR = 0x05
HDL_VERSION_ADDR = 0x04
SECU_EEP_ADDR = 0x3C00
SECU_STEPS = 8

# CTR0 bits
bmNONE = 0
bmFORCE_RDY = 1 << 1
bmFORCE_STOP = 1 << 2

DSL_REQUIRED_VERSION_MAJOR = 2
DSL_HDL_VERSION = 0x0E

# Mode word bits (DSL_setting.mode)
TRIG_EN_BIT = 0
CLK_TYPE_BIT = 1
CLK_EDGE_BIT = 2
RLE_MODE_BIT = 3
DSO_MODE_BIT = 4
HALF_MODE_BIT = 5
QUAR_MODE_BIT = 6
ANALOG_MODE_BIT = 7
FILTER_BIT = 8
INSTANT_BIT = 9
SLOW_ACQ_BIT = 10
STRIG_MODE_BIT = 11
STREAM_MODE_BIT = 12
LPB_TEST_BIT = 13
EXT_TEST_BIT = 14
INT_TEST_BIT = 15

NUM_TRIGGER_STAGES = 16
TRIGGER_PROBES = 16           # TriggerProbes: one 16-bit word per half of the 32 channels
S_TRIGGER_DATA_STAGE = 3      # STriggerDataStage
DS_MAX_TRIG_PERCENT = 90
SAMPLES_ALIGN = 1023
ATOMIC_SAMPLES = 64           # DSLOGIC_ATOMIC_SAMPLES: the device moves 64-sample words
ATOMIC_SIZE = 8               # bytes per 64-sample word
ATOMIC_MASK = (0xFFFF << 6) & 0xFFFF
TRIG_CHECKID = 0x55555555

SIMPLE_TRIGGER = 0
ADV_TRIGGER = 1
SERIAL_TRIGGER = 2

# ADF4360 sample clock bring-up done by dev_open() (adc_clk_init_500m): (reg, bytes, delay_ms)
ADC_CLK_INIT_500M = (
    (ADCC_ADDR + 2, (0x01,), 0),
    (ADCC_ADDR, (0x01, 0x61, 0x00, 0x30), 0),
    (ADCC_ADDR, (0x01, 0x40, 0xF1, 0x46), 0),
    (ADCC_ADDR, (0x01, 0x62, 0x3D, 0x40), 10),
)

# Feature caps used by this port
CAPS_FEATURE_VTH = 1 << 0
CAPS_FEATURE_BUF = 1 << 1
CAPS_FEATURE_USB30 = 1 << 6
CAPS_FEATURE_ADF4360 = 1 << 8
CAPS_FEATURE_LA_CH32 = 1 << 11
CAPS_FEATURE_MAX25_VTH = 1 << 13


def MHZ(n): return int(n * 1_000_000)
def KHZ(n): return int(n * 1_000)


SAMPLERATES_1000 = (
    10, 20, 50, 100, 200, 500,
    KHZ(1), KHZ(2), KHZ(5), KHZ(10), KHZ(20), KHZ(40), KHZ(50), KHZ(100), KHZ(200),
    KHZ(400), KHZ(500),
    MHZ(1), MHZ(2), MHZ(4), MHZ(5), MHZ(10), MHZ(20), MHZ(25), MHZ(50), MHZ(100),
    MHZ(125), MHZ(250), MHZ(500), MHZ(1000),
)


@dataclass(frozen=True)
class ChannelMode:
    """One entry of channel_modes[] (logic modes only)."""
    id: int
    stream: bool
    num: int            # channels the mode exposes (0..num-1)
    vld_num: int        # how many of them may be enabled at once
    min_samplerate: int
    max_samplerate: int
    hw_min_samplerate: int
    hw_max_samplerate: int
    pre_div: int
    descr: str


CHANNEL_MODES = {m.id: m for m in (
    ChannelMode(8,  True, 32, 32, KHZ(100), MHZ(10),  KHZ(10), MHZ(500), 5, "Use 32 Channels (Max 10MHz)"),
    ChannelMode(9,  True, 32, 16, KHZ(100), MHZ(20),  KHZ(10), MHZ(500), 5, "Use 16 Channels (Max 20MHz)"),
    ChannelMode(10, True, 32, 12, KHZ(100), MHZ(25),  KHZ(10), MHZ(500), 5, "Use 12 Channels (Max 25MHz)"),
    ChannelMode(11, True, 32, 6,  KHZ(100), MHZ(50),  KHZ(10), MHZ(500), 5, "Use 6 Channels (Max 50MHz)"),
    ChannelMode(12, True, 32, 3,  KHZ(100), MHZ(100), KHZ(10), MHZ(500), 5, "Use 3 Channels (Max 100MHz)"),
    ChannelMode(4,  True, 16, 16, KHZ(100), MHZ(20),  KHZ(10), MHZ(500), 5, "Use 16 Channels (Max 20MHz)"),
    ChannelMode(5,  True, 16, 12, KHZ(100), MHZ(25),  KHZ(10), MHZ(500), 5, "Use 12 Channels (Max 25MHz)"),
    ChannelMode(6,  True, 16, 6,  KHZ(100), MHZ(50),  KHZ(10), MHZ(500), 5, "Use 6 Channels (Max 50MHz)"),
    ChannelMode(7,  True, 16, 3,  KHZ(100), MHZ(100), KHZ(10), MHZ(500), 5, "Use 3 Channels (Max 100MHz)"),
    ChannelMode(13, True, 32, 32, MHZ(1), MHZ(50),   KHZ(10), MHZ(500), 5, "Use 32 Channels (Max 50MHz)"),
    ChannelMode(14, True, 32, 30, MHZ(1), MHZ(100),  KHZ(10), MHZ(500), 5, "Use 30 Channels (Max 100MHz)"),
    ChannelMode(15, True, 32, 12, MHZ(1), MHZ(250),  KHZ(10), MHZ(500), 5, "Use 12 Channels (Max 250MHz)"),
    ChannelMode(16, True, 16, 16, MHZ(1), MHZ(125),  KHZ(10), MHZ(500), 5, "Use 16 Channels (Max 125MHz)"),
    ChannelMode(17, True, 16, 12, MHZ(1), MHZ(250),  KHZ(10), MHZ(500), 5, "Use 12 Channels (Max 250MHz)"),
    ChannelMode(18, True, 16, 6,  MHZ(1), MHZ(500),  KHZ(10), MHZ(500), 5, "Use 6 Channels (Max 500MHz)"),
    ChannelMode(19, True, 8,  3,  MHZ(1), MHZ(1000), KHZ(10), MHZ(500), 5, "Use 3 Channels (Max 1GHz)"),
    ChannelMode(23, False, 32, 32, MHZ(1), MHZ(250),  KHZ(10), MHZ(500), 5, "Use Channels 0~31 (Max 250MHz)"),
    ChannelMode(24, False, 16, 16, MHZ(1), MHZ(500),  KHZ(10), MHZ(500), 5, "Use Channels 0~15 (Max 500MHz)"),
    ChannelMode(25, False, 8,  8,  MHZ(1), MHZ(1000), KHZ(10), MHZ(500), 5, "Use Channels 0~7 (Max 1GHz)"),
)}


@dataclass(frozen=True)
class Profile:
    pid: int
    superspeed: bool
    model: str
    fpga_bitstream: str
    feature_caps: int
    channel_modes: tuple       # ids, in DSView's listing order
    total_ch_num: int
    hw_depth: int              # sample memory in bits (SR_GB(2))
    half_samplerate: int
    quarter_samplerate: int

    @property
    def has_ch32(self):
        return bool(self.feature_caps & CAPS_FEATURE_LA_CH32)


_U3 = CAPS_FEATURE_VTH | CAPS_FEATURE_BUF | CAPS_FEATURE_USB30 | CAPS_FEATURE_ADF4360
GB2 = 2 * 1024 ** 3

PROFILES = (
    Profile(0x002C, False, "DSLogic U3Pro32", "DSLogicU3Pro32.bin", _U3 | CAPS_FEATURE_LA_CH32,
            (8, 9, 10, 11, 12, 23, 24, 25), 32, GB2, MHZ(500), MHZ(1000)),
    Profile(0x002C, True, "DSLogic U3Pro32", "DSLogicU3Pro32.bin", _U3 | CAPS_FEATURE_LA_CH32,
            (13, 14, 15, 18, 19, 23, 24, 25), 32, GB2, MHZ(500), MHZ(1000)),
    Profile(0x002A, False, "DSLogic U3Pro16", "DSLogicU3Pro16.bin", _U3,
            (4, 5, 6, 7, 24, 25), 16, GB2, MHZ(500), MHZ(1000)),
    Profile(0x002A, True, "DSLogic U3Pro16", "DSLogicU3Pro16.bin", _U3,
            (16, 17, 18, 19, 24, 25), 16, GB2, MHZ(500), MHZ(1000)),
)


def find_profile(pid, superspeed):
    for p in PROFILES:
        if p.pid == pid and p.superspeed == superspeed:
            return p
    return None
