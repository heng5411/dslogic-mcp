"""Drive a DreamSourceLab DSLogic U3Pro32 / U3Pro16 logic analyser from Python.

    from dslogic import DSLogic, SPI
    with DSLogic.open() as la:
        la.operation_mode = "buffer"
        la.samplerate = "250M"
        la.duration = "20ms"
        la.threshold = 1.65
        la.enable_channels([0, 1, 2, 3])
        la.set_names(["CS", "SCLK", "MISO", "MOSI"])
        la.set_trigger("CS", "falling")
        la.trigger_position = 10
        la.add_decoder(SPI(clk="SCLK", mosi="MOSI", miso="MISO", cs="CS"))
        cap = la.capture()
    for t in cap.decode(cap.decoders[0]):
        print(t)
    cap.save("spi.dsl")       # opens in DSView with names, colours and the SPI decoder
"""
from .capture import Capture, Channel, format_rate, parse_count, parse_rate, parse_time
from .decoders import I2C, SPI, UART, Annotation, Decoder, DSViewDecoder
from .device import CaptureCancelled, CaptureError, DeviceInfo, DSLogic, OverflowError_
from .trigger import SerialTrigger, Stage, Trigger
from .usbio import DeviceNotFound, DSLogicError

__version__ = "0.1.0"
__author__ = "Wong Yiek Heng"
__email__ = "yiekheng@04080616.xyz"

__all__ = [
    "DSLogic", "DeviceInfo", "Capture", "Channel", "Trigger", "Stage", "SerialTrigger",
    "SPI", "I2C", "UART", "DSViewDecoder", "Decoder", "Annotation",
    "DSLogicError", "DeviceNotFound", "CaptureError", "CaptureCancelled", "OverflowError_",
    "format_rate", "parse_rate", "parse_count", "parse_time",
]
