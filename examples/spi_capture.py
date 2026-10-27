"""Capture one SPI burst with a CS trigger, decode it and save a DSView file."""
from dslogic import SPI, DSLogic

with DSLogic.open() as la:
    la.operation_mode = "buffer"          # hold the capture in the analyser's 2 Gbit memory
    la.samplerate = "250M"
    la.duration = "5ms"
    la.threshold = 1.65                   # 3.3 V logic (use 0.9 for 1.8 V)
    la.enable_channels([0, 1, 2, 3])
    la.set_names({0: "CS", 1: "SCLK", 2: "MISO", 3: "MOSI"})
    la.set_colors({"CS": "yellow", "SCLK": "#00c0ff"})
    la.set_trigger("CS", "falling")
    la.trigger_position = 10              # 10 % of the capture before the trigger
    la.add_decoder(SPI(clk="SCLK", mosi="MOSI", miso="MISO", cs="CS", cpol=0, cpha=0))
    print(la.describe())
    cap = la.capture(timeout=10)

print(cap.summary())
for t in cap.decoders[0].transfers(cap):
    print(f"{cap.time_of(t.start) * 1e6:10.2f} us  {t.text}")
cap.save("spi.dsl")                       # open in DSView: names, colours and decoder included
cap.save("spi.vcd")
