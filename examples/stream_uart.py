"""Stream 2 s of a UART line at 10 MHz and print the decoded text."""
from dslogic import UART, DSLogic

with DSLogic.open() as la:
    la.operation_mode = "stream"          # data goes straight to the PC; length limited by RAM
    la.samplerate = "10M"
    la.duration = "2s"
    la.enable_channels([6])
    la.set_name(6, "DUT_TX")
    la.clear_trigger()                    # start immediately
    uart = UART(rx="DUT_TX", baudrate=115200)
    la.add_decoder(uart)
    cap = la.capture(progress=lambda trig, f: print(f"\r{f * 100:5.1f} %", end=""))

print()
print(uart.text(cap))
cap.save("uart.dsl")
