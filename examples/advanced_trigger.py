"""Two-stage trigger: CS falls, then MOSI is high on the 3rd rising SCLK edge in a row."""
from dslogic import DSLogic, Stage

with DSLogic.open() as la:
    la.samplerate = "100M"
    la.sample_count = "4M"
    la.enable_channels(range(4))
    la.set_names(["CS", "SCLK", "MISO", "MOSI"])
    la.set_advanced_trigger([
        Stage(cond0={0: "falling"}),
        Stage(cond0={1: "rising", 3: "high"}, count=3, contiguous=True),
    ])
    la.trigger_position = 20
    cap = la.capture(timeout=30)
    print(cap.summary())
    cap.save("adv.dsl")
