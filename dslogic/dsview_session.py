"""Import DSView settings.

DSView stores its full device setup as JSON ("session"), both inside every saved .dsl
and in its "Save Settings" files (.dsc). `settings_from_dsview()` converts that JSON to
the dict used by DSLogic.apply_settings(), so a capture can be repeated with exactly the
settings of an earlier DSView capture.

Keys are DSView 1.3.x names (MainWindow::gen_config_json, TriggerDock::get_session).
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

from .export import STRIG, SR_CHANNEL_LOGIC

INV_STRIG = {v: k for k, v in STRIG.items()}
OP_MODES = {0: "buffer", 1: "stream"}
STOP_OPTIONS = {0: "stop", 1: "upload"}


def read_session(path) -> dict:
    """DSView session JSON from a .dsl (zip) or a DSView settings file."""
    path = Path(path)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            if "session" not in z.namelist():
                raise ValueError(f"{path} has no DSView 'session' entry (saved by an old DSView?)")
            return json.loads(z.read("session").decode("utf-8", "replace"))
    return json.loads(path.read_text(encoding="utf-8"))


def is_dsview_session(d: dict) -> bool:
    return isinstance(d, dict) and "Version" in d and "channel" in d


def _pattern(trig: dict, key: str, ext_key: str, wide: bool) -> str:
    low = trig.get(key, "X " * 15 + "X")
    if wide:
        return trig.get(ext_key, "X " * 15 + "X") + " " + low
    return low


def settings_from_dsview(sess: dict) -> tuple[dict, list[str]]:
    """Return (settings dict for DSLogic.apply_settings, list of warnings)."""
    warn = []
    if int(sess.get("DeviceMode", 0)) != 0:
        raise ValueError("only DSView logic-analyser sessions can be imported")
    s: dict = {}
    op = int(sess.get("Operation Mode", 0))
    if op not in OP_MODES:
        warn.append(f"operation mode {op} (test mode) not supported; using buffer mode")
    s["operation_mode"] = OP_MODES.get(op, "buffer")
    if "Channel Mode" in sess:
        s["channel_mode"] = int(sess["Channel Mode"])
    if "Sample rate" in sess:
        s["samplerate"] = int(sess["Sample rate"])
    if "Sample count" in sess:
        s["sample_count"] = int(sess["Sample count"])
    if "Threshold Level" in sess:
        s["threshold"] = float(sess["Threshold Level"])
    s["filter"] = int(sess.get("Filter Targets", 0)) == 1
    s["external_clock"] = bool(sess.get("Using External Clock", 0))
    s["clock_negedge"] = bool(sess.get("Using Clock Negedge", 0))
    s["stop_option"] = STOP_OPTIONS.get(int(sess.get("Stop Options", 1)), "upload")
    if sess.get("Enable RLE Compress"):
        warn.append("RLE compression is not supported; ignored")

    chans = []
    for c in sess.get("channel", []):
        if int(c.get("type", SR_CHANNEL_LOGIC)) != SR_CHANNEL_LOGIC:
            continue
        col = c.get("colour")
        chans.append({"index": int(c["index"]), "name": str(c.get("name") or c["index"]),
                      "color": col if isinstance(col, str) and col.startswith("#") and len(col) == 7 else None,
                      "enabled": bool(c.get("enabled", True)),
                      "trigger": INV_STRIG.get(int(c.get("strigger", 0)), "X")})
    s["channels"] = chans
    s["channels_exclusive"] = True              # channels missing from the list are disabled

    trig = sess.get("trigger", {})
    s["trigger_position"] = int(trig.get("triggerPos", 0))
    if trig.get("advTriggerMode"):
        wide = any(ch["index"] >= 16 for ch in chans) or "stageTriggerExt32Value00" in trig
        if int(trig.get("triggerTab", 0)) == 0:
            stages = []
            for i in range(int(trig.get("triggerStages", 0)) + 1):
                stages.append({
                    "cond0": _pattern(trig, f"stageTriggerValue0{i}", f"stageTriggerExt32Value0{i}", wide),
                    "cond1": _pattern(trig, f"stageTriggerValue1{i}", f"stageTriggerExt32Value1{i}", wide),
                    "op": "and" if int(trig.get(f"stageTriggerLogic{i}", 1)) == 1 else "or",
                    "inv0": bool(trig.get(f"stageTriggerInv0{i}", 0)),
                    "inv1": bool(trig.get(f"stageTriggerInv1{i}", 0)),
                    "count": max(1, int(trig.get(f"stageTriggerCount{i}", 1))),
                    "contiguous": bool(trig.get(f"stageTriggerContiguous{i}", False)),
                })
            s["advanced_trigger"] = stages
        else:
            data = str(trig.get("serialTriggerData", "")).replace(" ", "")
            s["serial_trigger"] = {
                "start": _pattern(trig, "serialTriggerStart", "serialTriggerExt32Start", wide),
                "stop": _pattern(trig, "serialTriggerStop", "serialTriggerExt32Stop", wide),
                "clock": _pattern(trig, "serialTriggerClock", "serialTriggerExt32Clock", wide),
                "data_channel": int(trig.get("serialTriggerChannel", 0)),
                "value": data,
                "bits": int(trig.get("serialTriggerBits", 0)) + 1,
            }
            warn.append("serial trigger imported (experimental)")
    s["decoders"] = sess.get("decoder", [])
    return s, warn
