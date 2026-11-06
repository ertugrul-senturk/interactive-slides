"""Fills `const PLAQUE_V2 = ...;` in the Smart Teeth deck from the plaque-v2 results on this PC.

  python scripts/fill-plaque-v2.py

Reads SmartTeeth-AI/1_plaque_detection/outputs_v2_fetched/<model>/summary.json (5-fold held-out metrics, photo-level split)
and reports/conversion/<model>.json (verified TFLite file: size, kind, CPU ms per tooth). Models without results are listed as
still running. Re-run whenever more models arrive.
"""
import json, re
from pathlib import Path

DECK = Path(__file__).resolve().parent.parent / "public" / "decks" / "2026-11-smart-teeth.html"
SRC = Path(r"C:\Users\ASUS\Desktop\SmartTeeth-AI\1_plaque_detection")
MODELS = [  # key, encoder label, year, crops
    ("b4_256", "EfficientNet-B4 (current model)", "2019", "low-res"),
    ("mnv4l_384", "MobileNetV4-L", "2024", "low-res"),
    ("b4_hr512", "EfficientNet-B4", "2019", "high-res"),
    ("fastvit_hr512", "FastViT", "2023", "high-res"),
    ("mnv4l_hr512", "MobileNetV4-L", "2024", "high-res"),
    ("repvit_hr512", "RepViT", "2024", "high-res"),
    ("mci2_hr512", "MobileCLIP", "2024", "high-res"),
    ("hgnetv2_hr512", "HGNetV2", "2024", "high-res"),
    ("effvit_hr512", "EfficientViT-B2", "2023", "high-res"),
    ("effvitb3_hr512", "EfficientViT-B3", "2023", "high-res"),
    ("effsam_hr512", "EfficientSAM", "2024", "high-res"),
]


def load(p):
    try: return json.load(open(p))
    except Exception: return None


rows = []
for key, enc, year, crops in MODELS:
    s = load(SRC / "outputs_v2_fetched" / key / "summary.json")
    c = load(SRC / "reports" / "conversion" / f"{key}.json") or {}
    # a model is shown only once it has finished training and its TFLite conversion has been attempted
    r = dict(key=key, enc=enc, year=year, crops=crops, done=bool(s) and bool(c))
    if s:
        r.update(tooth=round(s["tooth_iou"]["mean"], 3), photo=round(s["photo_iou"]["mean"], 3), pooled=round(s["pooled_iou"]["mean"], 3),
                 f2=round(s["tooth_f2"]["mean"], 3), f2pooled=round(s["pooled_f2"]["mean"], 3), params=round(s["config"].get("params_m") or 0, 1))
    if c.get("verified"):
        r.update(mb=c.get("tflite_mb"), kind=c.get("tflite_kind"), ms=c.get("tflite_cpu_ms_per_tooth_4threads"))
    rows.append(r)

html = DECK.read_text(encoding="utf8")
line = "const PLAQUE_V2 = " + json.dumps(rows, separators=(",", ":")) + ";"
new, n = re.subn(r"const PLAQUE_V2 = .*?;\n", line + "\n", html, count=1)
assert n == 1, "marker line not found"
DECK.write_text(new, encoding="utf8")
print(f"filled {sum(r['done'] for r in rows)} of {len(rows)} models")
