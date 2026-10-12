"""Copies the measured results into the LoRA deck, and the combined VPT + LoRA deck if it exists, as `const RESULTS`.

LoRA runs and the cost benchmark of all four methods come from weights/*.json and weights/bench.json here; the
linear-probe, VPT-deep and full fine-tuning runs are the same ones as in the VPT talk, read from ../vpt/weights. The deck draws its tables and charts from this
object, so every number on them comes from a training run or a benchmark, never typed by hand.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent
VPT_W = ROOT.parent / "vpt" / "weights"
DECKS = ROOT.parent.parent / "public" / "decks"
TARGETS = ("2026-10-lora-low-rank-adaptation.html", "2026-10-peft-vpt-lora.html")   # the second only once it exists
TASKS = ("dermamnist", "bloodmnist", "pneumoniamnist")
METHODS = {"linear": (VPT_W, "linear"), "vpt": (VPT_W, "vpt-deep-p50"), "lora": (ROOT / "weights", "lora-r8"), "full": (VPT_W, "full")}


def load(task, method, regime):
    d, tag = METHODS[method]
    p = d / f"{task}_{tag}_{regime}.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    st = p.with_suffix(".safetensors")
    out = {"test": m["test"], "trainable": m["trainable"], "trainable_pct": round(m["trainable_pct"], 3),
           "train_images": m["train_images"], "train_minutes": m["train_minutes"], "epochs": m["epochs"],
           "base_lr": m["base_lr"], "file_kb": round(st.stat().st_size / 1024, 1) if st.exists() else None}
    if "test_merged" in m:
        out["test_merged"] = m["test_merged"]
    return out


results = {"runs": {}}
for task in TASKS:
    for regime in ("full", "1k"):
        runs = {k: load(task, k, regime) for k in METHODS}
        if any(runs.values()):
            results["runs"][f"{task}_{regime}"] = {k: v for k, v in runs.items() if v}
bench = ROOT / "weights" / "bench.json"
if bench.exists():
    results["bench"] = json.loads(bench.read_text())

for name in TARGETS:
    deck = DECKS / name
    if not deck.exists():
        continue
    html = deck.read_text(encoding="utf8")
    # the line must start with "const RESULTS = " (the combined deck keeps the VPT part's own numbers under another name)
    html, n = re.subn(r"(?m)^const RESULTS = .*?;\n", lambda _: "const RESULTS = " + json.dumps(results) + ";\n", html, count=1)
    assert n == 1, f"const RESULTS not found in {name}"
    deck.write_text(html, encoding="utf8")
    print("wrote", name)
print(json.dumps({k: {m: (v["test"].get("bal_acc"), v["file_kb"]) for m, v in r.items()} for k, r in results["runs"].items()}, indent=1))
