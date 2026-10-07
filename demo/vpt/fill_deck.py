"""Copies the measured results (weights/*.json, weights/bench.json) into the deck as `const RESULTS`.

The deck's cost and results slides draw their tables and charts from this object, so every
number on them comes from a training run or a benchmark, never typed by hand.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent
W = ROOT / "weights"
DECK = ROOT.parent.parent / "public" / "decks" / "2026-10-vpt-visual-prompt-tuning.html"
TASKS = ("dermamnist", "bloodmnist", "pneumoniamnist")
METHODS = {"linear": "linear", "vpt": "vpt-deep-p50", "full": "full"}


def load(task, method, regime):
    p = W / f"{task}_{METHODS[method]}_{regime}.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    st = W / f"{task}_{METHODS[method]}_{regime}.safetensors"
    return {"test": m["test"], "trainable": m["trainable"], "trainable_pct": round(m["trainable_pct"], 3),
            "train_images": m["train_images"], "train_minutes": m["train_minutes"], "epochs": m["epochs"],
            "base_lr": m["base_lr"], "file_kb": round(st.stat().st_size / 1024, 1) if st.exists() else None}


results = {"runs": {}}
for task in TASKS:
    for regime in ("full", "1k"):
        runs = {k: load(task, k, regime) for k in METHODS}
        if any(runs.values()):
            results["runs"][f"{task}_{regime}"] = {k: v for k, v in runs.items() if v}
bench = W / "bench.json"
if bench.exists():
    results["bench"] = json.loads(bench.read_text())

html = DECK.read_text(encoding="utf8")
html, n = re.subn(r"const RESULTS = .*?;\n", "const RESULTS = " + json.dumps(results) + ";\n", html, count=1)
assert n == 1, "const RESULTS not found in the deck"
DECK.write_text(html, encoding="utf8")
print(json.dumps({k: {m: (v["test"].get("bal_acc"), v["file_kb"]) for m, v in r.items()} for k, r in results["runs"].items()}, indent=1))
