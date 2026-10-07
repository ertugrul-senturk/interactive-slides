"""Exports test-set images for the demo (deck thumbnails and Space examples) and writes the list into the deck.

Picks are random (fixed seed), not chosen by how well the model does on them:
one per class for DermaMNIST and BloodMNIST, three per class for PneumoniaMNIST.
"""
import json
import re
from pathlib import Path

import numpy as np
from medmnist import INFO
from PIL import Image

ROOT = Path(__file__).parent
DECK = ROOT.parent.parent / "public" / "decks" / "2026-10-vpt-visual-prompt-tuning.html"
ASSETS = DECK.parent / "2026-10-vpt"
PER_CLASS = {"dermamnist": 1, "bloodmnist": 1, "pneumoniamnist": 3}

rng = np.random.default_rng(7)
samples = {}
for task, k in PER_CLASS.items():
    names = [INFO[task]["label"][str(i)] for i in range(len(INFO[task]["label"]))]
    with np.load(ROOT / "data" / f"{task}_224.npz") as z:
        x, y = z["test_images"], z["test_labels"][:, 0]
    ex = ROOT / "examples" / task
    ex.mkdir(parents=True, exist_ok=True)
    for old in ex.glob("*.png"):
        old.unlink()
    samples[task] = []
    for c in range(len(names)):
        for j, i in enumerate(rng.choice(np.flatnonzero(y == c), size=k, replace=False)):
            img = Image.fromarray(x[i]).convert("RGB")
            fname = f"{task}-{c}-{j}.png"
            img.save(ASSETS / fname)
            img.save(ex / f"{c}-{j}.png")
            samples[task].append({"file": fname, "label": names[c], "index": int(i)})
    print(task, len(samples[task]), "images")

for old in ASSETS.glob("derma-*.png"):   # names from the single-task version of the demo
    old.unlink()
html = DECK.read_text(encoding="utf8")
html = re.sub(r"const SAMPLES = .*?;\n", "const SAMPLES = " + json.dumps(samples) + ";\n", html, count=1)
DECK.write_text(html, encoding="utf8")
