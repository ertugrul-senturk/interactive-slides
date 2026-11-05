"""Crop the training snapshots of all three models on my globe into same-size images for the deck's training slide.

Every snapshot shows the same held-out photo (the first test view), never trained on:
  NeRF        nerf-globe/outputs/full/snapshots/iter_XXXXXX.png       photo | render | depth, half resolution
  Instant-NGP nerf-globe/outputs/ngp/snapshots/step_XXXXXX.png        photo | render | depth, half resolution
  Gaussians   nerf-globe/outputs/gaussian_splatting/snapshots/step_XXXXXX.png   photo | render, full resolution
Writes public/decks/2026-11-nerf/globe/strip-<model>-<step>.jpg and strip-photo.jpg, plus the training time at
each snapshot (from the logs, where they have it) into demo/nerf/train_strip.json.

    python demo/nerf/train_strip.py
"""
import csv
import json
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "public" / "decks" / "2026-11-nerf" / "globe"
RUNS = Path.home() / "Desktop" / "nerf-globe" / "outputs"
SIZE = (320, 303)                      # the half-resolution crop around the globe
STEPS = {"nerf": [2500, 25000, 100000], "ngp": [6000, 12000, 20000], "gs": [1000, 7000, 30000]}


def panel(path, k, n):
    im = Image.open(path).convert("RGB")
    w = im.width // n
    return im.crop((k * w, 0, (k + 1) * w, im.height))


def crop(im):
    s = im.height / 303                # 1 for the half-resolution snapshots, 2 for the Gaussian ones
    return im.crop((int(100 * s), 0, int(420 * s), im.height)).resize(SIZE, Image.LANCZOS)


def seconds(log, col, step):
    with open(log) as f:
        for r in csv.DictReader(f):
            if int(r[col]) == step:
                return float(r["seconds"])


times = {}
for m, steps in STEPS.items():
    times[m] = {}
    for st in steps:
        if m == "nerf":
            im = panel(RUNS / "full" / "snapshots" / f"iter_{st:06d}.png", 1, 3)
            times[m][st] = seconds(RUNS / "full" / "log.csv", "iter", st)
        elif m == "ngp":
            im = panel(RUNS / "ngp" / "snapshots" / f"step_{st:06d}.png", 1, 3)
            times[m][st] = seconds(RUNS / "ngp" / "log.csv", "step", st)
        else:
            im = panel(RUNS / "gaussian_splatting" / "snapshots" / f"step_{st:06d}.png", 1, 2)
        crop(im).save(OUT / f"strip-{m}-{st:06d}.jpg", quality=90)

# the photo itself, from the last NeRF snapshot (same crop as the renders), with its text line painted out;
# the masked photo's background is white
ph = panel(RUNS / "full" / "snapshots" / "iter_100000.png", 0, 3)
ph.paste((255, 255, 255), (0, 0, ph.width, 36))
crop(ph).save(OUT / "strip-photo.jpg", quality=90)
(Path(__file__).with_name("train_strip.json")).write_text(json.dumps({"steps": STEPS, "seconds": times}, indent=1))
print(json.dumps(times))
