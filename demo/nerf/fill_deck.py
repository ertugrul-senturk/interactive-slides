"""Write my measured globe results into the NeRF deck.

demo/nerf/globe_results.json comes from nerf-globe/tools/deck_assets.py (NeRF and Gaussian splatting on my capture).
This replaces the deck's single line `const GLOBE = ...;` with that data, so every number on the
"My own capture" slides is rendered from the measurements, never typed in.

    python demo/nerf/fill_deck.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DECK = ROOT / "public" / "decks" / "2026-11-nerf-neural-radiance-fields.html"
data = json.loads((Path(__file__).with_name("globe_results.json")).read_text())
html = DECK.read_text(encoding="utf-8")
line = "const GLOBE = " + json.dumps(data, separators=(",", ":")) + ";"
new, n = re.subn(r"^const GLOBE = .*;$", lambda m: line, html, count=1, flags=re.M)
if n != 1:
    raise SystemExit("marker line `const GLOBE = ...;` not found in the deck")
DECK.write_text(new, encoding="utf-8")
print(f"filled {DECK.name}: {len(line):,} characters")
