"""Re-scores saved models (VPT, linear probe and full fine-tuning) on their test split and checks the numbers stored in weights/*.json.

  python evaluate.py                       # every VPT / linear result file
  python evaluate.py dermamnist_linear_1k  # one of them

Loads exactly what the demo loads (frozen backbone + saved prompts and head), so it also serves as
an end-to-end check of the saved files. Missing metrics (e.g. AUC) are added to the json.
"""
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from torchvision import transforms

from train import MEDMNIST, build, evaluate
from vpt_model import VPTViT

ROOT = Path(__file__).parent


def main():
    tags = sys.argv[1:] or sorted(p.stem for p in (ROOT / "weights").glob("*.json") if p.stem.split("_")[0] in MEDMNIST)
    vit = VPTViT(2, num_prompts=0).vit
    norm = transforms.Normalize(vit.pretrained_cfg["mean"], vit.pretrained_cfg["std"])
    cache = {}
    for tag in tags:
        meta_path = ROOT / "weights" / f"{tag}.json"
        meta = json.loads(meta_path.read_text())
        data = meta["task"]
        if data not in cache:
            sets, _ = build(data, norm)
            cache[data] = DataLoader(sets["test"], batch_size=128, num_workers=2)
        state = load_file(ROOT / "weights" / f"{tag}.safetensors")
        if meta["method"].startswith("full"):   # a whole fine-tuned backbone, not a prompt file
            m = VPTViT(len(meta["classes"]), num_prompts=0, pretrained=False)
            m.load_state_dict(state)
        else:
            m = VPTViT(len(meta["classes"]), num_prompts=meta["num_prompts"], deep=meta["deep"], vit=vit)
            m.load_state_dict(state, strict=False)
        m = m.cuda()
        got = evaluate(m, cache[data], "cuda", len(meta["classes"]))
        same = all(abs(got[k] - meta["test"][k]) < 0.05 for k in meta["test"])
        print(f"{tag:40s} stored {meta['test']}  re-scored {got}  {'OK' if same else 'MISMATCH'}", flush=True)
        if same and set(got) - set(meta["test"]):
            meta["test"] = got
            meta_path.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    with torch.no_grad():
        main()
