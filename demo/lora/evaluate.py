"""Re-scores the saved LoRA models on their test split and checks the numbers stored in weights/*.json.

  python evaluate.py                          # every LoRA result file
  python evaluate.py dermamnist_lora-r8_1k    # one of them

Loads exactly what the demo loads (frozen backbone + saved pairs and head), so it also serves as an end-to-end
check of the saved files. It scores each model twice: with the LoRA bypass, and after merging B A into the
frozen weights. The two must agree, which is the paper's claim that merging changes nothing.
The linear-probe and full fine-tuning baselines are checked by ../vpt/evaluate.py.
"""
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from torchvision import transforms

from train import MEDMNIST, build, evaluate
from lora_model import LoRAViT, load_task

ROOT = Path(__file__).parent


def main():
    tags = sys.argv[1:] or sorted(p.stem for p in (ROOT / "weights").glob("*.json") if p.stem.split("_")[0] in MEDMNIST)
    cfg = LoRAViT(2, rank=0).vit.pretrained_cfg
    norm = transforms.Normalize(cfg["mean"], cfg["std"])
    cache = {}
    for tag in tags:
        meta_path = ROOT / "weights" / f"{tag}.json"
        meta = json.loads(meta_path.read_text())
        data = meta["task"]
        if data not in cache:
            sets, _ = build(data, norm)
            cache[data] = DataLoader(sets["test"], batch_size=128, num_workers=0)   # workers crashed here on a busy laptop; the test sets are small
        m = load_task(load_file(ROOT / "weights" / f"{tag}.safetensors"), len(meta["classes"])).cuda()
        got = evaluate(m, cache[data], "cuda", len(meta["classes"]))
        merged = evaluate(m.merge(), cache[data], "cuda", len(meta["classes"]))
        same = all(abs(got[k] - meta["test"][k]) < 0.05 for k in meta["test"])
        print(f"{tag:32s} stored {meta['test']}  re-scored {got}  {'OK' if same else 'MISMATCH'}  merged {merged}", flush=True)
        meta["test_merged"] = merged
        meta_path.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    with torch.no_grad():
        main()
