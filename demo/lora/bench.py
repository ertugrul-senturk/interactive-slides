"""Measures what each adaptation costs, under identical conditions, and writes weights/bench.json.

  python bench.py      # run with nothing else on the GPU

Same protocol as ../vpt/bench.py, and VPT-deep (50 prompts per layer, ../vpt/vpt_model.py) is measured again here, so
all four methods are timed in one session on the same machine. Training: ViT-B/16 at 224 x 224, batch 32,
bf16 autocast, one optimiser step as in train.py; time per image and peak GPU memory. Inference: batch 1 on CPU
with 2 threads. LoRA is timed twice at inference: with the bypass (as trained, and as in the browser demo) and
after merging B A into the weights, which is how the paper deploys it.
"""
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

from lora_model import LoRAViT

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent / "vpt"))
from vpt_model import VPTViT  # noqa: E402


def make(method):
    if method == "vpt":
        return VPTViT(7, num_prompts=50, deep=True)
    m = LoRAViT(7, rank=8 if method.startswith("lora") else 0)
    if method == "full":
        for p in m.vit.parameters():
            p.requires_grad = True
    if method == "lora_merged":
        for l in m.lora_layers():
            l.B.data.normal_(0, 0.01)   # a trained-looking B, so the merge really adds something
        m.merge()
    return m


def train_cost(method, batch=32, warm=5, steps=20):
    m = make(method).cuda().train()
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-5) if method in ("full", "lora") else torch.optim.SGD(params, lr=0.01, momentum=0.9)   # as in training
    x, y = torch.randn(batch, 3, 224, 224, device="cuda"), torch.randint(0, 7, (batch,), device="cuda")
    loss_fn = nn.CrossEntropyLoss()
    torch.cuda.reset_peak_memory_stats()
    for i in range(warm + steps):
        if i == warm:
            torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_fn(m(x), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / steps / batch * 1000
    mem = torch.cuda.max_memory_allocated() / 2**30
    del m, opt
    torch.cuda.empty_cache()
    return round(ms, 2), round(mem, 2)


@torch.no_grad()
def cpu_inference(method, threads=2, warm=5, runs=40):   # more runs than ../vpt/bench.py: the four CPU times are close
    torch.set_num_threads(threads)
    m = make(method).eval()
    x = torch.randn(1, 3, 224, 224)
    for _ in range(warm):
        m(x)
    t0 = time.perf_counter()
    for _ in range(runs):
        m(x)
    return round((time.perf_counter() - t0) / runs * 1000)


def main():
    out = {"gpu": torch.cuda.get_device_name(0), "train_batch": 32, "cpu_threads": 2}
    for method in ("linear", "vpt", "lora", "full"):
        ms, mem = train_cost(method)
        out[method] = {"train_ms_per_img": ms, "train_peak_gb": mem, "cpu_ms_per_img": cpu_inference(method)}
        print(method, out[method], flush=True)
    out["lora_merged"] = {"cpu_ms_per_img": cpu_inference("lora_merged")}
    print("lora_merged", out["lora_merged"], flush=True)
    (ROOT / "weights" / "bench.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
