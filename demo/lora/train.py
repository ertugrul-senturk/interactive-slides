"""Train LoRA (rank 8 on W_q and W_v) on a frozen ViT-B/16, for the same three MedMNIST+ tasks as the VPT demo.

The data, augmentation, splits, schedule and evaluation are exactly those of ../vpt/train.py (imported from
there), so LoRA lines up with the linear-probe and full fine-tuning runs already in ../vpt/weights.
Optimiser: AdamW, as in the LoRA paper; lr = base_lr * batch / 256, warm-up then cosine decay, model
selection on the val split (balanced accuracy), final numbers on the test split.

  python train.py --data dermamnist --subset 1000 --sweep 0.002 0.008 0.032
  python train.py --data dermamnist --base-lr 0.008 --epochs 30 --eval-every 3
"""
import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file
from torchvision import transforms

ROOT = Path(__file__).parent
VPT = ROOT.parent / "vpt"
sys.path.insert(0, str(VPT))
_spec = importlib.util.spec_from_file_location("vpt_train", VPT / "train.py")
vpt_train = sys.modules["vpt_train"] = importlib.util.module_from_spec(_spec)   # registered so DataLoader workers can unpickle its dataset
_spec.loader.exec_module(vpt_train)
MEDMNIST, build, loaders, evaluate = vpt_train.MEDMNIST, vpt_train.build, vpt_train.loaders, vpt_train.evaluate

from lora_model import BACKBONE, LoRAViT, count  # noqa: E402


def run(args, base_lr, epochs, dl, dev, n_cls, log=True):
    torch.manual_seed(args.seed)
    model = LoRAViT(n_cls, rank=args.rank, alpha=args.alpha, dropout=args.dropout).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    lr = base_lr * args.batch / 256
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=args.wd)
    per = len(dl["train"])
    steps, warm = epochs * per, max(1, min(10, epochs // 10)) * per
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    loss_fn = nn.CrossEntropyLoss()

    best, best_state = None, None
    for ep in range(epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y in dl["train"]:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        if (ep + 1) % args.eval_every == 0 or ep == epochs - 1:
            m = evaluate(model, dl["val"], dev, n_cls)
            if best is None or m[args.select] > best[args.select]:
                best, best_state = m, {k: v.detach().cpu().clone() for k, v in model.trainable_state_dict().items()}
            if log:
                print(f"  ep {ep + 1:3d}  loss {tot / per:.3f}  val {m}  ({time.time() - t0:.1f}s/ep)", flush=True)
    return model, best, best_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=list(MEDMNIST), default="dermamnist")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=8, help="scale alpha / rank; alpha = rank as in the paper (not tuned)")
    ap.add_argument("--dropout", type=float, default=0.1, help="dropout on the LoRA input")
    ap.add_argument("--subset", type=int, default=0, help="train on this many random train images (0 = all)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--base-lr", type=float, default=None)
    ap.add_argument("--sweep", type=float, nargs="+", default=[0.008], help="base learning rates to try on val")
    ap.add_argument("--sweep-epochs", type=int, default=20)
    ap.add_argument("--select", default="bal_acc", help="val metric for model selection")
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg = LoRAViT(2, rank=0).vit.pretrained_cfg
    sets, names = build(args.data, transforms.Normalize(cfg["mean"], cfg["std"]))
    dl = loaders(sets, args.batch, args.workers, args.subset, args.seed)   # same seed -> the same 1,000 images as VPT
    n_cls = len(names)
    print(f"{args.data}: {len(dl['train'].dataset)} train / {len(sets['val'])} val / {len(sets['test'])} test, {n_cls} classes", flush=True)

    base_lr = args.base_lr if args.base_lr is not None else args.sweep[0]
    if args.base_lr is None and len(args.sweep) > 1:
        scores = {}
        for lr in args.sweep:
            _, m, _ = run(args, lr, args.sweep_epochs, dl, dev, n_cls, log=False)
            scores[lr] = m[args.select]
            print(f"sweep base_lr {lr:<6} -> val {m}", flush=True)
        base_lr = max(scores, key=scores.get)
        print(f"picked base_lr {base_lr}", flush=True)

    t0 = time.time()
    model, val, state = run(args, base_lr, args.epochs, dl, dev, n_cls)
    model.load_state_dict({k: v for k, v in state.items() if not k.startswith("_")}, strict=False)
    test = evaluate(model, dl["test"], dev, n_cls)
    minutes = (time.time() - t0) / 60

    tag = f"lora-r{args.rank}" + (f"_{args.subset // 1000}k" if args.subset else "_full")
    out = ROOT / "weights"
    out.mkdir(exist_ok=True)
    save_file(state, out / f"{args.data}_{tag}.safetensors")
    meta = {"task": args.data, "method": tag, "classes": names, "backbone_id": BACKBONE, "rank": args.rank, "alpha": args.alpha,
            "targets": ["query", "value"], "dropout": args.dropout, "train_images": len(dl["train"].dataset), "base_lr": base_lr,
            "epochs": args.epochs, "select": args.select, "val": val, "test": test, "train_minutes": round(minutes, 1), **count(model)}
    (out / f"{args.data}_{tag}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "classes"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
