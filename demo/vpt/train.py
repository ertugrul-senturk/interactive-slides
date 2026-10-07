"""Train VPT-deep (and a linear-probe baseline) on a frozen ViT-B/16.

Recipe from the paper (Appendix A, Table 6): SGD momentum 0.9, lr = base_lr * batch / 256,
warm-up then cosine decay, prompt dropout 0.1, model selection on the val split, final
numbers on the test split.

  python train.py --data dermamnist --method vpt    --subset 1000 --sweep 2.5 10 25
  python train.py --data dermamnist --method vpt    --base-lr 10 --epochs 30
  python train.py --data dermamnist --method linear --subset 1000 --sweep 0.5 2.5 10

--subset N trains on N images drawn at random from the train split (VTAB-1k style);
val and test stay complete. Besides accuracy we report balanced accuracy (mean per-class
recall) and macro one-vs-rest AUC, the MedMNIST metric, because DermaMNIST is imbalanced.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Subset
from PIL import Image
from torchvision import datasets, transforms

from vpt_model import BACKBONE, VPTViT, count

ROOT = Path(__file__).parent
MEDMNIST = ("dermamnist", "bloodmnist", "pneumoniamnist")


class NpyImages(torch.utils.data.Dataset):
    """MedMNIST split stored as memory-mapped .npy files, so DataLoader workers share one copy in RAM."""

    def __init__(self, images, labels, transform):
        self.images, self.labels, self.transform = images, labels, transform
        self.x = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if self.x is None:  # open lazily inside each worker
            self.x = np.load(self.images, mmap_mode="r")
        return self.transform(Image.fromarray(np.asarray(self.x[i])).convert("RGB")), int(self.labels[i])


def medmnist_npy(npz, out, split):
    """Unpacks one split of a MedMNIST .npz into plain .npy files once."""
    out.mkdir(exist_ok=True)
    xi, yi = out / f"{split}_images.npy", out / f"{split}_labels.npy"
    if not xi.exists():
        with np.load(npz) as z:
            np.save(xi, z[f"{split}_images"])
            np.save(yi, z[f"{split}_labels"][:, 0])
    return xi, np.load(yi)


def build(name, norm):
    """Returns ({split: dataset}, class names)."""
    data = ROOT / "data"
    if name == "flowers102":
        from classes import FLOWERS102
        train_tf = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
        eval_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), norm])
        sets = {s: datasets.Flowers102(data, split=s, transform=train_tf if s == "train" else eval_tf, download=True)
                for s in ("train", "val", "test")}
        return sets, FLOWERS102
    if name in MEDMNIST:
        from medmnist import INFO
        info = INFO[name]
        if name == "pneumoniamnist":
            # Chest X-rays have a fixed anatomy (heart on the left), so no flips; mild crops only.
            aug = [transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1))]
        else:
            # Dermoscopy and blood smears have no canonical orientation, so both flips are safe.
            aug = [transforms.RandomResizedCrop(224, scale=(0.6, 1.0)), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
        train_tf = transforms.Compose(aug + [transforms.ToTensor(), norm])
        eval_tf = transforms.Compose([transforms.ToTensor(), norm])
        npz = data / f"{name}_224.npz"
        if not npz.exists():
            raise SystemExit(f"Missing {npz}. Download it from {info['url'].replace('.npz', '_224.npz')}")
        sets = {}
        for s in ("train", "val", "test"):
            xi, y = medmnist_npy(npz, data / f"{name}_224", s)
            sets[s] = NpyImages(xi, y, train_tf if s == "train" else eval_tf)
        names = [info["label"][str(i)] for i in range(len(info["label"]))]
        return sets, names
    raise ValueError(name)


def loaders(sets, batch, workers, subset, seed):
    if subset:
        g = np.random.default_rng(seed)
        idx = g.choice(len(sets["train"]), size=subset, replace=False)
        sets = {**sets, "train": Subset(sets["train"], idx.tolist())}
    kw = dict(batch_size=batch, num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    return {k: DataLoader(v, shuffle=(k == "train"), drop_last=(k == "train" and len(v) > batch), **kw) for k, v in sets.items()}


@torch.no_grad()
def evaluate(model, loader, dev, n_cls):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(x.to(dev, non_blocking=True)).float().softmax(-1)
        probs.append(p.cpu()); ys.append(y)
    p, y = torch.cat(probs).numpy(), torch.cat(ys).numpy()
    pred = p.argmax(1)
    acc = 100 * (pred == y).mean()
    recalls = [(pred[y == c] == c).mean() for c in range(n_cls) if (y == c).any()]
    out = {"acc": round(float(acc), 2), "bal_acc": round(float(100 * np.mean(recalls)), 2)}
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y, p[:, 1]) if n_cls == 2 else roc_auc_score(y, p, multi_class="ovr", average="macro")
        out["auc"] = round(float(100 * auc), 2)
    except Exception:
        pass
    return out


def run(args, base_lr, epochs, dl, dev, n_cls, log=True):
    torch.manual_seed(args.seed)
    model = VPTViT(n_cls, num_prompts=args.prompts if args.method == "vpt" else 0, deep=not args.shallow).to(dev)
    if args.method == "full":
        for p in model.vit.parameters():
            p.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad]
    lr = base_lr * args.batch / 256
    # Paper, Table 6: AdamW for full fine-tuning, SGD for linear probing and VPT.
    opt = (torch.optim.AdamW(params, lr=lr, weight_decay=args.wd) if args.method == "full"
           else torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=args.wd))
    keep = model.state_dict if args.method == "full" else model.trainable_state_dict
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
                best, best_state = m, {k: v.detach().cpu().clone() for k, v in keep().items()}
            if log:
                print(f"  ep {ep + 1:3d}  loss {tot / per:.3f}  val {m}  ({time.time() - t0:.1f}s/ep)", flush=True)
    return model, best, best_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["flowers102", *MEDMNIST], default="dermamnist")
    ap.add_argument("--method", choices=["vpt", "linear", "full"], default="vpt")
    ap.add_argument("--prompts", type=int, default=50)
    ap.add_argument("--shallow", action="store_true")
    ap.add_argument("--subset", type=int, default=0, help="train on this many random train images (0 = all)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--base-lr", type=float, default=None)
    ap.add_argument("--sweep", type=float, nargs="+", default=[10.0], help="base learning rates to try on val")
    ap.add_argument("--sweep-epochs", type=int, default=20)
    ap.add_argument("--select", default="bal_acc", help="val metric for model selection")
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg = VPTViT(2, num_prompts=0).vit.pretrained_cfg
    sets, names = build(args.data, transforms.Normalize(cfg["mean"], cfg["std"]))
    dl = loaders(sets, args.batch, args.workers, args.subset, args.seed)
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
    model.load_state_dict(state, strict=False)
    test = evaluate(model, dl["test"], dev, n_cls)
    minutes = (time.time() - t0) / 60

    tag = args.method if args.method in ("linear", "full") else f"vpt-{'shallow' if args.shallow else 'deep'}-p{args.prompts}"
    tag += f"_{args.subset // 1000}k" if args.subset else "_full"
    out = ROOT / "weights"
    out.mkdir(exist_ok=True)
    save_file(state, out / f"{args.data}_{tag}.safetensors")
    c = count(model)
    if args.method == "full":  # every backbone weight is task-specific now
        c["trainable"] = c["backbone"] + c["head"]
        c["trainable_pct"] = 100 * c["trainable"] / c["backbone"]
    meta = {"task": args.data, "method": tag, "classes": names, "backbone_id": BACKBONE, "num_prompts": model.num_prompts,
            "deep": model.deep, "train_images": len(dl["train"].dataset), "base_lr": base_lr, "epochs": args.epochs,
            "select": args.select, "val": val, "test": test, "train_minutes": round(minutes, 1), **c}
    (out / f"{args.data}_{tag}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "classes"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
