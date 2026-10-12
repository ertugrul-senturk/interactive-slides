"""Writes the notebooks of this folder (without outputs); run_all.py then executes them in place.

Kept as a script so the notebooks can be regenerated after a code change instead of edited by hand.
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
SETUP = '''import sys, json, time, math
from pathlib import Path
ROOT = Path.cwd().parent            # demo/lora
VPT = ROOT.parent / "vpt"           # data, linear-probe and full fine-tuning runs
sys.path.insert(0, str(ROOT))
import numpy as np, torch
print("torch", torch.__version__, "| CUDA:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'''


def nb(name, cells):
    book = nbf.v4.new_notebook()
    book.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    book.cells = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("md:") else nbf.v4.new_code_cell(c) for c in cells]
    nbf.write(book, HERE / name)
    print("wrote", name)


nb("01_lora_model.ipynb", [
'''md:# 1 · LoRA in code

This notebook builds the model used in every LoRA experiment of the talk and checks it against the paper
(Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022, Sec. 4).

**Base model.** The same as in the VPT talk: ViT-B/16 pretrained with labels on ImageNet-21k
(`vit_base_patch16_224.orig_in21k` in timm), 85.8M parameters, 12 blocks of width d = 768.

**LoRA.** The base model stays frozen. Beside a chosen weight matrix W (d × d), LoRA trains two thin matrices,
A (r × d) and B (d × r), and adds their product to the layer's output:

  h = W x + (α / r) · B A x

B starts at zero, so training starts exactly from the pretrained model. As in the paper's default, LoRA goes beside
the query and value weights of every attention block, with rank r = 8 and α = r. Only the pairs and a linear
classification head are trained. After training, B A can be added into W once, and the model is again a plain ViT.''',
SETUP,
'''md:## Where the query and value weights are

timm computes query, key and value of a block with one fused 768 → 2304 matrix, `qkv`. Its first 768 output rows
are the query weight W_q, the next 768 the key, the last 768 the value W_v. LoRA has to update the first and last
thirds only.''',
'''import timm
vit = timm.create_model("vit_base_patch16_224.orig_in21k", pretrained=True, num_classes=0)
n_params = sum(p.numel() for p in vit.parameters())
print(f"{n_params:,} parameters | width {vit.embed_dim} | blocks {len(vit.blocks)}")
print("block 1 attention:", vit.blocks[0].attn.qkv, "-> rows [0:768] query, [768:1536] key, [1536:2304] value")''',
'''md:## The LoRA layer

`lora_model.py` wraps each frozen `qkv` in a `LoRAQKV` module. Printed here is its forward pass: the frozen
output plus B(Ax) on the query and value slices.''',
'''import inspect
from lora_model import LoRAQKV, LoRAViT, count, load_task
print(inspect.getsource(LoRAQKV.forward))''',
'''md:## Parameter counts against the paper

Each pair has r × d + d × r = 2dr numbers. With query and value in 12 blocks, d = 768 and r = 8:
2 × 768 × 8 × 2 × 12 = 294,912. The head comes on top (here the 7 classes of DermaMNIST).
For comparison, VPT-deep with 50 prompts per layer trained 460,800 numbers.''',
'''base = LoRAViT(7, rank=0, vit=vit)
print(f"{'':22s}{'pairs':>10s}{'head':>8s}{'trainable':>12s}{'% of backbone':>15s}")
for r in (1, 2, 4, 8, 16, 64):
    c = count(LoRAViT(7, rank=r, vit=vit))
    print(f"{'LoRA r = ' + str(r):22s}{c['lora']:>10,}{c['head']:>8,}{c['trainable']:>12,}{c['trainable_pct']:>14.3f}%")
    assert c["lora"] == 2 * 768 * r * 2 * 12''',
'''md:## At step zero, LoRA is the pretrained model

B is zero at initialisation, so B A x = 0 and the features are exactly those of the frozen base model.''',
'''torch.manual_seed(0)
x = torch.randn(2, 3, 224, 224)
with torch.no_grad():
    plain = LoRAViT(7, rank=0, vit=vit).eval().forward_features(x)
    m = LoRAViT(7, rank=8, vit=vit).eval()
    print("largest feature difference at init:", (m.forward_features(x) - plain).abs().max().item())''',
'''md:## Only the pairs and the head learn

One training step with a random batch: the frozen weights receive no gradient, the A and B matrices of every block
do. The gradient still flows *through* every frozen layer to reach them, which is why LoRA's training memory on a
ViT is not much lower than full fine-tuning's (notebook 3).''',
'''m = LoRAViT(7, rank=8, vit=vit).train()
loss = torch.nn.functional.cross_entropy(m(torch.randn(4, 3, 224, 224)), torch.randint(0, 7, (4,)))
loss.backward()
frozen = sum(1 for n, p in m.vit.named_parameters() if p.grad is not None and not n.endswith((".A", ".B")))
print("frozen tensors that received a gradient:", frozen)
print("B gradient per block:", [f"{l.B.grad.abs().sum():.1e}" for l in m.lora_layers()])
print("A gradient per block (zero at the first step, because B = 0):", [f"{l.A.grad.abs().sum():.0e}" for l in m.lora_layers()][:4], "...")
m.zero_grad()''',
'''md:## Merging: the same outputs, the plain model's size and speed

With trained-looking B matrices (random here), `merge()` adds (α / r) B A into each `qkv` weight and removes the
bypass. The outputs agree up to float rounding, and the parameter count is back to the base model's.''',
'''import copy
m = LoRAViT(7, rank=8, vit=copy.deepcopy(vit)).eval()     # a copy, because merging changes the frozen weights
for l in m.lora_layers():
    l.B.data.normal_(0, 0.02)
torch.set_num_threads(4)
with torch.no_grad():
    def ms(f, n=5):
        f(); t0 = time.perf_counter()
        for _ in range(n): f()
        return (time.perf_counter() - t0) / n * 1000
    before, t_before = m(x), ms(lambda: m(x[:1]))
    m.merge()
    after, t_after = m(x), ms(lambda: m(x[:1]))
print("largest output difference after merging:", (after - before).abs().max().item())
print("parameters after merging:", f"{sum(p.numel() for p in m.vit.parameters()):,}", "(base model:", f"{n_params:,})")
print(f"CPU time per image: {t_before:.0f} ms with the bypass, {t_after:.0f} ms merged")''',
])

nb("02_training.ipynb", [
'''md:# 2 · Training LoRA

The training code behind every LoRA number in the talk. It is the VPT demo's pipeline (`../vpt/train.py`: the same
three MedMNIST+ datasets, splits, 1,000-image subsets with the same seed, augmentation, schedule and evaluation),
with LoRA in place of the prompts:

| | linear probe | LoRA | full fine-tuning |
| --- | --- | --- | --- |
| trained | head | rank-8 pairs on W_q, W_v in 12 blocks + head | every weight + head |
| optimiser | SGD, momentum 0.9 | AdamW, weight decay 0.01 (as in the LoRA paper) | AdamW |
| learning rate | base LR × batch / 256, base LR picked on val | same | same |
| schedule | warm-up, then cosine decay | same | same |
| model selection | best validation balanced accuracy | same | same |

Linear probe and full fine-tuning are not retrained: they are the VPT talk's runs (`../vpt/weights`). The data
itself is shown in `../vpt/notebooks/02_data_medmnist.ipynb`.''',
SETUP,
'''# ---- configuration -------------------------------------------------------------------------------
TASK   = "dermamnist"        # dermamnist | bloodmnist | pneumoniamnist
QUICK  = True                # True: 1,000 images, 12 epochs, one learning rate (about two minutes)
SUBSET = 1000
EPOCHS = 12 if QUICK else 100
SWEEP  = [0.004] if QUICK else [0.001, 0.004, 0.016]''',
'''md:## Data and model

`train.py` imports the data code from the VPT folder, so nothing about the data differs between the two talks.''',
'''from types import SimpleNamespace
from torchvision import transforms
import train as T
sets, classes = T.build(TASK, transforms.Normalize([0.5] * 3, [0.5] * 3))
dl = T.loaders(sets, 64, 2, SUBSET, 0)
print(f"{TASK}: {len(dl['train'].dataset)} train / {len(sets['val'])} val / {len(sets['test'])} test | classes: {classes}")
args = SimpleNamespace(rank=8, alpha=8, dropout=0.1, batch=64, wd=0.01, seed=0, eval_every=3, select="bal_acc")''',
'''md:## The training loop

`train.run` builds a `LoRAViT`, gives AdamW only the parameters that require gradients (the pairs and the head), and
keeps the checkpoint with the best validation balanced accuracy. Printed here:''',
'''import inspect
print(inspect.getsource(T.run))''',
'''md:## Run it

Learning-rate sweep on the validation split (skipped when `QUICK`), then training with the chosen rate, then one
score on the test split.''',
'''t_start = time.time()
if len(SWEEP) > 1:
    scores = {lr: T.run(args, lr, 20, dl, "cuda", len(classes), log=False)[1]["bal_acc"] for lr in SWEEP}
    print("sweep (val balanced accuracy):", scores)
    base_lr = max(scores, key=scores.get)
else:
    base_lr = SWEEP[0]
model, val, state = T.run(args, base_lr, EPOCHS, dl, "cuda", len(classes))
model.load_state_dict({k: v for k, v in state.items() if not k.startswith("_")}, strict=False)
test = T.evaluate(model, dl["test"], "cuda", len(classes))
print(f"\\nLoRA on {TASK}, {len(dl['train'].dataset)} training images, base LR {base_lr}: best val {val} | test {test}")
print(f"stored per task: {sum(v.numel() for v in state.values()) * 4 / 1e6:.2f} MB | {(time.time() - t_start) / 60:.1f} min")''',
'''md:## The real runs

The talk's numbers come from the full runs, launched from the command line so they survive a closed notebook
(`sh run_lora.sh`). Each writes its pairs and head to `weights/<task>_lora-r8_<1k|full>.safetensors` and its metrics
to the matching JSON. The validation curves of the DermaMNIST 1,000-image runs, LoRA beside the VPT talk's three
methods:''',
'''import re, matplotlib.pyplot as plt
def curve(path):
    return [(int(m.group(1)), float(m.group(2))) for m in re.finditer(r"ep\\s+(\\d+).*?'bal_acc': ([\\d.]+)", path.read_text(errors="ignore"))]
fig, ax = plt.subplots(figsize=(8, 4))
for name, path, color, ls in [("linear probe", VPT / "logs" / "derma_linear_1k.log", "#9EA3A8", "-"),
                              ("VPT-deep", VPT / "logs" / "derma_vpt_1k.log", "#9EA3A8", ":"),
                              ("LoRA", ROOT / "logs" / "dermamnist_lora_1k.log", "#BF3F2A", "-"),
                              ("full fine-tuning", VPT / "logs" / "derma_full_1k.log", "#1C1F23", "--")]:
    if path.exists() and curve(path):
        ax.plot(*zip(*curve(path)), label=name, color=color, ls=ls, lw=3 if name == "LoRA" else 1.5)
ax.set_xlabel("epoch"); ax.set_ylabel("val balanced accuracy %"); ax.set_title("DermaMNIST, 1,000 training images"); ax.legend()
plt.tight_layout(); plt.show()''',
])

nb("03_results.ipynb", [
'''md:# 3 · Results and cost

Every number of the LoRA experiments, read from the files the training runs wrote: `weights/*.json` and
`weights/bench.json` here, the linear-probe, VPT-deep and full fine-tuning runs from `../vpt/weights`.''',
SETUP,
'''TASKS = {"dermamnist": "Skin lesion", "bloodmnist": "Blood cell", "pneumoniamnist": "Chest X-ray"}
METHODS = {"linear": (VPT / "weights", "linear"), "vpt": (VPT / "weights", "vpt-deep-p50"),
           "lora": (ROOT / "weights", "lora-r8"), "full": (VPT / "weights", "full")}
R = {(t, m, r): json.loads((d / f"{t}_{tag}_{r}.json").read_text()) for t in TASKS for m, (d, tag) in METHODS.items() for r in ("1k", "full")}
print(f"{'task':12s}{'method':8s}{'images':>8s}{'acc':>8s}{'bal acc':>9s}{'AUC':>8s}{'stored':>11s}")
for t, name in TASKS.items():
    for r in ("1k", "full"):
        for m, (d, tag) in METHODS.items():
            e = R[(t, m, r)]; size = (d / f"{t}_{tag}_{r}.safetensors").stat().st_size
            print(f"{name:12s}{m:8s}{e['train_images']:>8,}{e['test']['acc']:>8.1f}{e['test']['bal_acc']:>9.1f}{e['test']['auc'] / 100:>8.3f}{size / 1e6:>9.2f} MB")
        print()''',
'''md:## Balanced accuracy, two training-set sizes

One group of bars per task and training-set size. LoRA in red, full fine-tuning dark, the two other methods grey.''',
'''import matplotlib.pyplot as plt
colors = {"linear": "#C9CCD0", "vpt": "#9EA3A8", "lora": "#BF3F2A", "full": "#1C1F23"}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, (t, name) in zip(axes, TASKS.items()):
    for g, r in enumerate(("1k", "full")):
        for j, m in enumerate(METHODS):
            v = R[(t, m, r)]["test"]["bal_acc"]
            ax.bar(g * 5 + j, v, color=colors[m], label=m if g == 0 else None)
            ax.text(g * 5 + j, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_xticks([1.5, 6.5], ["1,000 images", f"{R[(t, 'lora', 'full')]['train_images']:,} images"])
    ax.set_ylim(0, 105); ax.set_title(name); ax.set_ylabel("test balanced accuracy %")
axes[0].legend(ncol=4, fontsize=8); plt.tight_layout(); plt.show()
for t, name in TASKS.items():
    for r in ("1k", "full"):
        a = {m: R[(t, m, r)]["test"]["bal_acc"] for m in METHODS}
        print(f"{name:12s} {r:4s}  LoRA - VPT {a['lora'] - a['vpt']:+5.1f}   LoRA - linear {a['lora'] - a['linear']:+5.1f}   LoRA - full {a['lora'] - a['full']:+5.1f}")''',
'''md:## What each method costs

`bench.py` measures training memory and time at batch 32 (bf16) and inference on 2 CPU threads, with the same
protocol as the VPT talk's `bench.py`, all four methods in one session. LoRA is timed at inference twice: with the
bypass, and merged.''',
'''b = json.loads((ROOT / "weights" / "bench.json").read_text())
print(f"GPU: {b['gpu']} | batch {b['train_batch']} | CPU threads {b['cpu_threads']}\\n")
print(f"{'method':14s}{'train GB':>10s}{'train ms/img':>14s}{'CPU ms/img':>12s}")
for m in ("linear", "vpt", "lora", "full"):
    e = b[m]
    print(f"{m:14s}{e['train_peak_gb']:>10.2f}{e['train_ms_per_img']:>14.2f}{e['cpu_ms_per_img']:>12d}")
print(f"{'lora, merged':14s}{'':>10s}{'':>14s}{b['lora_merged']['cpu_ms_per_img']:>12d}")''',
'''md:## Re-scoring a saved model, with the bypass and merged

Loads the DermaMNIST LoRA file (full split) onto the base model and scores the whole test split, then merges B A into
the weights and scores again. `evaluate.py` does this for all six LoRA models.''',
'''from torch.utils.data import DataLoader
from torchvision import transforms
from safetensors.torch import load_file
from train import build, evaluate
sets, classes = build("dermamnist", transforms.Normalize([0.5] * 3, [0.5] * 3))
dl = DataLoader(sets["test"], batch_size=128, num_workers=0)
from lora_model import load_task
model = load_task(load_file(ROOT / "weights" / "dermamnist_lora-r8_full.safetensors"), len(classes)).cuda()
with torch.no_grad():
    got = evaluate(model, dl, "cuda", len(classes))
    merged = evaluate(model.merge(), dl, "cuda", len(classes))
print("stored    ", R[("dermamnist", "lora", "full")]["test"])
print("re-scored ", got)
print("merged    ", merged)''',
])

nb("04_browser_export.ipynb", [
'''md:# 4 · Exporting the models for the browser

The slide decks run the models themselves, in the viewer's browser, with onnxruntime-web, as in the VPT talk. This
notebook shows the export (`export_onnx.py` does it for all models) and the check that the browser runs exactly
what was trained.

One ONNX graph of the frozen base model serves three methods. Besides the image it takes both kinds of task file:
VPT prompts (B, 12, P, 768), inserted before every block, and LoRA pairs A (B, 12, 2, R, 768) and B (B, 12, 2, 768, R)
beside the query and value weights of every block, with α / r folded into B.

* no prompts and zero pairs: the plain base model (the linear probe's features)
* prompts only: VPT-deep
* pairs only: LoRA

Because the task files are inputs, every image of a batch could even carry a different task. LoRA runs *unmerged*
here, on purpose: merged, it would be a separate 172 MB model per task, like full fine-tuning. The fully fine-tuned
models and the VPT prompt files are the VPT talk's, loaded from its repository.''',
SETUP,
'''from export_onnx import Encoder, export
from lora_model import LoRAViT
torch.set_grad_enabled(False)
tmp = ROOT / "web" / "notebook_check"; tmp.mkdir(parents=True, exist_ok=True)
vit = LoRAViT(2, rank=0).vit.eval()
t0 = time.time()
sess = export(vit, tmp / "enc.fp16.onnx")            # also prints the check against PyTorch on random inputs
print(f"exported and checked in {time.time() - t0:.0f}s: {(tmp / 'enc.fp16.onnx').stat().st_size / 1e6:.0f} MB")
enc = Encoder(vit).eval()''',
'''md:## Does the browser model compute the same thing?

Cosine similarity between the PyTorch features and the fp16 ONNX features on 16 DermaMNIST test images: with nothing
fed in, with the real VPT-deep prompts, and with the real LoRA pairs.''',
'''from safetensors.torch import load_file
from lora_model import load_task
x = np.load(VPT / "data" / "dermamnist_224" / "test_images.npy", mmap_mode="r")[:16]
img = ((np.asarray(x).astype(np.float32) / 255 - 0.5) / 0.5).transpose(0, 3, 1, 2).copy()
P = load_file(VPT / "weights" / "dermamnist_vpt-deep-p50_full.safetensors")["prompts"].numpy()
A, B = (z.numpy() for z in load_task(load_file(ROOT / "weights" / "dermamnist_lora-r8_full.safetensors"), 7, pretrained=False).stacked())
rep = lambda a: np.broadcast_to(a, (16, *a.shape)).copy()
none_p, za, zb = np.zeros((16, 12, 0, 768), np.float32), np.zeros((16, 12, 2, 1, 768), np.float32), np.zeros((16, 12, 2, 768, 1), np.float32)
for name, pr, pa, pb in [("nothing (linear probe)", none_p, za, zb), ("VPT-deep prompts", rep(P), za, zb), ("LoRA pairs, rank 8", none_p, rep(A), rep(B))]:
    ref = enc(*(torch.from_numpy(v) for v in (img, pr, pa, pb))).numpy()
    got = sess.run(None, {"image": img, "prompts": pr, "lora_a": pa, "lora_b": pb})[0]
    cos = (ref * got).sum(1) / np.linalg.norm(ref, axis=1) / np.linalg.norm(got, axis=1)
    print(f"{name:24s} cosine to PyTorch: min {cos.min():.6f}")''',
'''md:## Where the decks get the models

`upload_models.py` publishes `web/` to the free model repository
[valinor61/lora-medmnist](https://huggingface.co/valinor61/lora-medmnist). The decks fetch `peft_encoder.fp16.onnx`
and the task's 1.2 MB LoRA file from there, and the VPT prompt file and the fully fine-tuned model from
[valinor61/vpt-medmnist](https://huggingface.co/valinor61/vpt-medmnist), on first use. They keep them in the browser's
Cache Storage (shared with the VPT deck) and run them with onnxruntime-web. `heads.json` holds every classification
head and the test metrics of exactly these ONNX models:''',
'''heads = json.loads((ROOT / "web" / "heads.json").read_text())
for t, e in heads["tasks"].items():
    for key in ("methods_1k", "methods"):
        print(f"{t:15s} {key:10s}", {m: (v["test_torch"]["bal_acc"], v["test_web"]["bal_acc"]) for m, v in e[key].items()}, "(PyTorch, ONNX balanced accuracy)")
del sess
import shutil; shutil.rmtree(tmp, ignore_errors=True)   # the check files are not needed''',
])

