"""Inference server for the VPT talk. Runs as a Hugging Face Space or locally (python app.py).

One frozen ViT-B/16 (ImageNet-21k, 85.8M parameters) is loaded once and never changes.
For each medical task there are three adaptations of the same pretrained ViT:
  linear probe  a linear head on the frozen backbone                       (~21 KB per task)
  VPT-deep      12 layers x 50 prompt tokens + head, frozen backbone        (~1.8 MB per task)
  full          every weight fine-tuned, i.e. a separate backbone per task (~343 MB per task)
Tasks:
  dermamnist      skin lesions, dermoscopy, 7 classes
  bloodmnist      blood cells, microscopy, 8 classes
  pneumoniamnist  chest X-ray, 2 classes
Endpoints used by the slide deck (through @gradio/client):
  /classify  one image + a task: runs the linear probe, VPT-deep and the fully fine-tuned model
  /mixed     up to one image per task, all run in a single forward pass
Research demo only: not a diagnostic tool.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms

from vpt_model import MultiTaskVPT, VPTViT

ROOT = Path(__file__).parent
WEIGHTS = ROOT / "weights"
torch.set_num_threads(os.cpu_count() or 2)
torch.set_grad_enabled(False)

TASKS = {
    "dermamnist": "Skin lesion (dermoscopy)",
    "bloodmnist": "Blood cell (microscopy)",
    "pneumoniamnist": "Chest X-ray",
}

vit = VPTViT(2, num_prompts=0).vit.eval()  # downloads timm/vit_base_patch16_224.orig_in21k once
cfg = vit.pretrained_cfg


def fingerprint(module):
    """SHA-256 over every backbone weight, to show it is the same model for every task."""
    h = hashlib.sha256()
    for k, v in module.state_dict().items():
        h.update(k.encode()); h.update(v.detach().float().numpy().tobytes())
    return h.hexdigest()


BACKBONE_SHA = fingerprint(vit)
DEVICE = f"CPU, {torch.get_num_threads()} threads"  # the model always runs on CPU, as on a free Space
BACKBONE_PARAMS = sum(p.numel() for p in vit.parameters())
BACKBONE_MB = BACKBONE_PARAMS * 4 / 1e6

model = MultiTaskVPT(vit).eval()      # VPT-deep: every task's prompts on the one frozen backbone
LINEAR, FULL, META = {}, {}, {}


def info(f, meta, sha=None):
    return {"file": f.name, "file_kb": round(f.stat().st_size / 1024, 1), "test": meta["test"],
            "trainable": meta["trainable"], "train_images": meta["train_images"], "backbone_sha256": sha or BACKBONE_SHA}


for task in TASKS:
    f = WEIGHTS / f"{task}_vpt-deep-p50_full.safetensors"
    if not f.exists():
        f = WEIGHTS / f"{task}_vpt-deep-p50_1k.safetensors"
    if not f.exists():
        continue
    meta = json.loads(f.with_suffix(".json").read_text())
    n = len(meta["classes"])
    model.add_task(task, load_file(f), n)
    META[task] = {"name": TASKS[task], "classes": meta["classes"], "methods": {"vpt": info(f, meta)}}
    f = WEIGHTS / f"{task}_linear_full.safetensors"          # linear probe: a head on the same frozen backbone
    if f.exists():
        LINEAR[task] = VPTViT(n, num_prompts=0, vit=vit).eval()
        LINEAR[task].load_state_dict(load_file(f), strict=False)
        META[task]["methods"]["linear"] = info(f, json.loads(f.with_suffix(".json").read_text()))
    f = WEIGHTS / f"{task}_full_full.safetensors"            # full fine-tuning: a whole separate backbone
    if f.exists():
        FULL[task] = VPTViT(n, num_prompts=0, pretrained=False).eval()
        FULL[task].load_state_dict(load_file(f))
        META[task]["methods"]["full"] = info(f, json.loads(f.with_suffix(".json").read_text()), fingerprint(FULL[task].vit))
AVAILABLE = [t for t in TASKS if t in META]
TOKENS = {"linear": 197, "vpt": 247, "full": 197}

# MedMNIST+ images are 224 x 224 already; other uploads are resized and centre-cropped.
prep = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                           transforms.Normalize(cfg["mean"], cfg["std"])])


def tensor(image):
    return prep(image.convert("RGB"))


def probs_of(task, logits):
    return {c: float(p) for c, p in zip(META[task]["classes"], logits.softmax(-1))}


def describe(task, logits):
    """VPT result for one row of a (mixed) batch."""
    m = META[task]["methods"]["vpt"]
    return {"task": task, "name": TASKS[task], "probs": probs_of(task, logits),
            "prompt_file": m["file"], "prompt_kb": m["file_kb"], "test": m["test"], "train_images": m["train_images"]}


def common(batch):
    return {"batch": batch, "backbone_sha256": BACKBONE_SHA, "backbone_params": BACKBONE_PARAMS,
            "backbone_mb": round(BACKBONE_MB), "device": DEVICE}


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t0) * 1000


def classify(image: Image.Image, task: str):
    """Runs the image through all three adaptations of the same pretrained ViT for this task."""
    if image is None:
        raise gr.Error("Add an image first.")
    if task not in META:
        raise gr.Error(f"Unknown task '{task}'. Available: {', '.join(AVAILABLE)}.")
    x = tensor(image).unsqueeze(0)
    runs = {"vpt": lambda: model(x, [task])[0]}
    if task in LINEAR:
        runs["linear"] = lambda: LINEAR[task](x)[0]
    if task in FULL:
        runs["full"] = lambda: FULL[task](x)[0]
    methods = {}
    for name in ("linear", "vpt", "full"):
        if name in runs:
            logits, ms = timed(runs[name])
            methods[name] = {**META[task]["methods"][name], "probs": probs_of(task, logits), "ms": round(ms, 1), "tokens": TOKENS[name]}
    out = {"task": task, "name": TASKS[task], "methods": methods, **common(1)}
    labels = [methods[k]["probs"] if k in methods else None for k in ("linear", "vpt", "full")]
    return (*labels, out)


def mixed(derma: Image.Image, blood: Image.Image, xray: Image.Image):
    pairs = [(t, im) for t, im in zip(TASKS, (derma, blood, xray)) if im is not None and t in META]
    if not pairs:
        raise gr.Error("Add at least one image.")
    x = torch.stack([tensor(im) for _, im in pairs])
    logits, ms = timed(lambda: model(x, [t for t, _ in pairs]))   # one forward pass, a different task per row
    return {"results": [describe(t, l) for (t, _), l in zip(pairs, logits)], "ms": round(ms, 1), "tokens": TOKENS["vpt"], **common(len(pairs))}


def examples_for(task):
    return sorted(str(p) for p in (ROOT / "examples" / task).glob("*.png"))


with gr.Blocks(title="Visual Prompt Tuning: one frozen ViT, three medical tasks") as demo:
    gr.Markdown(
        "### Visual Prompt Tuning: one frozen ViT-B/16, three medical tasks\n"
        f"The backbone ({BACKBONE_PARAMS / 1e6:.1f}M parameters, sha256 `{BACKBONE_SHA[:12]}…`) is loaded once and never changes. "
        "Choosing a task only switches which ~2 MB prompt file (12 layers × 50 tokens) and head are used. "
        "Data: MedMNIST+ 224 × 224. **Research demo, not a diagnostic tool.**")
    with gr.Tab("One image, three methods"):
        with gr.Row():
            with gr.Column(scale=1):
                img = gr.Image(type="pil", label="Image", height=300)
                task = gr.Radio([(TASKS[t], t) for t in AVAILABLE], value=AVAILABLE[0] if AVAILABLE else None, label="Task")
                btn = gr.Button("Classify", variant="primary")
            with gr.Column(scale=2):
                with gr.Row():
                    lab_lin = gr.Label(num_top_classes=3, label="Linear probe (head only)")
                    lab_vpt = gr.Label(num_top_classes=3, label="VPT-deep (prompts + head)")
                    lab_full = gr.Label(num_top_classes=3, label="Full fine-tuning (every weight)")
                raw = gr.JSON(label="Raw output")
        btn.click(classify, [img, task], [lab_lin, lab_vpt, lab_full, raw], api_name="classify")
        ex = [[p, t] for t in AVAILABLE for p in examples_for(t)[:2]]
        if ex:
            gr.Examples(ex, [img, task])
    with gr.Tab("Mixed batch"):
        with gr.Row():
            a, b, c = (gr.Image(type="pil", label=TASKS[t], height=220) for t in TASKS)
        out_mixed = gr.JSON(label="One forward pass, one task per image")
        gr.Button("Run all in one forward pass", variant="primary").click(mixed, [a, b, c], out_mixed, api_name="mixed")

if __name__ == "__main__":
    demo.queue(max_size=16).launch(show_error=True)
