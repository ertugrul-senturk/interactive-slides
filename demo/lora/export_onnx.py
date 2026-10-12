"""Exports the LoRA demo models to ONNX for in-browser inference (onnxruntime-web), and re-scores them.

The deck runs the models itself, as in the VPT demo. Layout of web/:
  peft_encoder.fp16.onnx   the frozen base model, ViT-B/16 ImageNet-21k, that takes both kinds of task file as inputs:
                           image (B,3,224,224); prompts (B,12,P,768), inserted before every block as in VPT-deep; and
                           lora_a (B,12,2,R,768), lora_b (B,12,2,768,R), a LoRA bypass on W_q and W_v of every block,
                           with alpha / r already folded into B. Output: the final [CLS] feature (B,768).
                           P = 0 and all-zero pairs: the plain backbone (linear probe). Prompts only: VPT-deep.
                           Pairs only: LoRA. So one 172 MB download serves three of the four methods, in the LoRA deck
                           and in the combined VPT + LoRA deck.
  <task>_lora.bin          float32 pairs of one task, A (12,2,8,768) followed by B (12,2,768,8) (<task>_lora_1k.bin:
                           trained on 1,000 images). 1.2 MB.
  heads.json               every linear head (linear probe, VPT-deep, LoRA, full) and class names, plus metrics; the
                           full-data models under "methods", the 1,000-image ones under "methods_1k".
Linear probing, VPT-deep and full fine-tuning are the same models as in the VPT demo: their heads and scores are copied
from ../vpt/web/heads.json, and the prompt files and fully fine-tuned models are loaded from the VPT model repository
(VPT_REPO), so nothing is stored twice. Linear probe and VPT-deep are re-scored here on the new graph and must match
their scores on the VPT graph. Weights are stored in fp16 (172 MB). Every exported model is checked against PyTorch and
re-scored on the whole test split, so the deck can report the accuracy of exactly what runs in the browser.

  python export_onnx.py                  # export, check, evaluate both regimes (full data and 1,000 images)
  python export_onnx.py --regimes 1k     # only the 1,000-image models, keeping the rest of heads.json
"""
import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxruntime.transformers.float16 import convert_float_to_float16
from safetensors.torch import load_file

from lora_model import LoRAViT, load_task

ROOT = Path(__file__).parent
VPT = ROOT.parent / "vpt"
W = ROOT / "weights"
OUT = ROOT / "web"
TASKS = ("dermamnist", "bloodmnist", "pneumoniamnist")
TAG = "lora-r8"
VPT_REPO = "https://huggingface.co/valinor61/vpt-medmnist/resolve/main/"   # prompt files and fully fine-tuned models of the VPT demo


class Encoder(nn.Module):
    """ViT-B/16 whose every block takes (possibly empty) VPT prompts and (possibly zero) LoRA pairs per image;
    returns the final [CLS] feature. The prompt handling is that of ../vpt/export_onnx.py."""

    def __init__(self, vit):
        super().__init__()
        self.vit = vit

    def forward(self, image, prompts, lora_a, lora_b):  # prompts (B,12,P,d); lora_a (B,12,2,R,d); lora_b (B,12,2,d,R)
        v = self.vit
        p = prompts.shape[2]
        x = v.norm_pre(v._pos_embed(v.patch_embed(image)))
        try:
            for i, blk in enumerate(v.blocks):
                blk.attn.qkv.ext = (lora_a[:, i], lora_b[:, i])
                rest = x[:, 1:] if i == 0 else x[:, 1 + p:]
                x = blk(torch.cat([x[:, :1], prompts[:, i], rest], dim=1))
        finally:
            for blk in v.blocks:
                blk.attn.qkv.ext = None
        return v.norm(x)[:, 0]


def zeros(B, R=1):
    return np.zeros((B, 12, 2, R, 768), np.float32), np.zeros((B, 12, 2, 768, R), np.float32)


def no_prompts(B):
    return np.zeros((B, 12, 0, 768), np.float32)


def export(vit, path):
    enc = Encoder(vit).eval()
    img = torch.randn(2, 3, 224, 224)
    pr = torch.randn(2, 12, 50, 768) * 0.02
    a, b = torch.randn(2, 12, 2, 8, 768) * 0.02, torch.randn(2, 12, 2, 768, 8) * 0.02
    fp32 = path.with_suffix(".fp32.onnx")
    torch.onnx.export(enc, (img, pr, a, b), fp32, input_names=["image", "prompts", "lora_a", "lora_b"], output_names=["cls"],
                      opset_version=17, dynamo=False,
                      dynamic_axes={"image": {0: "B"}, "prompts": {0: "B", 2: "P"}, "lora_a": {0: "B", 3: "R"},
                                    "lora_b": {0: "B", 4: "R"}, "cls": {0: "B"}})
    onnx.save(convert_float_to_float16(onnx.load(str(fp32)), keep_io_types=True), str(path))
    fp32.unlink()
    # numerical check against PyTorch: plain backbone, prompts only, pairs only (two ranks)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        for P, R, s in ((0, 1, 0.0), (50, 1, 0.0), (0, 8, 0.05), (0, 4, 0.05)):
            pr = torch.randn(2, 12, P, 768) * 0.02
            a, b = torch.randn(2, 12, 2, R, 768) * s, torch.randn(2, 12, 2, 768, R) * s
            ref = enc(img, pr, a, b).numpy()
            got = sess.run(None, {"image": img.numpy(), "prompts": pr.numpy(), "lora_a": a.numpy(), "lora_b": b.numpy()})[0]
            cos = (ref * got).sum(1) / np.linalg.norm(ref, axis=1) / np.linalg.norm(got, axis=1)
            print(f"  {path.name}  P={P} R={R} scale={s}: cosine to PyTorch fp32 {cos.min():.5f}", flush=True)
    return sess


def softmax_np(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def score(probs, y, n):
    from sklearn.metrics import roc_auc_score
    pred = probs.argmax(1)
    rec = [(pred[y == c] == c).mean() for c in range(n) if (y == c).any()]
    auc = roc_auc_score(y, probs[:, 1]) if n == 2 else roc_auc_score(y, probs, multi_class="ovr", average="macro")
    return {"acc": round(100 * float((pred == y).mean()), 2), "bal_acc": round(100 * float(np.mean(rec)), 2), "auc": round(100 * float(auc), 2)}


def test_batches(task, bs=64):
    x = np.load(VPT / "data" / f"{task}_224" / "test_images.npy", mmap_mode="r")
    y = np.load(VPT / "data" / f"{task}_224" / "test_labels.npy")
    for i in range(0, len(y), bs):
        a = np.asarray(x[i:i + bs])
        if a.ndim == 3:                                  # grayscale X-rays -> RGB
            a = np.repeat(a[..., None], 3, axis=-1)
        a = (a.astype(np.float32) / 255.0 - 0.5) / 0.5   # ViT-B/16 in21k: mean = std = 0.5
        yield a.transpose(0, 3, 1, 2).copy(), y[i:i + bs]


# Two training regimes per task: "full" (all training images) and "1k" (1,000 images, as in VTAB-1k).
REGIMES = {"full": ("methods", ""), "1k": ("methods_1k", "_1k")}


def main(regimes=("full", "1k")):
    OUT.mkdir(exist_ok=True)
    torch.set_grad_enabled(False)
    vpt_heads = json.loads((VPT / "web" / "heads.json").read_text())
    hj = OUT / "heads.json"
    heads = json.loads(hj.read_text()) if hj.exists() else {}
    heads.update({"backbone": "timm/vit_base_patch16_224.orig_in21k", "mean": [0.5] * 3, "std": [0.5] * 3,
                  "lora": {"rank": 8, "targets": ["query", "value"], "blocks": 12}})
    heads.setdefault("tasks", {})
    enc_path = OUT / "peft_encoder.fp16.onnx"
    if "full" in regimes or not enc_path.exists():
        print("base model with prompt and LoRA inputs", flush=True)
        sess = export(LoRAViT(2, rank=0).vit.eval(), enc_path)
    else:                                                # the frozen base model is the same for every regime
        sess = ort.InferenceSession(str(enc_path), providers=["CPUExecutionProvider"])
    for task in TASKS:
        t = heads["tasks"].setdefault(task, {})
        for regime in regimes:
            key, sfx = REGIMES[regime]
            print(task, regime, flush=True)
            vm = vpt_heads["tasks"][task][key]
            t["classes"] = vpt_heads["tasks"][task]["classes"]
            src = W / f"{task}_{TAG}_{regime}"
            meta = json.loads(src.with_suffix(".json").read_text())
            assert meta["classes"] == t["classes"]
            state = load_file(src.with_suffix(".safetensors"))
            model = load_task(state, len(meta["classes"]), pretrained=False)
            A, B = (z.detach().numpy().astype(np.float32) for z in model.stacked())
            name = f"{task}_lora{sfx}.bin"
            (OUT / name).write_bytes(A.tobytes() + B.tobytes())
            methods = {
                "linear": dict(vm["linear"]),
                "vpt": {**vm["vpt"], "prompts": VPT_REPO + vm["vpt"]["prompts"]},
                "lora": {"head_w": state["head.weight"].numpy().round(7).tolist(), "head_b": state["head.bias"].numpy().round(7).tolist(),
                         "train_images": meta["train_images"], "test_torch": meta["test"], "trainable": meta["trainable"],
                         "file_bytes": src.with_suffix(".safetensors").stat().st_size, "lora": name, "lora_shape": list(A.shape)},
                "full": {**vm["full"], "model": VPT_REPO + vm["full"]["model"]},
            }
            # re-score exactly what the browser runs: linear probe, VPT-deep and LoRA on this graph
            n = len(t["classes"])
            P = np.fromfile(VPT / "web" / vm["vpt"]["prompts"], dtype=np.float32).reshape(vm["vpt"]["prompts_shape"])
            probs, ys = {"linear": [], "vpt": [], "lora": []}, []
            t0 = time.time()
            for xb, yb in test_batches(task):
                Bn = len(yb)
                za, zb = zeros(Bn)
                feeds = {
                    "linear": {"image": xb, "prompts": no_prompts(Bn), "lora_a": za, "lora_b": zb},
                    "vpt": {"image": xb, "prompts": np.broadcast_to(P, (Bn, *P.shape)).copy(), "lora_a": za, "lora_b": zb},
                    "lora": {"image": xb, "prompts": no_prompts(Bn), "lora_a": np.broadcast_to(A, (Bn, *A.shape)).copy(),
                             "lora_b": np.broadcast_to(B, (Bn, *B.shape)).copy()},
                }
                for k, fd in feeds.items():
                    f = sess.run(None, fd)[0]
                    e = methods[k]
                    probs[k].append(softmax_np(f @ np.array(e["head_w"], np.float32).T + np.array(e["head_b"], np.float32)))
                ys.append(yb)
            y = np.concatenate(ys)
            for k in ("linear", "vpt"):                       # must reproduce their scores on the VPT graph
                got = score(np.concatenate(probs[k]), y, n)
                assert abs(got["bal_acc"] - vm[k]["test_web"]["bal_acc"]) < 0.2, (k, got, vm[k]["test_web"])
                print(f"  {k:6s}  VPT graph {vm[k]['test_web']}  this graph {got}", flush=True)
            methods["lora"]["test_web"] = score(np.concatenate(probs["lora"]), y, n)
            print(f"  lora    torch {methods['lora']['test_torch']}  fp16 onnx {methods['lora']['test_web']}", flush=True)
            print(f"  scored {len(y)} test images in {time.time() - t0:.0f}s", flush=True)
            t[key] = methods
            hj.write_text(json.dumps(heads))                 # saved after every regime, so a crash loses little
    heads["encoder"] = enc_path.name
    heads["encoder_bytes"] = enc_path.stat().st_size
    hj.write_text(json.dumps(heads))
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", default="full,1k", help="comma-separated subset of: full, 1k")
    main(tuple(ap.parse_args().regimes.split(",")))
