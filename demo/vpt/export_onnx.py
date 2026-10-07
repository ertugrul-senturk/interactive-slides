"""Exports the demo models to ONNX for in-browser inference (onnxruntime-web), and re-scores them.

Free Hugging Face Spaces no longer run Gradio, so the deck runs the models itself. Layout of web/:
  encoder.fp16.onnx        the frozen base model, ViT-B/16 ImageNet-21k. Inputs: image (B,3,224,224)
                           and prompts (B,12,P,768); output: the final [CLS] feature (B,768).
                           P = 0 gives the plain backbone (linear probe), P = 50 gives VPT-deep.
  full_<task>.fp16.onnx    the same graph with fully fine-tuned weights (fed P = 0).
  <task>_vpt_prompts.bin   float32 prompts, 12 x 50 x 768.
  heads.json               every linear head (linear probe, VPT, full) and class names, plus metrics.
Weights are stored in fp16 (172 MB per model instead of 343 MB). int8 quantization was smaller but changed
about 5% of predictions (quant_check.py); fp16 agrees with PyTorch on every test image checked.
Every exported model is checked against PyTorch and re-scored on the whole test split, so the deck
can report the accuracy of exactly what runs in the browser.

  python export_onnx.py            # export, check, evaluate
"""
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16
from safetensors.torch import load_file

from vpt_model import VPTViT

ROOT = Path(__file__).parent
W = ROOT / "weights"
OUT = ROOT / "web"
TASKS = ("dermamnist", "bloodmnist", "pneumoniamnist")


class Encoder(nn.Module):
    """ViT-B/16 whose every layer takes a (possibly empty) set of prompts; returns the final [CLS] feature."""

    def __init__(self, vit):
        super().__init__()
        self.vit = vit

    def forward(self, image, prompts):                 # prompts: (B, N_layers, P, d)
        v = self.vit
        p = prompts.shape[2]
        x = v.norm_pre(v._pos_embed(v.patch_embed(image)))
        for i, blk in enumerate(v.blocks):
            rest = x[:, 1:] if i == 0 else x[:, 1 + p:]
            x = blk(torch.cat([x[:, :1], prompts[:, i], rest], dim=1))
        return v.norm(x)[:, 0]


def export(vit, path):
    enc = Encoder(vit).eval()
    img, pr = torch.randn(2, 3, 224, 224), torch.randn(2, 12, 50, 768) * 0.02
    fp32 = path.with_suffix(".fp32.onnx")
    torch.onnx.export(enc, (img, pr), fp32, input_names=["image", "prompts"], output_names=["cls"], opset_version=17,
                      dynamic_axes={"image": {0: "B"}, "prompts": {0: "B", 2: "P"}, "cls": {0: "B"}}, dynamo=False)
    onnx.save(convert_float_to_float16(onnx.load(str(fp32)), keep_io_types=True), str(path))
    fp32.unlink()
    # numerical check against PyTorch, with and without prompts
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        for P in (0, 50):
            pr = torch.randn(2, 12, P, 768) * 0.02
            ref = enc(img, pr).numpy()
            got = sess.run(None, {"image": img.numpy(), "prompts": pr.numpy()})[0]
            cos = (ref * got).sum(1) / np.linalg.norm(ref, axis=1) / np.linalg.norm(got, axis=1)
            print(f"  {path.name}  P={P}: cosine to PyTorch fp32 {cos.min():.5f}", flush=True)
    return sess


def load_head(state):
    return state["head.weight"].numpy().astype(np.float32), state["head.bias"].numpy().astype(np.float32)


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
    x = np.load(ROOT / "data" / f"{task}_224" / "test_images.npy", mmap_mode="r")
    y = np.load(ROOT / "data" / f"{task}_224" / "test_labels.npy")
    for i in range(0, len(y), bs):
        a = np.asarray(x[i:i + bs])
        if a.ndim == 3:                                  # grayscale X-rays -> RGB
            a = np.repeat(a[..., None], 3, axis=-1)
        a = (a.astype(np.float32) / 255.0 - 0.5) / 0.5   # ViT-B/16 in21k: mean = std = 0.5
        yield a.transpose(0, 3, 1, 2).copy(), y[i:i + bs]


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_grad_enabled(False)
    base = VPTViT(2, num_prompts=0).vit.eval()
    print("base model", flush=True)
    sessions = {"base": export(base, OUT / "encoder.fp16.onnx")}
    heads = {"backbone": "timm/vit_base_patch16_224.orig_in21k", "mean": [0.5] * 3, "std": [0.5] * 3, "tasks": {}}
    for task in TASKS:
        print(task, flush=True)
        t = {"methods": {}}
        for method, tag in (("linear", "linear"), ("vpt", "vpt-deep-p50"), ("full", "full")):
            meta = json.loads((W / f"{task}_{tag}_full.json").read_text())
            state = load_file(W / f"{task}_{tag}_full.safetensors")
            t["classes"] = meta["classes"]
            w, b = load_head(state)
            entry = {"head_w": w.round(7).tolist(), "head_b": b.round(7).tolist(), "train_images": meta["train_images"],
                     "test_torch": meta["test"], "trainable": meta["trainable"]}
            if method == "vpt":
                pr = state["prompts"].numpy().astype(np.float32)
                (OUT / f"{task}_vpt_prompts.bin").write_bytes(pr.tobytes())
                entry["prompts"] = f"{task}_vpt_prompts.bin"
                entry["prompts_shape"] = list(pr.shape)
                entry["file_bytes"] = (W / f"{task}_{tag}_full.safetensors").stat().st_size
            elif method == "full":
                m = VPTViT(len(meta["classes"]), num_prompts=0, pretrained=False)
                m.load_state_dict(state)
                path = OUT / f"full_{task}.fp16.onnx"
                sessions[task] = export(m.vit.eval(), path)
                entry["model"] = path.name
                entry["file_bytes"] = (W / f"{task}_{tag}_full.safetensors").stat().st_size
                entry["web_bytes"] = path.stat().st_size
            else:
                entry["file_bytes"] = (W / f"{task}_{tag}_full.safetensors").stat().st_size
            t["methods"][method] = entry
        # re-score exactly what the browser runs
        n = len(t["classes"])
        probs = {k: [] for k in t["methods"]}
        ys = []
        pr = np.fromfile(OUT / t["methods"]["vpt"]["prompts"], dtype=np.float32).reshape(t["methods"]["vpt"]["prompts_shape"])
        t0 = time.time()
        for xb, yb in test_batches(task):
            B = len(yb)
            empty = np.zeros((B, 12, 0, 768), np.float32)
            f_base = sessions["base"].run(None, {"image": xb, "prompts": empty})[0]
            f_vpt = sessions["base"].run(None, {"image": xb, "prompts": np.broadcast_to(pr, (B, *pr.shape)).copy()})[0]
            f_full = sessions[task].run(None, {"image": xb, "prompts": empty})[0]
            for k, f in (("linear", f_base), ("vpt", f_vpt), ("full", f_full)):
                e = t["methods"][k]
                probs[k].append(softmax_np(f @ np.array(e["head_w"], np.float32).T + np.array(e["head_b"], np.float32)))
            ys.append(yb)
        y = np.concatenate(ys)
        for k in probs:
            t["methods"][k]["test_web"] = score(np.concatenate(probs[k]), y, n)
            print(f"  {k:6s} torch {t['methods'][k]['test_torch']}  fp16 onnx {t['methods'][k]['test_web']}", flush=True)
        print(f"  scored {len(y)} test images in {time.time() - t0:.0f}s", flush=True)
        heads["tasks"][task] = t
    heads["encoder"] = "encoder.fp16.onnx"
    heads["encoder_bytes"] = (OUT / "encoder.fp16.onnx").stat().st_size
    (OUT / "heads.json").write_text(json.dumps(heads))
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
