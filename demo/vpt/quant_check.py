"""Compares ONNX weight formats for the browser demo: how often each agrees with the PyTorch model.

Checks the shared base model with the DermaMNIST linear head (P = 0) and VPT prompts (P = 50) on the
first 512 test images, and reports file size, mean cosine of the [CLS] features, and prediction agreement.
"""
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.transformers.float16 import convert_float_to_float16
from safetensors.torch import load_file

from export_onnx import Encoder
from vpt_model import VPTViT

ROOT = Path(__file__).parent
TMP = ROOT / "web" / "tmp"
TMP.mkdir(parents=True, exist_ok=True)
torch.set_grad_enabled(False)

vit = VPTViT(2, num_prompts=0).vit.eval()
enc = Encoder(vit).eval()
fp32 = TMP / "enc.fp32.onnx"
if not fp32.exists():
    torch.onnx.export(enc, (torch.randn(2, 3, 224, 224), torch.randn(2, 12, 50, 768)), fp32, input_names=["image", "prompts"],
                      output_names=["cls"], opset_version=17, dynamic_axes={"image": {0: "B"}, "prompts": {0: "B", 2: "P"}, "cls": {0: "B"}}, dynamo=False)
variants = {"fp32": fp32}
quantize_dynamic(fp32, TMP / "enc.int8_pc.onnx", weight_type=QuantType.QInt8, per_channel=True); variants["int8 per-channel"] = TMP / "enc.int8_pc.onnx"
m16 = convert_float_to_float16(onnx.load(str(fp32)), keep_io_types=True); onnx.save(m16, str(TMP / "enc.fp16.onnx")); variants["fp16"] = TMP / "enc.fp16.onnx"

x = np.load(ROOT / "data" / "dermamnist_224" / "test_images.npy", mmap_mode="r")[:512]
xb = ((np.asarray(x).astype(np.float32) / 255 - 0.5) / 0.5).transpose(0, 3, 1, 2).copy()
lin, vpt = load_file(ROOT / "weights" / "dermamnist_linear_full.safetensors"), load_file(ROOT / "weights" / "dermamnist_vpt-deep-p50_full.safetensors")
P = vpt["prompts"].numpy()


def feats(sess, prompts):
    out = []
    for i in range(0, len(xb), 64):
        B = len(xb[i:i + 64])
        pr = np.zeros((B, 12, 0, 768), np.float32) if prompts is None else np.broadcast_to(prompts, (B, *prompts.shape)).copy()
        out.append(sess.run(None, {"image": xb[i:i + 64], "prompts": pr})[0])
    return np.concatenate(out)


ref = {}
with torch.no_grad():
    for k, pr, st in (("linear", None, lin), ("vpt", P, vpt)):
        f = torch.cat([enc(torch.from_numpy(xb[i:i + 64]), torch.zeros(len(xb[i:i + 64]), 12, 0, 768) if pr is None else torch.from_numpy(pr).expand(len(xb[i:i + 64]), -1, -1, -1)) for i in range(0, len(xb), 64)]).numpy()
        ref[k] = (f, (f @ st["head.weight"].numpy().T + st["head.bias"].numpy()).argmax(1))
for name, path in variants.items():
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    row = [f"{name:18s} {path.stat().st_size / 1e6:6.0f} MB"]
    for k, pr, st in (("linear", None, lin), ("vpt", P, vpt)):
        f = feats(s, pr)
        cos = ((f * ref[k][0]).sum(1) / np.linalg.norm(f, axis=1) / np.linalg.norm(ref[k][0], axis=1)).mean()
        agree = ((f @ st["head.weight"].numpy().T + st["head.bias"].numpy()).argmax(1) == ref[k][1]).mean()
        row.append(f"{k}: cos {cos:.5f} agree {100 * agree:5.1f}%")
    print("  ".join(row), flush=True)
