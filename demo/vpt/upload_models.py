"""Publishes the browser models (web/, from export_onnx.py) to a Hugging Face model repository.

  python upload_models.py                  # -> https://huggingface.co/<you>/vpt-medmnist

Model repositories are free (unlike Gradio Spaces), and their files can be fetched from any web
page, which is how the slide deck loads them. Uses the token from `hf auth login`.
"""
import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).parent
WEB = ROOT / "web"

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="vpt-medmnist")
args = ap.parse_args()

heads = json.loads((WEB / "heads.json").read_text())
rows = []
for task, t in heads["tasks"].items():
    for m in ("linear", "vpt", "full"):
        e = t["methods"][m]
        rows.append(f"| {task} | {m} | {e['train_images']:,} | {e['test_torch']['acc']:.1f} | {e['test_torch']['bal_acc']:.1f} | "
                    f"{e['test_web']['bal_acc']:.1f} | {e['file_bytes'] / 1e6:.2f} MB |")
card = f"""---
license: cc-by-nc-4.0
library_name: onnx
tags: [vision-transformer, visual-prompt-tuning, medmnist, medical-imaging, onnx]
---

# Visual Prompt Tuning on MedMNIST+: browser models

Models for the live demo in a talk on *Visual Prompt Tuning* (Jia et al., ECCV 2022). They run in the
browser with onnxruntime-web. **Research demo, not a diagnostic tool.**

All models start from the same base model, ViT-B/16 pretrained on ImageNet-21k
(`timm/vit_base_patch16_224.orig_in21k`, the checkpoint used in the VPT paper), adapted to three
MedMNIST+ 224 × 224 tasks in three ways:

* **linear probe**: base model frozen, a linear head trained
* **VPT-deep**: base model frozen, 50 prompt tokens before each of the 12 layers + a head
* **full fine-tuning**: every weight trained, i.e. a separate model per task

| file | what |
| --- | --- |
| `encoder.fp16.onnx` | the frozen base model; inputs `image` (B,3,224,224) normalised with mean = std = 0.5, and `prompts` (B,12,P,768); output `cls` (B,768). P = 0 for the linear probe, P = 50 for VPT-deep |
| `full_<task>.fp16.onnx` | the fully fine-tuned model of a task (same graph, fed P = 0) |
| `<task>_vpt_prompts.bin` | VPT-deep prompts, float32, 12 × 50 × 768 |
| `heads.json` | every classification head, class names, and test metrics |

Weights are stored in fp16; on the test images checked they give the same predictions as the PyTorch models.

## Test results (balanced accuracy = mean per-class recall)

| task | method | training images | accuracy | balanced acc. (PyTorch) | balanced acc. (this ONNX) | stored per task |
| --- | --- | --- | --- | --- | --- | --- |
""" + "\n".join(rows) + """

Data: DermaMNIST (from HAM10000, CC BY-NC 4.0), BloodMNIST and PneumoniaMNIST, MedMNIST v2
(Yang et al., Scientific Data 2023). Training code: the `demo/vpt` folder of the talk's slide repository.
"""
(WEB / "README.md").write_text(card, encoding="utf8")

api = HfApi()
repo = f"{api.whoami()['name']}/{args.name}"
api.create_repo(repo, repo_type="model", exist_ok=True)
api.upload_folder(repo_id=repo, repo_type="model", folder_path=WEB, allow_patterns=["*.onnx", "*.bin", "heads.json", "README.md"],
                  commit_message="Browser models for the VPT demo")
print(f"https://huggingface.co/{repo}")
print(f"files at https://huggingface.co/{repo}/resolve/main/")
