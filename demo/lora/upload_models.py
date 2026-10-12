"""Publishes the LoRA browser models (web/, from export_onnx.py) to a Hugging Face model repository.

  python upload_models.py                  # -> https://huggingface.co/<you>/lora-medmnist

Model repositories are free (unlike Gradio Spaces), and their files can be fetched from any web page, which is
how the slide deck loads them. The fully fine-tuned baselines are not uploaded again: heads.json points to them
in the VPT repository. Uses the token from `hf auth login`.
"""
import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).parent
WEB = ROOT / "web"

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="lora-medmnist")
args = ap.parse_args()

heads = json.loads((WEB / "heads.json").read_text())
rows = []
for task, t in heads["tasks"].items():
    for key in ("methods_1k", "methods"):                 # 1,000 training images, then the full training set
        for m in ("linear", "vpt", "lora", "full"):
            if key not in t:
                continue
            e = t[key][m]
            rows.append(f"| {task} | {m} | {e['train_images']:,} | {e['test_torch']['acc']:.1f} | {e['test_torch']['bal_acc']:.1f} | "
                        f"{e['test_web']['bal_acc']:.1f} | {e['file_bytes'] / 1e6:.2f} MB |")
card = f"""---
license: cc-by-nc-4.0
library_name: onnx
tags: [vision-transformer, lora, low-rank-adaptation, medmnist, medical-imaging, onnx]
---

# LoRA on MedMNIST+: browser models

Models for the live demo in a talk on *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al., ICLR 2022),
applied to a vision transformer. They run in the browser with onnxruntime-web. **Research demo, not a diagnostic tool.**

All models start from the same base model, ViT-B/16 pretrained on ImageNet-21k
(`timm/vit_base_patch16_224.orig_in21k`), adapted to three MedMNIST+ 224 × 224 tasks in four ways:

* **linear probe**: base model frozen, a linear head trained
* **VPT-deep**: base model frozen, 50 prompt tokens before each of the 12 blocks + a head (the VPT repository's files)
* **LoRA**: base model frozen, rank-8 pairs B A beside the query and value weights of all 12 blocks + a head
* **full fine-tuning**: every weight trained, i.e. a separate model per task

This is the companion of [valinor61/vpt-medmnist](https://huggingface.co/valinor61/vpt-medmnist) (Visual Prompt
Tuning, same base model, same tasks, same data splits). Linear probe, VPT-deep and full fine-tuning are the very same
models; their prompt files and fully fine-tuned models are loaded from that repository and not stored here again.

| file | what |
| --- | --- |
| `peft_encoder.fp16.onnx` | the frozen base model with both kinds of task file as inputs: `image` (B,3,224,224) normalised with mean = std = 0.5; `prompts` (B,12,P,768), VPT-deep prompts before every block; `lora_a` (B,12,2,R,768) and `lora_b` (B,12,2,768,R), a LoRA bypass on W_q and W_v with alpha / r folded in, [query, value] on axis 2. Output `cls` (B,768). Nothing fed in gives the plain backbone (linear probe), prompts give VPT-deep, pairs give LoRA |
| `<task>_lora.bin` | one task's pairs, float32: A (12,2,8,768) followed by B (12,2,768,8) |
| `<task>_lora_1k.bin` | the same, trained on 1,000 training images |
| `heads.json` | every classification head, class names, and test metrics; full-data models under `methods`, 1,000-image models under `methods_1k` |

Weights are stored in fp16. Every exported model was re-scored on the whole test split; the table gives the PyTorch
and the fp16 ONNX balanced accuracy side by side. "Stored per task" is the trained part in 32-bit.

## Test results (balanced accuracy = mean per-class recall)

| task | method | training images | accuracy | balanced acc. (PyTorch) | balanced acc. (ONNX) | stored per task |
| --- | --- | --- | --- | --- | --- | --- |
""" + "\n".join(rows) + """

Data: DermaMNIST (from HAM10000, CC BY-NC 4.0), BloodMNIST and PneumoniaMNIST, MedMNIST v2
(Yang et al., Scientific Data 2023). Training code: the `demo/lora` folder of the talk's slide repository.
"""
(WEB / "README.md").write_text(card, encoding="utf8")

api = HfApi()
repo = f"{api.whoami()['name']}/{args.name}"
api.create_repo(repo, repo_type="model", exist_ok=True)
api.upload_folder(repo_id=repo, repo_type="model", folder_path=WEB, allow_patterns=["*.onnx", "*.bin", "heads.json", "README.md"],
                  commit_message="Browser models for the LoRA demo")
print(f"https://huggingface.co/{repo}")
print(f"files at https://huggingface.co/{repo}/resolve/main/")
