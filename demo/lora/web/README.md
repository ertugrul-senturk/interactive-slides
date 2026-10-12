---
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
| dermamnist | linear | 1,000 | 75.1 | 58.4 | 58.1 | 0.02 MB |
| dermamnist | vpt | 1,000 | 79.4 | 63.3 | 62.5 | 1.86 MB |
| dermamnist | lora | 1,000 | 82.0 | 63.0 | 63.0 | 1.20 MB |
| dermamnist | full | 1,000 | 78.8 | 64.9 | 64.8 | 343.23 MB |
| dermamnist | linear | 7,007 | 83.0 | 68.1 | 68.5 | 0.02 MB |
| dermamnist | vpt | 7,007 | 88.3 | 81.0 | 81.0 | 1.86 MB |
| dermamnist | lora | 7,007 | 90.7 | 82.6 | 82.5 | 1.20 MB |
| dermamnist | full | 7,007 | 91.2 | 84.4 | 84.4 | 343.23 MB |
| bloodmnist | linear | 1,000 | 96.0 | 96.1 | 96.2 | 0.02 MB |
| bloodmnist | vpt | 1,000 | 98.1 | 98.2 | 98.2 | 1.87 MB |
| bloodmnist | lora | 1,000 | 98.2 | 98.3 | 98.3 | 1.21 MB |
| bloodmnist | full | 1,000 | 97.6 | 98.0 | 98.0 | 343.23 MB |
| bloodmnist | linear | 11,959 | 97.4 | 97.3 | 97.4 | 0.02 MB |
| bloodmnist | vpt | 11,959 | 99.1 | 99.1 | 99.1 | 1.87 MB |
| bloodmnist | lora | 11,959 | 98.9 | 99.0 | 99.0 | 1.21 MB |
| bloodmnist | full | 11,959 | 99.1 | 99.2 | 99.2 | 343.23 MB |
| pneumoniamnist | linear | 1,000 | 91.3 | 89.2 | 89.2 | 0.01 MB |
| pneumoniamnist | vpt | 1,000 | 91.8 | 89.7 | 89.7 | 1.85 MB |
| pneumoniamnist | lora | 1,000 | 94.4 | 92.8 | 92.8 | 1.19 MB |
| pneumoniamnist | full | 1,000 | 94.4 | 93.4 | 93.4 | 343.22 MB |
| pneumoniamnist | linear | 4,708 | 92.8 | 90.9 | 90.9 | 0.01 MB |
| pneumoniamnist | vpt | 4,708 | 95.3 | 94.1 | 94.1 | 1.85 MB |
| pneumoniamnist | lora | 4,708 | 94.5 | 93.1 | 93.1 | 1.19 MB |
| pneumoniamnist | full | 4,708 | 94.9 | 93.2 | 93.2 | 343.22 MB |

Data: DermaMNIST (from HAM10000, CC BY-NC 4.0), BloodMNIST and PneumoniaMNIST, MedMNIST v2
(Yang et al., Scientific Data 2023). Training code: the `demo/lora` folder of the talk's slide repository.
