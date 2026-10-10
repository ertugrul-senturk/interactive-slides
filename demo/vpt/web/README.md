---
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
| `full_<task>_1k.fp16.onnx` | the same, fine-tuned on 1,000 training images |
| `<task>_vpt_prompts.bin` | VPT-deep prompts, float32, 12 × 50 × 768 |
| `<task>_vpt_prompts_1k.bin` | the same, trained on 1,000 training images |
| `heads.json` | every classification head, class names, and test metrics; full-data models under `methods`, 1,000-image models under `methods_1k` |

Weights are stored in fp16. Every exported model was re-scored on the whole test split; the table gives the PyTorch
and the fp16 ONNX balanced accuracy side by side.

## Test results (balanced accuracy = mean per-class recall)

| task | method | training images | accuracy | balanced acc. (PyTorch) | balanced acc. (this ONNX) | stored per task |
| --- | --- | --- | --- | --- | --- | --- |
| dermamnist | linear | 1,000 | 75.1 | 58.4 | 58.1 | 0.02 MB |
| dermamnist | vpt | 1,000 | 79.4 | 63.3 | 62.5 | 1.86 MB |
| dermamnist | full | 1,000 | 78.8 | 64.9 | 64.8 | 343.23 MB |
| dermamnist | linear | 7,007 | 83.0 | 68.1 | 68.5 | 0.02 MB |
| dermamnist | vpt | 7,007 | 88.3 | 81.0 | 81.0 | 1.86 MB |
| dermamnist | full | 7,007 | 91.2 | 84.4 | 84.4 | 343.23 MB |
| bloodmnist | linear | 1,000 | 96.0 | 96.1 | 96.2 | 0.02 MB |
| bloodmnist | vpt | 1,000 | 98.1 | 98.2 | 98.2 | 1.87 MB |
| bloodmnist | full | 1,000 | 97.6 | 98.0 | 98.0 | 343.23 MB |
| bloodmnist | linear | 11,959 | 97.4 | 97.3 | 97.4 | 0.02 MB |
| bloodmnist | vpt | 11,959 | 99.1 | 99.1 | 99.1 | 1.87 MB |
| bloodmnist | full | 11,959 | 99.1 | 99.2 | 99.2 | 343.23 MB |
| pneumoniamnist | linear | 1,000 | 91.3 | 89.2 | 89.2 | 0.01 MB |
| pneumoniamnist | vpt | 1,000 | 91.8 | 89.7 | 89.7 | 1.85 MB |
| pneumoniamnist | full | 1,000 | 94.4 | 93.4 | 93.4 | 343.22 MB |
| pneumoniamnist | linear | 4,708 | 92.8 | 90.9 | 90.9 | 0.01 MB |
| pneumoniamnist | vpt | 4,708 | 95.3 | 94.1 | 94.1 | 1.85 MB |
| pneumoniamnist | full | 4,708 | 94.9 | 93.2 | 93.2 | 343.22 MB |

Data: DermaMNIST (from HAM10000, CC BY-NC 4.0), BloodMNIST and PneumoniaMNIST, MedMNIST v2
(Yang et al., Scientific Data 2023). Training code: the `demo/vpt` folder of the talk's slide repository.
