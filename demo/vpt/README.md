---
title: Visual Prompt Tuning, three medical tasks
sdk: gradio
sdk_version: 6.29.1
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: One frozen ViT-B/16, VPT prompts for skin, blood, X-ray
---

# Visual Prompt Tuning demo

Live inference for a talk on *Visual Prompt Tuning* (Jia et al., ECCV 2022).

One frozen ViT-B/16 (ImageNet-21k, the paper's checkpoint, 85.8M parameters) is loaded once.
Each task is a ~2 MB file of VPT-deep prompts (50 tokens before each of the 12 layers) plus a
linear head, trained on MedMNIST+ at 224 × 224:

| task | images | classes |
| --- | --- | --- |
| DermaMNIST | dermoscopy (HAM10000) | 7 |
| BloodMNIST | blood-cell microscopy | 8 |
| PneumoniaMNIST | chest X-ray | 2 |

Choosing a task swaps only the prompt file; the backbone's SHA-256 is shown to prove it never
changes. The *Mixed batch* tab runs one image per task through a single forward pass, each row
with its own prompts. **Research demo, not a diagnostic tool.**

## How the live demo runs

Free Hugging Face Spaces no longer run Gradio apps (they need a PRO plan), so the slide deck runs the
models itself, in the viewer's browser, with onnxruntime-web:

* `export_onnx.py` exports the base model and the three fully fine-tuned models to fp16 ONNX
  (int8 was smaller but changed about 5% of predictions, see `quant_check.py`) and re-scores every
  model on the full test sets.
* `upload_models.py` publishes them to the free model repository
  [valinor61/vpt-medmnist](https://huggingface.co/valinor61/vpt-medmnist); the deck downloads them
  from there once (about 350 MB for the first task) and keeps them in the browser cache.
* `app.py` is the same demo as a Gradio server. Run it locally and type its URL
  (http://127.0.0.1:7860/) into the demo slide's server field as a fallback.

## Files

| file | what |
| --- | --- |
| `vpt_model.py` | VPT-shallow / VPT-deep / linear probe on a timm ViT, and the multi-task model |
| `train.py` | the paper's recipe (SGD for VPT and linear, AdamW for full fine-tuning, LR picked on val) |
| `evaluate.py` | re-scores every saved model on its test split |
| `bench.py` | training memory and speed, CPU inference time, per method |
| `export_onnx.py`, `quant_check.py`, `upload_models.py` | browser models |
| `app.py` | Gradio server with `/classify` and `/mixed` |
| `export_samples.py`, `fill_deck.py` | random test images and every result number for the deck |
| `run_*.sh` | the training runs in the order they were made |

## Reproduce (Windows, CUDA GPU)

    python -m venv .venv
    .venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    .venv\Scripts\pip install timm gradio safetensors medmnist scikit-learn onnx onnxruntime onnxscript
    # data: dermamnist_224.npz, bloodmnist_224.npz, pneumoniamnist_224.npz from zenodo.org/records/10519652 into data/
    sh run_derma.sh; sh run_queue2.sh; sh run_full.sh; sh run_queue3.sh; sh run_queue4.sh
    .venv\Scripts\python evaluate.py
    .venv\Scripts\python bench.py
    .venv\Scripts\python fill_deck.py
    .venv\Scripts\python export_onnx.py
    .venv\Scripts\python upload_models.py      # after `hf auth login`
