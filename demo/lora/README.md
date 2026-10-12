# LoRA demo

Live inference and training code for a talk on *LoRA: Low-Rank Adaptation of Large Language Models*
(Hu et al., ICLR 2022), applied to a vision transformer. It is the companion of `../vpt` (the Visual Prompt
Tuning talk) and set up the same way, so the two methods can be compared directly:

* the same base model: ViT-B/16 pretrained on ImageNet-21k (`timm/vit_base_patch16_224.orig_in21k`, 85.8M parameters);
* the same three MedMNIST+ tasks at 224 × 224, with the same splits and the same random 1,000-image subsets:

  | task | images | classes |
  | --- | --- | --- |
  | DermaMNIST | dermoscopy (HAM10000) | 7 |
  | BloodMNIST | blood-cell microscopy | 8 |
  | PneumoniaMNIST | chest X-ray | 2 |

* the same baselines: linear probing and full fine-tuning are **not retrained**. They are the VPT talk's runs
  (`../vpt/weights`), and the demo loads the same fully fine-tuned browser models from
  [valinor61/vpt-medmnist](https://huggingface.co/valinor61/vpt-medmnist).

LoRA follows the paper's default: a rank-8 pair B A beside the query and the value weight of every attention
block (α = r = 8, B initialised to zero), plus a linear head. That is 294,912 numbers in the pairs, 0.35% of the
model with a 7-class head, a 1.2 MB file per task. Training uses AdamW; everything else (augmentation, warm-up and
cosine schedule, learning rate picked on val with a sweep on the 1,000-image subset, model selection on val
balanced accuracy, test once) is the VPT talk's recipe. **Research demo, not a diagnostic tool.**

## How the live demo runs

As in the VPT talk, the slide deck runs the models in the viewer's browser with onnxruntime-web:

* `export_onnx.py` exports one fp16 ONNX graph of the frozen base model, `peft_encoder.fp16.onnx`, whose task files
  are *inputs*: `prompts` (B,12,P,768) as in VPT-deep, and `lora_a` (B,12,2,R,768), `lora_b` (B,12,2,768,R) with α / r
  folded into B. Nothing fed in gives the plain backbone (linear probe), prompts give VPT-deep, pairs give LoRA.
  This is LoRA unmerged, on purpose: one 172 MB download serves every task and three methods, and switching task swaps
  a 1.2 MB file. Merged, it would be one model per task. The script re-scores every model on the whole test sets, and
  checks that linear probe and VPT-deep score the same on this graph as on the VPT graph.
* `upload_models.py` publishes `web/` to the free model repository
  [valinor61/lora-medmnist](https://huggingface.co/valinor61/lora-medmnist). `heads.json` points the VPT prompt files
  and the fully fine-tuned models to the VPT repository, so they are not stored twice. The deck keeps downloads in
  the browser cache, shared with the VPT deck.
* The same files serve the combined VPT + LoRA deck (`public/decks/2026-10-peft-vpt-lora.html`), whose demo slide
  shows all four methods side by side.
* The demo slide switches every method to the models trained on 1,000 images (`heads.json` → `methods_1k`).
  Try a local export with a static server that sends `Access-Control-Allow-Origin: *` and open the deck with
  `?models=http://localhost:8765/`.
* `app.py` is the same demo as a Gradio server (`/classify`, `/mixed`). Run it locally and open the deck with
  `?demo=http://127.0.0.1:7860/` as a fallback.

## Notebooks

`notebooks/` walks through the same code with its outputs, in order: the LoRA model, its parameter counts, the
zero start and merging (01), the training loop with a short live run (02), every result and the cost benchmark
(03), and the browser export check (04). The data is shown in `../vpt/notebooks/02_data_medmnist.ipynb`.
`python notebooks/make_notebooks.py` regenerates them, `python notebooks/run_all.py` re-executes them.

## Files

| file | what |
| --- | --- |
| `lora_model.py` | LoRA on a timm ViT (`LoRAViT`, rank 0 = linear probe), merging, and the multi-task model |
| `train.py` | AdamW training with the VPT talk's data and schedule (imported from `../vpt/train.py`) |
| `run_lora.sh` | the six training runs, in the order they were made |
| `evaluate.py` | re-scores every saved LoRA model on its test split, with the bypass and merged |
| `bench.py` | training memory and speed, CPU inference time (LoRA with the bypass and merged) |
| `export_onnx.py`, `upload_models.py` | browser models |
| `app.py` | Gradio server with `/classify` and `/mixed` |
| `fill_deck.py` | every result number for the deck (`const RESULTS`) |

## Reproduce (Windows, CUDA GPU)

Uses the virtual environment and the data of `../vpt` (see its README).

    sh run_lora.sh
    ../vpt/.venv/Scripts/python evaluate.py
    ../vpt/.venv/Scripts/python bench.py
    ../vpt/.venv/Scripts/python fill_deck.py
    ../vpt/.venv/Scripts/python export_onnx.py
    ../vpt/.venv/Scripts/python upload_models.py      # after `hf auth login`
