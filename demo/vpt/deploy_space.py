"""Creates or updates a Hugging Face Space for the Gradio demo. Note: Gradio Spaces on the free
cpu-basic hardware now need a PRO plan; the deck runs the models in the browser instead (export_onnx.py).

  python deploy_space.py                # -> https://huggingface.co/spaces/<you>/vpt-demo
  python deploy_space.py --name other   # a different Space name

Uploads only what inference needs: the app, the model code, and per task the linear head (~21 KB),
the VPT-deep prompts (~1.8 MB) and the fully fine-tuned model (~343 MB), plus example images.
The shared pretrained backbone is not uploaded; the Space downloads it from timm's public repo
on start-up. Uses the token from `hf auth login`.
"""
import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).parent
TASKS = ("dermamnist", "bloodmnist", "pneumoniamnist")

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="vpt-demo")
args = ap.parse_args()

api = HfApi()
user = api.whoami()["name"]
repo = f"{user}/{args.name}"

stage = Path(tempfile.mkdtemp())
for f in ("app.py", "vpt_model.py", "requirements.txt", "README.md"):
    shutil.copy(ROOT / f, stage / f)
(stage / "weights").mkdir()
for t in TASKS:
    for method in ("vpt-deep-p50", "linear", "full"):
        for suffix in (".safetensors", ".json"):
            src = ROOT / "weights" / f"{t}_{method}_full{suffix}"
            if not src.exists():
                raise SystemExit(f"missing {src.name}; train it first")
            shutil.copy(src, stage / "weights" / src.name)
shutil.copytree(ROOT / "examples", stage / "examples", ignore=shutil.ignore_patterns("*.jpg"))

api.create_repo(repo, repo_type="space", space_sdk="gradio", exist_ok=True)
api.upload_folder(repo_id=repo, repo_type="space", folder_path=stage, commit_message="Deploy VPT demo",
                  delete_patterns=["weights/*", "examples/**"])
print(f"https://huggingface.co/spaces/{repo}")
print(f"Space id for the deck: {repo}")
