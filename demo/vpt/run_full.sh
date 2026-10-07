#!/bin/sh
# Full fine-tuning baseline on DermaMNIST, paper recipe (AdamW, Table 6 grid). 1e-4 was clearly worst
# in a first sweep (val bal. acc. 43.0 vs 58.5 at 5e-4 and 60.7 at 1e-3), so this sweep spans the upper
# end of the paper's grid, including 5e-3. Batch 32 keeps the whole backbone's gradients in 12 GB.
set -e
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1
lr() { $PY -c "import json;print(json.load(open('weights/$1.json'))['base_lr'])"; }
$PY train.py --data dermamnist --method full --subset 1000 --batch 32 --wd 0.01 --sweep 0.0005 0.001 0.005 --epochs 100 > logs/derma_full_1k.log 2>&1
echo "done full 1k"
$PY train.py --data dermamnist --method full --batch 32 --wd 0.01 --base-lr "$(lr dermamnist_full_1k)" --epochs 30 --eval-every 3 > logs/derma_full_full.log 2>&1
echo "done full full"
