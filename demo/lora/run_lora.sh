#!/bin/sh
# LoRA (rank 8, W_q and W_v) on the three demo tasks, at both training-set sizes. The schedule matches the VPT
# runs in ../vpt/run_*.sh: the learning rate is picked on val in a short sweep on the 1,000-image subset, then
# reused for the full train split. Epochs as for VPT: 100 / 30 on DermaMNIST, 60 / 20 on the other two.
set -e
cd "$(dirname "$0")"
PY=../vpt/.venv/Scripts/python.exe
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1
mkdir -p logs
lr() { $PY -c "import json;print(json.load(open('weights/$1.json'))['base_lr'])"; }
SWEEP="0.001 0.004 0.016"

$PY train.py --data dermamnist --subset 1000 --sweep $SWEEP --sweep-epochs 20 --epochs 100 > logs/dermamnist_lora_1k.log 2>&1
$PY train.py --data dermamnist --base-lr "$(lr dermamnist_lora-r8_1k)" --epochs 30 --eval-every 3 > logs/dermamnist_lora_full.log 2>&1
echo "done dermamnist"
for d in pneumoniamnist bloodmnist; do
  $PY train.py --data $d --subset 1000 --sweep $SWEEP --sweep-epochs 15 --epochs 60 > logs/${d}_lora_1k.log 2>&1
  $PY train.py --data $d --base-lr "$(lr ${d}_lora-r8_1k)" --epochs 20 --eval-every 2 > logs/${d}_lora_full.log 2>&1
  echo "done $d"
done
