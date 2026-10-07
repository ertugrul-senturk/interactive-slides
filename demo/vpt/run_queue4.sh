#!/bin/sh
# After run_queue3.sh: the 1,000-image runs for linear probing and full fine-tuning on Blood and
# Pneumonia, so every task has all three methods at both data sizes. Same 60 epochs as their VPT
# 1k runs; linear gets its own LR sweep, full reuses the LR picked on the DermaMNIST 1k sweep.
set -e
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1
until [ -f weights/bloodmnist_full_full.json ]; do sleep 30; done
FULL_LR=$($PY -c "import json;print(json.load(open('weights/dermamnist_full_1k.json'))['base_lr'])")
for d in pneumoniamnist bloodmnist; do
  $PY train.py --data $d --method linear --subset 1000 --sweep 2.5 10 --sweep-epochs 15 --epochs 60 > logs/${d}_linear_1k.log 2>&1
  $PY train.py --data $d --method full --subset 1000 --batch 32 --wd 0.01 --base-lr "$FULL_LR" --epochs 60 > logs/${d}_full_1k.log 2>&1
  echo "done $d 1k"
done
