#!/bin/sh
# DermaMNIST: VPT-deep vs linear probe, on a 1,000-image subset and on the full train split.
set -e
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1
mkdir -p logs
lr() { $PY -c "import json;print(json.load(open('weights/dermamnist_$1.json'))['base_lr'])"; }

$PY train.py --data dermamnist --method vpt    --subset 1000 --sweep 2.5 10 25 --epochs 100 > logs/derma_vpt_1k.log 2>&1
echo "done vpt 1k"
$PY train.py --data dermamnist --method linear --subset 1000 --sweep 0.5 2.5 10 --epochs 100 > logs/derma_linear_1k.log 2>&1
echo "done linear 1k"
$PY train.py --data dermamnist --method vpt    --base-lr "$(lr vpt-deep-p50_1k)" --epochs 30 --eval-every 3 > logs/derma_vpt_full.log 2>&1
echo "done vpt full"
$PY train.py --data dermamnist --method linear --base-lr "$(lr linear_1k)" --epochs 30 --eval-every 3 > logs/derma_linear_full.log 2>&1
echo "done linear full"
