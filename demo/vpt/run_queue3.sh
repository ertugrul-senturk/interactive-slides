#!/bin/sh
# After run_queue2.sh: linear probe and full fine-tuning for Blood and Pneumonia, so the demo has
# all three methods for all three tasks. Full fine-tuning reuses the learning rate picked on the
# DermaMNIST 1k sweep; linear probing gets its own short sweep.
set -e
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1
until [ -f weights/dermamnist_full_full.json ]; do sleep 30; done
FULL_LR=$($PY -c "import json;print(json.load(open('weights/dermamnist_full_1k.json'))['base_lr'])")

$PY train.py --data pneumoniamnist --method linear --sweep 2.5 10 --sweep-epochs 8 --epochs 20 --eval-every 2 > logs/pneumoniamnist_linear_full.log 2>&1
echo "done pneumonia linear"
$PY train.py --data bloodmnist --method linear --sweep 2.5 10 --sweep-epochs 6 --epochs 20 --eval-every 2 > logs/bloodmnist_linear_full.log 2>&1
echo "done blood linear"
$PY train.py --data pneumoniamnist --method full --batch 32 --wd 0.01 --base-lr "$FULL_LR" --epochs 20 --eval-every 2 > logs/pneumoniamnist_full_full.log 2>&1
echo "done pneumonia full"
$PY train.py --data bloodmnist --method full --batch 32 --wd 0.01 --base-lr "$FULL_LR" --epochs 15 --eval-every 3 > logs/bloodmnist_full_full.log 2>&1
echo "done blood full"
