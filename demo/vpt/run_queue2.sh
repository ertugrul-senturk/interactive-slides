#!/bin/sh
# Runs after run_derma.sh: VPT-deep prompts for the two extra demo tasks, then the full fine-tuning baseline.
set -e
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1
# data/.downloaded is written once both archives pass their MD5 check
until [ -f weights/dermamnist_linear_full.json ] && [ -f data/.downloaded ]; do sleep 20; done
lr() { $PY -c "import json;print(json.load(open('weights/$1.json'))['base_lr'])"; }

for d in pneumoniamnist bloodmnist; do
  $PY train.py --data $d --method vpt --subset 1000 --sweep 2.5 10 --sweep-epochs 15 --epochs 60 > logs/${d}_vpt_1k.log 2>&1
  $PY train.py --data $d --method vpt --base-lr "$(lr ${d}_vpt-deep-p50_1k)" --epochs 20 --eval-every 2 > logs/${d}_vpt_full.log 2>&1
  echo "done $d"
done


echo "done full 1k"
$PY train.py --data dermamnist --method full --batch 32 --wd 0.01 --base-lr "$(lr dermamnist_full_1k)" --epochs 30 --eval-every 3 > logs/derma_full_full.log 2>&1
echo "done full full"
