#!/usr/bin/env bash
# Post-training chain: waits for the robust checkpoint, then runs Round B
# white-box attacks, unseen-family (zoo) attacks, and robust recalibration.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
LOG=data/robust/logs/post_training_chain.log

echo "[chain] waiting for data/models/robust_model.pt ..." | tee -a "$LOG"
while [ ! -f data/models/robust_model.pt ]; do
  sleep 60
done
echo "[chain] checkpoint found; letting training finish writing summaries" | tee -a "$LOG"
sleep 20

echo "[chain] round B: robust-model val attacks" | tee -a "$LOG"
"$PY" scripts/generate_attack_suite.py --model-tag robust --splits val \
  --attacks fgsm,pgd,patch --max-per-class 8 >> "$LOG" 2>&1

echo "[chain] round B: robust-model test attacks" | tee -a "$LOG"
"$PY" scripts/generate_attack_suite.py --model-tag robust --splits test \
  --attacks fgsm,pgd,patch --max-per-class 8 >> "$LOG" 2>&1

echo "[chain] unseen families: robust-model test zoo attacks" | tee -a "$LOG"
"$PY" scripts/generate_attack_suite.py --model-tag robust --splits test \
  --attacks cw_l2,deepfool,mim,square,hopskipjump,autoattack --zoo-max-per-class 2 >> "$LOG" 2>&1

echo "[chain] recalibrating against the robust model" | tee -a "$LOG"
"$PY" scripts/calibrate_detectors.py --model-tag robust >> "$LOG" 2>&1

echo "[chain] done" | tee -a "$LOG"
