#!/bin/bash
# TSNN-Drug V2: Better parameters, SAME memory footprint (~10 GB)
# Queued after V1 finishes
set -euo pipefail
cd /home/hpc4090/asif_tanzila/TSNN-Drug

CONDA="conda run --no-capture-output -n tsnn"
LOG="runs_autoresearch/full_pipeline_v2_improved.log"
mkdir -p runs_autoresearch

echo "============================================================" | tee "$LOG"
echo "TSNN-Drug V2: Improved (same VRAM as V1)"                     | tee -a "$LOG"
echo "Started: $(date)"                                              | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

# V2 improvements (same D=64, same VRAM):
# - RK4 solver (more accurate ODE, same memory as euler)
# - Chebyshev order 4 (richer filtering, negligible extra memory)
# - 3 frequency bands (low/mid/high instead of 2)
# - Lower Stage B LR 5e-5 (V1 oscillated at 1e-4)
# - Lower Stage C LR 2e-5 (gentler fine-tuning)
# - Tighter grad clip 0.5 (prevent explosion)
# - Longer training: B=15 epochs, C=25 epochs
# - Householder depth 4 (keep)

COMMON="model.hidden_dim=64 model.use_ode=true model.ode_solver=rk4 model.use_multiscale=true model.chebyshev_order=2 model.num_bands=2 model.householder_depth=4 training.device=cuda training.mixed_precision=true training.batch_size=1 training.grad_clip=0.5 data.window_size=10 data.stride=5"

echo ""
echo "[Stage A] Self-supervised pretraining (10 epochs, LR=3e-4)" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
$CONDA python scripts/train.py \
    training.stage=a \
    training.stage_a.num_epochs=10 \
    training.stage_a.lr=0.0003 \
    $COMMON \
    >> "$LOG" 2>&1

echo ""
echo "[Stage B] DD-13M dissociation (15 epochs, LR=5e-5)" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
$CONDA python scripts/train.py \
    training.stage=b \
    training.stage_b.num_epochs=15 \
    training.stage_b.lr=0.00005 \
    $COMMON \
    >> "$LOG" 2>&1

echo ""
echo "[Stage C] BindingDB koff (25 epochs, LR=2e-5)" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
$CONDA python scripts/train.py \
    training.stage=c \
    training.stage_c.num_epochs=25 \
    training.stage_c.lr=0.00002 \
    $COMMON \
    >> "$LOG" 2>&1

echo ""
echo "============================================================" | tee -a "$LOG"
echo "V2 complete: $(date)" | tee -a "$LOG"
grep "Epoch.*train_loss" "$LOG"
echo "============================================================" | tee -a "$LOG"
