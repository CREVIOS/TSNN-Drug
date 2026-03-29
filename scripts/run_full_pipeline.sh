#!/bin/bash
# Full end-to-end TSNN-Drug pipeline with ODE+Multiscale (best config from v1)
set -euo pipefail
cd /home/hpc4090/asif_tanzila/TSNN-Drug

CONDA="conda run --no-capture-output -n tsnn"
LOG="runs_autoresearch/full_pipeline_ode_ms.log"
mkdir -p runs_autoresearch

echo "============================================================" | tee "$LOG"
echo "TSNN-Drug Full Pipeline: ODE+Multiscale (best config)"       | tee -a "$LOG"
echo "Started: $(date)"                                              | tee -a "$LOG"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader)" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

# Best config from v1_ode_ms_baseline
COMMON="model.hidden_dim=64 model.use_ode=true model.ode_solver=euler model.use_multiscale=true model.chebyshev_order=2 model.num_bands=2 model.householder_depth=4 training.device=cuda training.mixed_precision=true training.batch_size=1 training.grad_clip=1.0 data.window_size=10 data.stride=5"

echo ""
echo "[Stage A] Self-supervised MD pretraining (MDD/synthetic)" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
$CONDA python scripts/train.py \
    training.stage=a \
    training.stage_a.num_epochs=10 \
    training.stage_a.lr=0.0003 \
    $COMMON \
    >> "$LOG" 2>&1

echo ""
echo "[Stage B] Dissociation pretraining (DD-13M real data)" | tee -a "$LOG"  
echo "============================================================" | tee -a "$LOG"
$CONDA python scripts/train.py \
    training.stage=b \
    training.stage_b.num_epochs=10 \
    training.stage_b.lr=0.0001 \
    $COMMON \
    >> "$LOG" 2>&1

echo ""
echo "[Stage C] Kinetics fine-tuning (BindingDB koff labels)" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
$CONDA python scripts/train.py \
    training.stage=c \
    training.stage_c.num_epochs=15 \
    training.stage_c.lr=0.00005 \
    $COMMON \
    >> "$LOG" 2>&1

echo ""
echo "============================================================" | tee -a "$LOG"
echo "Full pipeline complete: $(date)" | tee -a "$LOG"
echo "Results:" | tee -a "$LOG"
grep "Epoch.*train_loss" "$LOG" | tee -a /dev/stderr
echo "============================================================" | tee -a "$LOG"
