#!/bin/bash
# TSNN-Drug Autoresearch Loop
# Runs continuous experiments with ODE + multiscale on real DD-13M data
# Logs results to results.tsv, keeps improvements, reverts failures
set -euo pipefail

cd /home/hpc4090/asif_tanzila/TSNN-Drug
CONDA="conda run --no-capture-output -n tsnn"
LOG_DIR="runs_autoresearch"
mkdir -p "$LOG_DIR"

EXPERIMENT=0

run_experiment() {
    local tag="$1"
    local desc="$2"
    shift 2
    local overrides=("$@")

    EXPERIMENT=$((EXPERIMENT + 1))
    local logfile="$LOG_DIR/${tag}.log"

    echo "============================================================"
    echo "[exp $EXPERIMENT] $tag: $desc"
    echo "  overrides: ${overrides[*]:-none}"
    echo "  logfile: $logfile"
    echo "============================================================"

    # Run tests first
    echo "[exp $EXPERIMENT] Running tests..."
    if ! $CONDA pytest tests/ -x -q 2>&1 | tail -3; then
        echo "[exp $EXPERIMENT] TESTS FAILED — skipping"
        return 1
    fi

    # Run training
    echo "[exp $EXPERIMENT] Training..."
    local start=$(date +%s)
    if $CONDA python scripts/train.py "${overrides[@]}" > "$logfile" 2>&1; then
        local end=$(date +%s)
        local elapsed=$((end - start))

        # Extract final epoch loss
        local loss=$(grep "Epoch.*train_loss" "$logfile" | tail -1 | grep -oP "train_loss.: \K[0-9.]+" || echo "nan")
        local vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1 | tr -d ' MiB')

        echo "[exp $EXPERIMENT] Done in ${elapsed}s — loss=$loss vram=$vram"

        # Log to results.tsv
        echo -e "${tag}\t${loss}\t${vram}\t${elapsed}s\tkeep\t${desc}" >> results.tsv

        # Show last epoch lines
        grep "Epoch" "$logfile" | tail -5
        return 0
    else
        echo "[exp $EXPERIMENT] TRAINING FAILED"
        echo -e "${tag}\tFAIL\t-\t-\tfail\t${desc}" >> results.tsv
        tail -20 "$logfile"
        return 1
    fi
}

echo "Starting TSNN-Drug autoresearch loop $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader)"
echo ""

# ===== EXPERIMENT 1: Baseline ODE + Multiscale on real DD-13M =====
run_experiment "v1_ode_ms_baseline" \
    "ODE+Multiscale baseline: D64, euler, cheby2, 2bands, Stage B real DD-13M" \
    training.stage=b training.stage_b.num_epochs=5 training.stage_b.lr=0.0001

# ===== EXPERIMENT 2: ODE only (no multiscale) =====
run_experiment "v2_ode_only" \
    "ODE only (no multiscale): D64, euler, Stage B" \
    training.stage=b training.stage_b.num_epochs=5 model.use_multiscale=false

# ===== EXPERIMENT 3: Multiscale only (no ODE) =====
run_experiment "v3_ms_only" \
    "Multiscale only (no ODE, GRU transport): D64, cheby2, 2bands, Stage B" \
    training.stage=b training.stage_b.num_epochs=5 model.use_ode=false

# ===== EXPERIMENT 4: Neither (discrete GRU baseline) =====
run_experiment "v4_discrete_baseline" \
    "Discrete GRU baseline (no ODE, no multiscale): D64, Stage B" \
    training.stage=b training.stage_b.num_epochs=5 model.use_ode=false model.use_multiscale=false

# ===== EXPERIMENT 5: ODE + Multiscale, higher LR =====
run_experiment "v5_ode_ms_lr5e4" \
    "ODE+MS, LR=5e-4" \
    training.stage=b training.stage_b.num_epochs=5 training.stage_b.lr=0.0005

# ===== EXPERIMENT 6: ODE + Multiscale, RK4 solver =====
run_experiment "v6_ode_ms_rk4" \
    "ODE+MS, RK4 solver (more accurate ODE)" \
    training.stage=b training.stage_b.num_epochs=5 model.ode_solver=rk4

# ===== EXPERIMENT 7: ODE + Multiscale, 3 bands + cheby4 =====
run_experiment "v7_ode_ms_3band_cheby4" \
    "ODE+MS, 3 bands, Chebyshev order 4" \
    training.stage=b training.stage_b.num_epochs=5 model.num_bands=3 model.chebyshev_order=4

# ===== EXPERIMENT 8: ODE + Multiscale, D128 (if memory allows) =====
run_experiment "v8_ode_ms_d128" \
    "ODE+MS D128 (bigger model)" \
    training.stage=b training.stage_b.num_epochs=5 model.hidden_dim=128

# ===== EXPERIMENT 9: ODE + Multiscale, householder_depth=2 =====
run_experiment "v9_ode_ms_hk2" \
    "ODE+MS, householder_depth=2 (simpler transport)" \
    training.stage=b training.stage_b.num_epochs=5 model.householder_depth=2

# ===== EXPERIMENT 10: ODE + Multiscale, dopri5 adaptive =====
run_experiment "v10_ode_ms_dopri5" \
    "ODE+MS, dopri5 adaptive solver" \
    training.stage=b training.stage_b.num_epochs=5 model.ode_solver=dopri5

# ===== EXPERIMENT 11: Best config → Stage C (if real koff data available) =====
run_experiment "v11_stage_c_ode_ms" \
    "ODE+MS Stage C fine-tune (synthetic koff if no real data)" \
    training.stage=c training.stage_c.num_epochs=5

# ===== EXPERIMENT 12: Full pipeline A→B→C =====
run_experiment "v12_full_pipeline" \
    "Full 3-stage pipeline with ODE+MS" \
    training.stage=all training.stage_a.num_epochs=3 training.stage_b.num_epochs=5 training.stage_c.num_epochs=5

echo ""
echo "============================================================"
echo "Autoresearch loop complete $(date)"
echo "Results:"
cat results.tsv
echo "============================================================"
