# TSNN-Drug Autoresearch Loop

Adapted from karpathy/autoresearch for iterative improvement of the
Equivariant Temporal Sheaf Network for Protein-Ligand Dissociation Kinetics.

## Setup

1. **Create experiment branch**: `git checkout -b autoresearch/<tag>` from main.
2. **Read the in-scope files**:
   - `configs/default.yaml` — all hyperparameters
   - `scripts/train.py` — training entry point (modifiable)
   - `tsnn/model/tsnn.py` — top-level model
   - `tsnn/model/equivariant_encoder.py` — E(3) encoder
   - `tsnn/model/sheaf_transport.py` — temporal sheaf block
   - `tsnn/model/contact_hazard_head.py` — contact risk scores
   - `tsnn/model/survival_head.py` — survival/koff head
   - `tsnn/losses/combined.py` — combined loss (Eq. 16)
   - `tsnn/training/trainer.py` — 3-stage trainer
3. **Initialize results.tsv** with header.
4. **Run baseline** and record it.

## What You CAN Modify

- `configs/default.yaml` — all hyperparameters
- `tsnn/model/` — architecture (encoder, sheaf transport, heads, layers)
- `tsnn/losses/` — loss functions and weights
- `tsnn/training/trainer.py` — training loop, optimizer, scheduler
- `scripts/train.py` — data pipeline, synthetic data generation
- `tsnn/data/` — data pipeline modules

## What You CANNOT Modify

- Test files in `tests/` (they define correctness contracts)
- `paper.tex` (the paper spec)

## Goal

**Minimize training loss on Stage C while maintaining all 51 tests passing.**

Since we use synthetic data, the metric is `train_loss` from Stage C.
Each experiment runs for a **fixed 5 epochs** to keep runs comparable (~30 seconds each).

Secondary goals:
- Improve model architecture for better gradient flow
- Reduce VRAM usage
- Speed up training throughput (samples/sec)

## Output Format

After training, extract the final epoch loss from the log:
```
grep "Epoch 5/5" run.log
```

## Logging Results

Log to `results.tsv` (tab-separated):
```
commit	train_loss	vram_mb	status	description
a1b2c3d	4.726	1234	keep	baseline
```

## Experiment Loop

LOOP FOREVER:

1. Check git state (branch, last commit)
2. Make a change (architecture, hyperparameters, optimizer, loss)
3. `git commit`
4. Run tests: `conda run -n tsnn pytest tests/ -x -q 2>&1 | tail -5`
5. If tests fail: fix or revert
6. Run training: `conda run -n tsnn python scripts/train.py training.stage=c training.device=cuda training.stage_c.num_epochs=5 > run.log 2>&1`
7. Extract results: `grep "Epoch 5/5" run.log` and `nvidia-smi --query-gpu=memory.used --format=csv,noheader`
8. Log to results.tsv
9. If loss improved: keep commit
10. If loss worse or crash: `git reset --hard HEAD~1`

**NEVER STOP**: Run indefinitely until manually interrupted.

## Experiment Ideas (Priority Order)

1. **Baseline** — run as-is to establish reference
2. **Learning rate sweep** — try 1e-4, 5e-4, 1e-3 for Stage C
3. **Hidden dim** — try 64, 256
4. **Encoder depth** — try 2, 6, 8 layers
5. **Householder depth** — try 1, 2, 8
6. **Optimizer** — try SGD+momentum, LAMB, or learning rate warmup tuning
7. **Activation functions** — try GELU, SiLU in MLP layers
8. **Sheaf transport layers** — try 2, 3 layers
9. **Gradient clipping** — try 0.5, 2.0, 5.0
10. **Mixed precision** — toggle on/off
11. **Batch size effect** — synthetic dataset size 50, 200
12. **Loss weights** — tune alpha, beta, gamma
13. **Residual connections** — add skip connections in sheaf transport
14. **Layer normalization** — add/remove in various places
15. **Dropout tuning** — try 0.0, 0.05, 0.2
