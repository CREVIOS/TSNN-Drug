# TSNN: How to Run

Equivariant Temporal Sheaf Networks for Protein-Ligand Dissociation Kinetics

---

## 1. Environment Setup

```bash
cd /Users/asif/Desktop/tsnn_drug

# Create and activate the virtual environment
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv tsnn_env
source tsnn_env/bin/activate

# Install dependencies
pip install torch torch-geometric numpy scipy scikit-learn \
            pytest einops pyyaml biopython h5py pandas tqdm

# Optional: for trajectory loading from raw MD files
pip install MDAnalysis

# Optional: for ligand scaffold splits
pip install rdkit

# Optional: for experiment tracking
pip install wandb

# Verify installation
python -m pytest tests/ -v
# Expected: 51 passed
```

---

## 2. Data Preparation

The model supports 5 datasets across 3 training stages. You need at least one.

### Option A: Start with MDD (recommended for prototyping)

```
data/mdd/
├── complexes/
│   ├── complex_001/
│   │   ├── topology.pdb
│   │   └── trajectory.xtc
│   ├── complex_002/
│   │   └── ...
├── splits/
│   ├── train.txt          # one complex_id per line
│   ├── val.txt
│   └── test.txt
└── metadata.csv           # columns: complex_id, koff (optional)
```

To preprocess raw trajectories into .pt files for fast loading:

```bash
python -c "
from tsnn.data.datasets.mdd import MDDDataset
ds = MDDDataset(root='data/mdd', split='train', use_mdanalysis=True)
print(f'Loaded {len(ds)} complexes')
"
```

### Option B: Use MISATO (Stage A pretraining at scale)

Download from the MISATO release. Place HDF5 at:

```
data/misato/
├── misato_dataset.h5
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

### Option C: Use DD-13M (Stage B dissociation pretraining)

```
data/dd13m/
├── trajectories/
│   ├── complex_001/
│   │   ├── 0.h5
│   │   ├── 1.h5
│   │   └── ...
├── contact_labels/
│   ├── complex_001_0_contacts.npz
│   └── ...
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

### Option D: Kinetics labels for Stage C

```
data/kinetics/
├── kinetics_labels.csv    # columns: complex_id, log_koff, censored, source, series_id
├── structures/
│   ├── complex_001.pdb
│   └── ...
└── splits/
    ├── random/
    │   ├── train.txt
    │   ├── val.txt
    │   └── test.txt
    ├── cold_protein/
    │   └── ...
    └── interaction_deleaked/
        └── ...
```

### Option E: No data (synthetic test run)

If no data directory exists, the training script automatically generates synthetic data for pipeline testing. This is useful for verifying the setup works.

---

## 3. Configuration

Edit `configs/default.yaml` or pass overrides on the command line.

Key settings:

```yaml
model:
  hidden_dim: 128           # Main embedding dimension
  encoder_layers: 4         # EGNN layers
  householder_depth: 4      # Householder reflections (k)

data:
  root: "data/mdd"          # Path to your dataset
  pocket_cutoff: 10.0       # Angstroms
  context_cutoff: 15.0      # Angstroms
  window_size: 20           # Frames per temporal window
  include_water: true

training:
  stage: "c"                # "a", "b", "c", or "all"
  device: "cuda"
  batch_size: 4
  mixed_precision: true

losses:
  alpha: 0.1                # Regression weight
  beta: 0.05                # Ranking weight
  gamma: 0.01               # Sheaf smoothness weight
```

---

## 4. Training

### Quick test (synthetic data, no GPU needed)

```bash
source tsnn_env/bin/activate

# Stage C with synthetic data (verifies pipeline works)
python scripts/train.py training.stage=c training.device=cpu
```

### Full 3-stage pipeline

The paper recommends training in order: A → B → C. Each stage saves checkpoints that the next stage loads automatically.

```bash
# Stage A: Self-supervised MD pretraining
# Data: MISATO / MDbind / MDD
python scripts/train.py \
    training.stage=a \
    data.root=data/mdd \
    training.stage_a.num_epochs=50 \
    training.stage_a.lr=3e-4

# Stage B: Dissociation pretraining on DD-13M
python scripts/train.py \
    training.stage=b \
    data.root=data/dd13m \
    training.stage_b.num_epochs=30 \
    training.stage_b.lr=1e-4

# Stage C: Kinetics fine-tuning
python scripts/train.py \
    training.stage=c \
    data.root=data/kinetics \
    training.stage_c.num_epochs=100 \
    training.stage_c.lr=5e-5
```

Or run all three sequentially:

```bash
python scripts/train.py training.stage=all data.root=data/mdd
```

### Checkpoints

Saved to `checkpoints/` by default:

```
checkpoints/
├── stage_a_best.pt
├── stage_a_final.pt
├── stage_b_best.pt
├── stage_b_final.pt
├── stage_c_best.pt
└── stage_c_final.pt
```

---

## 5. Evaluation

### Run benchmark across all 6 splits

```bash
python scripts/evaluate.py checkpoints/stage_c_best.pt results/
```

This produces:

```
results/
├── benchmark_results.json    # All metrics as JSON
└── benchmark_table.tex       # LaTeX table for the paper
```

### Metrics reported

| Category | Metrics |
|----------|---------|
| Kinetics regression | RMSE, Spearman ρ, Pearson r, MAE |
| Censored prediction | Concordance index (C-index), Integrated Brier Score |
| Mechanistic | Contact-break AUROC/AUPRC, lead time before rupture |

---

## 6. Ablation Studies

Run all 10 ablations from Section 8 of the paper:

```python
from tsnn.model.tsnn import TSNNConfig
from tsnn.evaluation.ablation_runner import run_all_ablations

base_config = TSNNConfig()

results = run_all_ablations(
    base_config=base_config,
    train_fn=your_training_function,   # (overrides, name) -> model
    evaluate_fn=your_evaluation_function,  # (model) -> metrics
    output_dir="ablation_results",
)
```

Individual ablations can be run directly via config overrides:

```bash
# Ablation 1: No sheaf transport (Q=I)
python scripts/train.py model.identity_transport=true

# Ablation 2: Static frames
python scripts/train.py model.static_frames=true

# Ablation 3: No E(3) encoder
python scripts/train.py model.no_equivariant=true

# Ablation 7: No survival head
python scripts/train.py model.use_survival=false

# Ablation 10: Householder depth k=1,2,8
python scripts/train.py model.householder_depth=1
python scripts/train.py model.householder_depth=2
python scripts/train.py model.householder_depth=8

# Ablation 6: No water edges
python scripts/train.py data.include_water=false

# Ablation 9: Short/long windows
python scripts/train.py data.window_size=5
python scripts/train.py data.window_size=50
```

---

## 7. Mechanistic Analysis

Generate sheaf disagreement case studies for individual complexes:

```python
from tsnn.evaluation.mechanistic import generate_case_study

generate_case_study(
    complex_id="3HTB",
    disagreement_sequence=disagreements,  # from model output
    edge_index_sequence=edge_indices,
    contact_break_times=break_times,      # from DD-13M labels
    output_dir="case_studies/",
)
```

---

## 8. Generating Benchmark Splits

Create the 6 de-leaked splits for your dataset:

```python
from tsnn.data.splits import SPLIT_REGISTRY

# Random split
train, val, test = SPLIT_REGISTRY["random"](complex_ids)

# Cold-protein split (needs sequences)
train, val, test = SPLIT_REGISTRY["cold_protein"](
    complex_ids, protein_sequences, identity_threshold=0.3
)

# Cold-scaffold split (needs SMILES, requires rdkit)
train, val, test = SPLIT_REGISTRY["cold_scaffold"](complex_ids, smiles)

# Interaction-deleaked split (needs fingerprints)
train, val, test = SPLIT_REGISTRY["interaction_deleaked"](
    complex_ids, interaction_fingerprints=fingerprints,
    tanimoto_threshold=0.4
)
```

Save splits:

```python
for name, ids in [("train", train), ("val", val), ("test", test)]:
    with open(f"data/splits/{split_type}/{name}.txt", "w") as f:
        f.write("\n".join(ids))
```

---

## 9. Project Structure

```
tsnn_drug/
├── configs/default.yaml          # All hyperparameters
├── scripts/
│   ├── train.py                  # Training entry point
│   └── evaluate.py               # Evaluation entry point
├── tsnn/
│   ├── model/                    # Architecture (4 components)
│   │   ├── tsnn.py               # Top-level model
│   │   ├── equivariant_encoder.py
│   │   ├── sheaf_transport.py
│   │   ├── householder.py
│   │   ├── contact_hazard_head.py
│   │   └── survival_head.py
│   ├── data/                     # Data pipeline
│   │   ├── graph_builder.py      # Hybrid residue-atom graphs
│   │   ├── datasets/             # MISATO, MDbind, MDD, DD-13M, Kinetics
│   │   ├── splits/               # 6 benchmark split strategies
│   │   └── collate.py            # Custom DataLoader collation
│   ├── losses/                   # All loss functions
│   │   ├── combined.py           # Eq. 16: L_surv + α·L_reg + β·L_rank + γ·L_sheaf
│   │   ├── survival_nll.py       # Discrete-time survival NLL
│   │   └── pretraining_losses.py # Stage A + B auxiliary losses
│   ├── training/trainer.py       # 3-stage training orchestrator
│   └── evaluation/               # Metrics, benchmark, ablations, mechanistic
├── tests/                        # 51 tests (all passing)
└── paper.tex                     # The paper
```

---

## 10. Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: torch_geometric` | `pip install torch-geometric` |
| `No data found` warning | Expected if no real data. Uses synthetic data automatically. |
| CUDA out of memory | Reduce `model.hidden_dim`, `data.window_size`, or use `training.mixed_precision=true` |
| Slow training on CPU | Normal. Use GPU: `training.device=cuda` |
| `ImportError: rdkit` | Only needed for cold-scaffold split. `pip install rdkit` |
| `ImportError: MDAnalysis` | Only needed for raw .xtc/.dcd loading. `pip install MDAnalysis` |
