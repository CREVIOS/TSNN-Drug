# NeurIPS 2026 Submission Plan

## Deadline: May 6, 2026 (Submission portal opens April 5)

- Format: 9 pages main + unlimited appendix, single-column NeurIPS LaTeX template
- Review: Double-blind, ~25-27% acceptance rate
- Also consider: **Datasets & Benchmarks Track** (separate submission portal)
- Submission: https://neurips.cc/Conferences/2026/CallForPapers

---

## Paper: Equivariant Temporal Sheaf Networks for Protein-Ligand Dissociation Kinetics

**Pitch**: An equivariant temporal sheaf framework for learning dissociation-relevant
geometry from MD trajectories, paired with a rigorous kinetics benchmark and mechanistic
contact-level analyses.

**Contribution Triple** (all three required for competitiveness):
1. **Method**: Equivariant temporal sheaf transport
2. **Benchmark**: 6-split de-leaked kinetics evaluation
3. **Mechanistic analysis**: Sheaf disagreement as contact-rupture early-warning

---

## Tables

### Table 1: Main Results — Kinetics Prediction Across 6 Splits

| Method | Random | Cold-Protein | Cold-Scaffold | Pocket-Cluster | Congeneric | Interaction-Deleaked |
|--------|--------|-------------|---------------|----------------|------------|---------------------|
| **TSNN (Ours)** | rho / RMSE | **rho** / RMSE | rho / RMSE | rho / RMSE | rho / RMSE | **rho** / RMSE |
| STELLAR-koff | | | | | | |
| Dynaformer | | | | | | |
| DynamicDTA | | | | | | |
| DynHeter-DTA | | | | | | |
| MDbind Timenucy | | | | | | |
| MDbind Videonucy | | | | | | |
| GEMS (CleanSplit) | | | | | | |
| Static Equivariant | | | | | | |

**Primary metric**: Spearman rho (bold on cold-protein + interaction-deleaked)
**Secondary**: RMSE (log k_off), C-index, Integrated Brier Score (appendix)

### Table 2: Ablation Study (Cold-Protein Split)

| # | Ablation | Spearman rho | Δρ | RMSE | C-index |
|---|----------|-------------|-----|------|---------|
| — | **TSNN Full** | **X.XXX** | — | X.XXX | X.XXX |
| 1 | No sheaf transport (Q=I) | | | | |
| 2 | Static frames (time-invariant U_v) | | | | |
| 3 | No E(3) encoder (RBF only) | | | | |
| 4 | No Stage B pretraining | | | | |
| 5 | No contact auxiliary task | | | | |
| 6 | No water edges | | | | |
| 7 | No survival head | | | | |
| 8 | Atom-only graph | | | | |
| 9 | Window: 5ns / 50ns | | | | |
| 10 | Householder depth: 1/2/4/8 | | | | |

### Table 3: Mechanistic — Contact-Break Early Warning

| Metric | TSNN (sheaf disagreement) | Distance baseline | RMSF baseline | Flat GNN attention |
|--------|--------------------------|-------------------|---------------|-------------------|
| Contact-break AUROC | | | | |
| Contact-break AUPRC | | | | |
| Mean lead time (frames) | | | | |
| Median lead time (frames) | | | | |

---

## Figures

1. **Architecture diagram** — 4-component pipeline (exists in paper.tex)
2. **3-stage training pipeline** — Stage A→B→C with transfer (exists in paper.tex)
3. **Split comparison bar chart** — x=6 splits, y=Spearman rho, bars=top methods
4. **Sheaf disagreement trajectories** — D_uv(t) rising before contact rupture (2-3 case studies from DD-13M)
5. **AUROC vs lead time tradeoff** — threshold sweep curve
6. **Householder depth vs performance** (appendix) — k={1,2,4,8} line plot

---

## Paper Outline (9 pages)

### Section 1: Introduction (~1 page)
- k_off prediction is critical for drug design but underserved by ML
- Recent benchmarking papers (PLINDER, HiQBind, CleanSplit) exposed leakage in affinity models
- New opportunity: large MD corpora + sheaf-theoretic GNN tools
- Central claim + contribution triple

### Section 2: Related Work (~1 page)
- Static affinity models and their leakage problems
- MD-aware models: Dynaformer, DynamicDTA, DynHeter-DTA, MDbind
- Kinetics-specific: STELLAR-koff
- Sheaf neural networks: Neural Sheaf Diffusion, physics-informed sheaf Laplacians
- Survival analysis in ML

### Section 3: Problem Setup (~0.5 page)
- Formal task definition: MD frames → log k_off, hazard, survival
- Hybrid residue-atom graph definition
- Sheaf disagreement definition and central hypothesis

### Section 4: Method (~2 pages)
- 4.1: E(3)-equivariant local encoder (EGNN)
- 4.2: Temporal sheaf transport (Householder Q_uv, GRU update)
- 4.3: Contact hazard head (disagreement → risk scores)
- 4.4: Survival head (hazard → survival curve → log k_off)
- Figures: architecture diagram

### Section 5: Theory (~0.75 page)
- Theorem 1: Non-expansiveness of orthogonal sheaf diffusion
- Proposition 1: Hazard upper bound via sheaf disagreement
- Transport as local frame alignment

### Section 6: Training Pipeline (~0.75 page)
- Stage A: Self-supervised MD pretraining (MISATO, MDbind, MDD)
- Stage B: Dissociation pretraining (DD-13M) — strongest reviewer defense
- Stage C: Kinetics fine-tuning with combined loss (Eq. 16)
- Figure: 3-stage pipeline diagram

### Section 7: Experiments (~2.5 pages)
- 7.1: Benchmark protocol (6 splits, de-leaking procedure)
- 7.2: Baselines (8 methods, all evaluated under same splits)
- 7.3: Main results (Table 1, Figure 3)
- 7.4: Ablation study (Table 2)
- 7.5: Mechanistic analysis (Table 3, Figures 4-5)
- 7.6: Efficiency (training time, memory)

### Section 8: Discussion & Limitations (~0.5 page)
- Independent contact assumption (Proposition 1) ignores correlation
- Short MD as pretraining source, not sufficient supervision
- Window length sensitivity (ablation 9)

### Section 9: Conclusion (~0.25 page)
- Contribution triple recap
- Future: correlated contacts, enhanced sampling, k_on prediction

### Appendix (unlimited)
- A: Full metric tables (all methods × splits × metrics)
- B: Dataset statistics
- C: Implementation details (hyperparameters, hardware)
- D: Extended ablations + plots
- E: Additional case studies
- F: Proof of Theorem 1

---

## Code → Paper Mapping

| Code Module | Paper Section | Output |
|---|---|---|
| `tsnn/model/tsnn.py` + sub-modules | §4 Method | Architecture |
| `tsnn/training/trainer.py` | §6 Training | Pipeline |
| `tsnn/losses/*.py` | §6.3 Stage C | Loss formulation |
| `tsnn/data/graph_builder.py` | §3 + §4.1 | Graph construction |
| `tsnn/data/splits/*.py` | §7.1 | Split definitions |
| `tsnn/data/datasets/*.py` | §6 | Data loading |
| `tsnn/evaluation/metrics.py` | §7.3 | All metrics |
| `tsnn/evaluation/benchmark.py` | §7.3 | Table 1 |
| `tsnn/evaluation/ablation_runner.py` | §7.4 | Table 2 |
| `tsnn/evaluation/mechanistic.py` | §7.5 | Table 3 + Figs 4-5 |

---

## Data Requirements

| Dataset | Size | Stage | Source | Status |
|---------|------|-------|--------|--------|
| MISATO | 133 GB | A (pretrain) | Zenodo 7711953 | Downloading |
| MDD | 24.5 GB | A (prototype) | Zenodo 11172815 | Downloading |
| DD-13M | 204 GB | B (dissociation) | HuggingFace SZBL-IDEA/MD | Downloading |
| BindingDB | 525 MB | C (koff labels) | bindingdb.org | Pending |
| KOFFI | Small | C (koff labels) | koffidb.org API | API down |
| PDBbind-koff | Small | C (koff labels) | pdbbind.org.cn (registration) | Manual |

**Download commands:**
```bash
# MISATO (133 GB)
wget -O data/raw/misato/MD.hdf5 https://zenodo.org/record/7711953/files/MD.hdf5
wget -O data/raw/misato/QM.hdf5 https://zenodo.org/record/7711953/files/QM.hdf5

# MDD (24.5 GB)
python scripts/download_data.py --mdd

# DD-13M (204 GB)
python scripts/download_data.py --dd13m

# After download:
python scripts/preprocess_data.py --all
```

---

## Timeline (39 days to May 6)

### Week 1 (Mar 28 - Apr 3): Data + Infrastructure
- [x] Model architecture (51 tests passing)
- [x] 3-stage training pipeline working
- [x] Data download/preprocessing pipeline
- [ ] Complete all data downloads
- [ ] Preprocess MISATO, MDD → Stage A HDF5
- [ ] Preprocess DD-13M → Stage B HDF5
- [ ] Merge koff labels → Stage C CSV

### Week 2 (Apr 4 - Apr 10): Training
- [ ] Stage A pretraining on MISATO (~2 days GPU)
- [ ] Stage B pretraining on DD-13M (~1 day GPU)
- [ ] Stage C fine-tuning on kinetics data (~hours)
- [ ] Validate on random split — sanity check

### Week 3 (Apr 11 - Apr 17): Evaluation
- [ ] Run 6-split benchmark (Table 1)
- [ ] Run 10 ablations (Table 2)
- [ ] Mechanistic case studies on DD-13M (Table 3)
- [ ] Generate all figures

### Week 4 (Apr 18 - Apr 24): Baselines + Writing
- [ ] Implement/run baseline models OR collect reported numbers
- [ ] Write Sections 1-4 (Intro, Related Work, Setup, Method)
- [ ] Write Section 5 (Theory) + proof in appendix

### Week 5 (Apr 25 - May 1): Writing + Polish
- [ ] Write Sections 6-9 (Training, Experiments, Discussion, Conclusion)
- [ ] Format all tables and figures
- [ ] Write appendix (full tables, dataset stats, implementation details)
- [ ] Internal review pass

### Week 6 (May 2 - May 6): Final
- [ ] Final proofreading
- [ ] Anonymize code repository
- [ ] Check NeurIPS formatting compliance
- [ ] **Submit by May 6 AOE**

---

## What a Convincing Result Looks Like

1. Competitive on random split (no regression from baselines)
2. Clearly better on cold-protein and interaction-deleaked splits
3. Stage B dissociation pretraining improves kinetics fine-tuning
4. Contact-risk head predicts rupture earlier than flat baselines
5. Case studies show interpretable failure modes at binding site
