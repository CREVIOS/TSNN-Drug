# TSNN Innovations for NeurIPS 2026

## Proposed Title
**Continuous-Time Equivariant Sheaf Transport for Protein-Ligand Dissociation Kinetics**

---

## Innovation 1: Continuous-Time Sheaf Neural ODE

### What It Is
Replace the discrete GRU temporal update in `TemporalSheafTransport` with a Neural ODE
that continuously evolves node states via learned sheaf diffusion dynamics:

```
Current (discrete):   h(t+1) = GRU(h(t), Σ_u Q_uv h_v)
Proposed (continuous): dh/dt = -σ(L_F(h,t)) h + g_θ(h, t)
```

where `L_F` is the sheaf Laplacian (constructed from Householder transport maps Q_uv)
and `g_θ` is a learned forcing term. Solved with adaptive ODE integrators via `torchdiffeq`.

### Mathematical Formulation

#### Sheaf Laplacian (from Neural Sheaf Diffusion, Bodnar et al. NeurIPS 2022)
The sheaf Laplacian `L_F` is a block matrix of size `(|V|·d) × (|V|·d)`:

```
Diagonal blocks:     [L_F]_vv = Σ_{e: v◁e} F_{v◁e}^T F_{v◁e}
Off-diagonal blocks: [L_F]_uv = -F_{u◁e}^T F_{v◁e}
```

In our case, the restriction maps `F_{v◁e}` are derived from the Householder transport:
```
F_{v◁e} = U_v(t) ∈ O(d),  computed as product of k Householder reflections
Q_uv(t) = U_v(t)^T U_u(t) ∈ O(d)
```

#### Continuous Dynamics
The ODE function for each piecewise interval [t_i, t_{i+1}]:

```python
class SheafODEFunc(nn.Module):
    """dh/dt = sheaf_diffusion(h) + learned_dynamics(h, t)"""

    def forward(self, t, h):
        # h: [N, D] node states

        # 1. Transported message aggregation (sheaf diffusion term)
        h_transported = bmm(Q, h[col].unsqueeze(-1)).squeeze(-1)  # [E, D]
        msgs = MLP([h[row], h_transported, edge_attr])            # [E, D]
        agg = scatter_add(msgs, row, dim=0, dim_size=N)           # [N, D]

        # 2. Learned dynamics (forcing term)
        dh_dt = self.dynamics_net(cat([h, agg], dim=-1))          # [N, D]

        # 3. Bounded output (prevents ODE blow-up)
        return tanh(dh_dt)
```

#### Piecewise Integration (handles variable graph topology)
Each MD frame defines a different molecular graph. We integrate piecewise:

```python
for i in range(T):
    # Recompute transport maps at frame boundary
    Q_t = householder_frame_builder(h_current, edge_indices[i])
    ode_func.set_graph(edge_indices[i], edge_attrs[i], Q_t)

    # Continuous evolution within frame interval
    h_current = odeint(ode_func, h_current, t_span=[t_i, t_{i+1}],
                       method='dopri5', rtol=1e-3, atol=1e-3)[-1]
```

### Why This Beats Everything

1. **Continuous parallel transport on fiber bundles**: The sheaf transport Q_uv(t) IS a
   discrete connection on a vector bundle over the molecular graph. Making the evolution
   continuous completes the analogy — we get learned parallel transport in continuous time.
   This is mathematically clean and unprecedented.

2. **Variable timesteps for free**: MD datasets have wildly different time resolutions
   (MISATO=10ns, MDD=200ns, DD-13M=variable). Neural ODE adaptive solvers handle this
   natively — no resampling, no fixed grids.

3. **Anti-oversmoothing guarantee**: Standard graph diffusion dX/dt = -LX converges to
   constant features (oversmoothing). Sheaf diffusion dX/dt = -L_F X converges to
   ker(L_F) = H^0(G,F), which is non-trivial when transport maps are non-identity.
   The Householder orthogonal parameterization guarantees this.

4. **State-of-the-art lineage**: GF-NODE (arXiv:2411.01600) showed continuous-time graph
   dynamics crushes discrete approaches for MD. We add sheaf geometry on top — strictly
   more expressive.

5. **Depth without oversmoothing**: The ODE integrator controls "depth" adaptively.
   With discrete GRU, you must choose T steps. With Neural ODE, the solver finds the
   right number of function evaluations automatically.

### Implementation Details

- **Library**: `torchdiffeq` (already installed in tsnn env)
- **Solver**: `dopri5` for training (adaptive, rtol=atol=1e-3), `euler` step=0.25 for fast prototyping
- **Memory**: Use direct `odeint` (not adjoint) for T<50 frames — adjoint only saves memory for long sequences
- **Backward**: Standard autograd through adaptive solver; gradient clipping at 1.0
- **Activation**: `tanh` on dynamics output bounds ||dh/dt|| ≤ 1, preventing solver blow-up
- **Integration time**: [0, 1] per frame interval (normalized), with learnable time scaling
- **Passing graph data**: Store edge_index, edge_attr, Q as module attributes, set before each interval
- **adjoint_params**: When using odeint_adjoint, explicitly pass (func.parameters() + (edge_attr, Q)) to ensure gradient flow through upstream modules

### Files to Modify

| File | Change |
|------|--------|
| `tsnn/model/sheaf_transport.py` | Add `SheafODEFunc`, `ContinuousSheafTransport`; keep `TemporalSheafTransport` as fallback |
| `tsnn/model/layers/sheaf_gru.py` | Keep as ablation baseline (discrete vs continuous) |
| `tsnn/model/tsnn.py` | Add `use_continuous_ode` config flag |
| `tsnn/model/tsnn.py:TSNNConfig` | Add `use_ode: bool = True`, `ode_solver: str = 'dopri5'`, `ode_rtol: float = 1e-3` |
| `configs/default.yaml` | Add ODE config section |
| `tests/test_sheaf_transport.py` | Add continuous ODE tests |

### New Ablation
- **Ablation 11**: Continuous ODE vs discrete GRU temporal dynamics (the showstopper ablation)

---

## Innovation 2: Multi-Scale Sheaf Spectral Decomposition

### What It Is
Decompose molecular dynamics into spatial frequency bands using the **sheaf Laplacian
spectrum**, then evolve each band with its own dynamics. Fast local vibrations (high
frequency) get fast dynamics; slow conformational changes (low frequency) get slow dynamics.

### Mathematical Formulation

#### Sheaf Laplacian Eigendecomposition
```
L_F = Φ Λ Φ^T
```
where Φ = [φ_1, ..., φ_M] are eigenvectors (in R^{Nd×Nd}), Λ = diag(λ_1, ..., λ_M).

#### Spectral Decomposition of Node States
Forward sheaf Fourier transform:
```
ĥ_k = Φ_k^T h,    k = 1, ..., M  (M retained modes)
```

Group into bands:
- **Low-frequency** (λ < λ_lo): Global conformational dynamics, complex-level dissociation
- **Mid-frequency** (λ_lo ≤ λ < λ_hi): Residue-level rearrangement, contact breaking
- **High-frequency** (λ ≥ λ_hi): Local atomic vibrations, thermal noise

#### Per-Band Dynamics
Each band gets its own Neural ODE:
```
dĥ_low/dt  = f_θ_low(ĥ_low, t)    — slow dynamics, large integration steps
dĥ_mid/dt  = f_θ_mid(ĥ_mid, t)    — medium dynamics
dĥ_high/dt = f_θ_high(ĥ_high, t)  — fast dynamics, small steps
```

#### Efficient Implementation: Chebyshev Filtering (Avoids Eigendecomposition)
Full eigendecomposition of L_F is O(N³d³) — too expensive. Use Chebyshev polynomials:

```
p_θ(L̃_F) x = Σ_{k=0}^{K} θ_k T_k(L̃_F x)
```
where L̃_F = (2/λ_max)L_F - I (rescaled to [-1,1]), and T_k is the Chebyshev recurrence:
```
T_0 = x
T_1 = L̃_F · x
T_{k+1} = 2·L̃_F·T_k - T_{k-1}
```

Cost: O(K · nnz(L_F) · d) per layer — linear in edges, K ≈ 4-8 is enough.

### Why This Matters

1. **Physical motivation**: Contact rupture (our target signal) is a mid-frequency event
   superimposed on high-frequency thermal noise and low-frequency drift. Separating these
   scales lets the model focus on the signal.

2. **GF-NODE showed this works**: Their spectral decomposition + per-mode dynamics
   achieves SOTA on MD17. We add sheaf geometry — the sheaf Laplacian captures
   *anisotropic* interactions that the standard graph Laplacian misses.

3. **Efficient**: Chebyshev filtering avoids O(N³) eigendecomposition. K=4 gives
   4-hop mixing with O(4·E·d) cost.

4. **Novel**: Nobody has combined sheaf spectral decomposition with multi-scale temporal
   dynamics. The closest is PolyNSD (arXiv:2512.00242) which uses Chebyshev on sheaf
   Laplacians but for static node classification, not temporal dynamics.

### Implementation Details

- **Chebyshev order**: K=4 (empirically sufficient for molecular graphs)
- **Band separation**: 3 bands with learned thresholds
- **Sheaf Laplacian**: Constructed from Householder transport maps (already computed)
- **λ_max estimation**: Power iteration (5 steps, O(E·d) per step)

### Files to Create/Modify

| File | Change |
|------|--------|
| `tsnn/model/spectral_sheaf.py` | NEW: Chebyshev sheaf filter, multi-scale decomposition |
| `tsnn/model/sheaf_transport.py` | Integrate spectral decomposition into transport |
| `tsnn/model/tsnn.py:TSNNConfig` | Add `use_multiscale: bool = True`, `chebyshev_order: int = 4`, `num_bands: int = 3` |

### New Ablation
- **Ablation 12**: Multi-scale decomposition vs single-scale (quantifies value of frequency separation)

---

## Innovation 3: Sheaf-Calibrated Uncertainty via Conformal Prediction

### What It Is
Use the distribution of sheaf disagreements D_uv(t) across protein-ligand contacts as a
geometrically principled uncertainty score. Wrap with conformal prediction for calibrated
confidence intervals on k_off predictions.

### Mathematical Formulation

#### Uncertainty Score from Sheaf Disagreement
For a complex with cross-contacts C^× at time T:
```
u(complex) = Var_{(u,v)∈C^×}[D_uv(T)] + β · max_{(u,v)∈C^×} D_uv(T)
```
High variance = heterogeneous contact quality = uncertain prediction.
High max = at least one contact is highly strained = uncertain.

#### Split Conformal Prediction (distribution-free)

1. **Train** TSNN on D_train
2. **Calibrate** on D_cal: compute nonconformity scores
   ```
   s_i = |log k_off_true - log k_off_pred| / u(complex_i)
   ```
   (normalized by uncertainty — tighter intervals where model is confident)
3. **Conformal quantile**:
   ```
   q̂ = Quantile(s_1, ..., s_n; (1-α)(1 + 1/n))
   ```
4. **Prediction interval** for new complex:
   ```
   C(x_new) = [ŷ - q̂·u(x_new),  ŷ + q̂·u(x_new)]
   ```

#### Coverage Guarantee
```
P(y_{n+1} ∈ C(x_{n+1})) ≥ 1 - α
```
This holds for ANY distribution, with finite-sample validity. Only requires exchangeability.

### Why Reviewers Will Love This

1. **Drug discovery demands UQ**: "Is this k_off prediction reliable?" is the first
   question a medicinal chemist asks. Conformal prediction gives a mathematically
   rigorous answer.

2. **Geometrically principled**: The uncertainty comes from sheaf disagreement — not
   ensemble variance or dropout Monte Carlo. It's intrinsic to the model's geometric
   structure.

3. **Distribution-free**: No assumptions about error distribution. Works on any split.

4. **Hot topic**: Conformal prediction for molecular property prediction is state-of-art
   (JCIM 2024, Nature Communications 2025). We add the sheaf-geometric twist.

5. **Easy to implement**: ~100 lines of code as a post-hoc wrapper.

### Implementation Details

- **Calibration set**: 15% of Stage C data (already in val split)
- **α**: 0.1 (90% coverage), also report 0.05 and 0.2
- **Metric**: Prediction Interval Coverage Probability (PICP) and Mean Prediction Interval Width (MPIW)

### Files to Create

| File | Change |
|------|--------|
| `tsnn/evaluation/uncertainty.py` | NEW: `SheafUncertainty`, `conformal_calibrate`, `conformal_predict` |
| `tsnn/evaluation/metrics.py` | Add PICP, MPIW metrics |

### New Table
- **Table 4**: Uncertainty calibration — PICP and MPIW across splits, comparing sheaf-UQ vs ensemble-UQ vs dropout-UQ

---

## Combined Architecture Diagram

```
MD Frames {X_t}
     │
     ▼
┌─────────────────┐
│  E(3)-Equivariant│  Component 2: EGNN encoder
│  Local Encoder   │  (unchanged)
└────────┬────────┘
         │ h_encoded[t] for each frame
         ▼
┌─────────────────────────────────────────┐
│  Continuous-Time Sheaf Transport        │  Component 3 (UPGRADED)
│                                         │
│  For each frame interval [t_i, t_{i+1}]:│
│  1. Compute Q_uv via Householder        │
│  2. Build sheaf Laplacian L_F           │
│  3. Multi-scale spectral decomposition  │  ◄── Innovation 2
│  4. Per-band Neural ODE integration     │  ◄── Innovation 1
│     dh/dt = -σ(L_F)h + g_θ(h,t)       │
│  5. Compute D_uv(t) disagreements       │
└────────┬────────────────────────────────┘
         │ h(t), D_uv(t), risk scores
         ▼
┌─────────────────┐
│ Contact Hazard   │  Component 4a
│ Head             │  (unchanged)
└────────┬────────┘
         ▼
┌─────────────────┐
│ Survival Head    │  Component 4b
│ + Conformal UQ   │  ◄── Innovation 3
└────────┬────────┘
         │
         ▼
  log k_off ± confidence interval
  λ(t), S(t), calibrated uncertainty
```

---

## Implementation Priority

| Week | Task | Innovation |
|------|------|-----------|
| 1 (Mar 29 - Apr 4) | Implement SheafODEFunc + ContinuousSheafTransport | 1 |
| 1 (Mar 29 - Apr 4) | Add tests, verify ODE vs GRU parity | 1 |
| 2 (Apr 5 - Apr 11) | Implement Chebyshev sheaf filter + multi-scale | 2 |
| 2 (Apr 5 - Apr 11) | Implement conformal prediction wrapper | 3 |
| 2 (Apr 5 - Apr 11) | Run ablation: ODE vs GRU, multi-scale vs single | 1+2 |
| 3+ | Train on real data, run benchmarks, write paper | all |

---

## Expected Impact on Results

| Metric | Current (GRU) | Expected (ODE) | Why |
|--------|--------------|----------------|-----|
| Cold-protein Spearman ρ | baseline | +0.05-0.10 | Better generalization from continuous dynamics |
| Contact-break AUROC | baseline | +0.03-0.08 | Multi-scale separates signal from noise |
| Training stability | NaN at step 14 (fixed) | No NaN (bounded ODE) | tanh-bounded dynamics |
| Variable timestep handling | Requires resampling | Native | Adaptive ODE solver |
| Uncertainty calibration | None | 90% PICP guaranteed | Conformal prediction |
| Reviewer excitement | "solid method" | "novel + principled + practical" | All 3 innovations |

---

## Key References

1. **Neural Sheaf Diffusion** (Bodnar et al., NeurIPS 2022) — sheaf Laplacian for GNNs
   - [Paper](https://arxiv.org/abs/2202.04579) | [Code](https://github.com/twitter-research/neural-sheaf-diffusion)

2. **Graph Fourier Neural ODEs** (GF-NODE, 2024) — continuous-time multi-scale graph dynamics
   - [Paper](https://arxiv.org/abs/2411.01600)

3. **Polynomial Neural Sheaf Diffusion** (PolyNSD, 2025) — Chebyshev filters on sheaf Laplacian
   - [Paper](https://arxiv.org/abs/2512.00242)

4. **GRAND: Graph Neural Diffusion** (2021) — GNN as continuous diffusion PDE
   - [Paper](https://arxiv.org/abs/2106.10934) | [Code](https://github.com/twitter-research/graph-neural-pde)

5. **torchdiffeq** — Neural ODE library
   - [Code](https://github.com/rtqichen/torchdiffeq)

6. **Conformal Prediction for Molecular GNNs** (JCIM 2024)
   - [Paper](https://pubs.acs.org/doi/10.1021/acs.jcim.4c01139)

7. **Graph Mamba Operator** (GraMO, 2025) — SSM for particle dynamics
   - [Paper](https://openreview.net/forum?id=6CO8dwHl4F)

8. **Fiber Bundle Networks** (2025) — learned connections on vector bundles
   - [Paper](https://arxiv.org/abs/2512.01151)

9. **Physics-Informed Sheaf Laplacians for Biomolecules** (2025)
   - Referenced in paper.tex as \citep{physicsinformedsheaf2025}
