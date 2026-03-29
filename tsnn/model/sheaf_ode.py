"""Continuous-Time Sheaf Neural ODE (Innovation 1).

Replaces the discrete GRU temporal update with a Neural ODE that evolves
node states via learned sheaf diffusion dynamics:

    dh/dt = -σ(L_F(h,t)) h + g_θ(h, t)

where L_F is the sheaf Laplacian constructed from Householder transport maps,
and g_θ is a learned forcing term. Connects to parallel transport on fiber
bundles — Q_uv(t) is a learned connection on a vector bundle over the
molecular graph.

Key advantages:
- Continuous-time: handles variable MD timesteps natively
- Anti-oversmoothing: sheaf Laplacian equilibrium preserves discriminative info
- Bounded dynamics: tanh prevents ODE blow-up
- Adaptive depth: solver chooses number of function evaluations
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from torchdiffeq import odeint

from tsnn.model.layers.mlp import MLP
from tsnn.utils.scatter import scatter_add


class SheafODEFunc(nn.Module):
    """ODE dynamics function for continuous sheaf transport.

    Computes dh/dt given current node states h, using transported
    messages through sheaf transport maps Q_uv.

    The graph structure (edge_index, edge_attr, Q) is set externally
    before each integration interval via set_graph().

    Args:
        hidden_dim: Node embedding dimension.
        edge_dim: Edge feature dimension.
        dropout: Dropout probability.
    """

    def __init__(self, hidden_dim: int, edge_dim: int = 0, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Message network: h_u || Q_uv h_v || e_uv -> message
        msg_input_dim = 2 * hidden_dim + edge_dim
        self.msg_net = MLP(
            msg_input_dim, hidden_dim, hidden_dim,
            num_layers=2, dropout=dropout,
        )

        # Dynamics network: h || aggregated_msg -> dh/dt
        self.dynamics_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),  # Bounds ||dh/dt|| for solver stability
        )

        # Graph data (set before each integration interval)
        self._edge_index: Tensor | None = None
        self._edge_attr: Tensor | None = None
        self._Q: Tensor | None = None
        self._N: int = 0

    def set_graph(self, edge_index: Tensor, edge_attr: Tensor | None, Q: Tensor):
        """Set graph structure for the current integration interval."""
        self._edge_index = edge_index
        self._edge_attr = edge_attr
        self._Q = Q
        self._N = max(edge_index.max().item() + 1, 1) if edge_index.numel() > 0 else 0

    def forward(self, t: Tensor, h: Tensor) -> Tensor:
        """Compute dh/dt.

        Args:
            t: Current time (scalar tensor, unused but required by odeint).
            h: Node states [N, D].

        Returns:
            dh/dt [N, D].
        """
        if self._edge_index is None or self._edge_index.shape[1] == 0:
            return torch.zeros_like(h)

        row, col = self._edge_index

        # Sheaf-transported messages: Q_uv @ h_v
        h_source = h[col]  # [E, D]
        h_transported = torch.bmm(
            self._Q, h_source.unsqueeze(-1)
        ).squeeze(-1)  # [E, D]

        # Build message
        msg_parts = [h[row], h_transported]
        if self._edge_attr is not None:
            msg_parts.append(self._edge_attr)
        msg_input = torch.cat(msg_parts, dim=-1)
        messages = self.msg_net(msg_input)  # [E, D]

        # Aggregate to nodes
        agg = scatter_add(messages, row, dim=0, dim_size=h.shape[0])  # [N, D]

        # Dynamics: dh/dt = f(h, agg) — bounded by tanh
        dh_dt = self.dynamics_net(torch.cat([h, agg], dim=-1))  # [N, D]
        return dh_dt


class ContinuousSheafTransport(nn.Module):
    """Continuous-time sheaf transport via piecewise Neural ODE.

    For each MD frame interval [t_i, t_{i+1}]:
    1. Compute Householder transport maps Q_uv from current states
    2. Set graph structure on ODE function
    3. Integrate dh/dt continuously using adaptive solver
    4. Compute sheaf disagreements on evolved states

    This is the continuous-time analog of TemporalSheafTransport,
    replacing the discrete GRU update with Neural ODE integration.

    Args:
        hidden_dim: Node embedding dimension.
        edge_dim: Edge feature dimension.
        householder_depth: Number of Householder reflections.
        dropout: Dropout probability.
        static_frames: If True, use time-invariant frames (ablation).
        identity_transport: If True, Q_uv=I (ablation).
        ode_solver: ODE solver method ('dopri5', 'euler', 'rk4').
        ode_rtol: Relative tolerance for adaptive solvers.
        ode_atol: Absolute tolerance for adaptive solvers.
        ode_step_size: Step size for fixed-step solvers.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int = 0,
        householder_depth: int = 4,
        dropout: float = 0.0,
        static_frames: bool = False,
        identity_transport: bool = False,
        ode_solver: str = "dopri5",
        ode_rtol: float = 1e-3,
        ode_atol: float = 1e-3,
        ode_step_size: float = 0.25,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.static_frames = static_frames
        self.identity_transport = identity_transport
        self.ode_solver = ode_solver
        self.ode_rtol = ode_rtol
        self.ode_atol = ode_atol
        self.ode_step_size = ode_step_size

        # Householder frame builder
        if not identity_transport:
            from tsnn.model.householder import HouseholderFrameBuilder
            self.frame_builder = HouseholderFrameBuilder(
                hidden_dim, householder_depth
            )

        # ODE dynamics function
        self.ode_func = SheafODEFunc(hidden_dim, edge_dim, dropout)

        # Message MLP for disagreement computation (shared with ODE func)
        self.message_mlp = self.ode_func.msg_net

    def _compute_transport(self, h: Tensor, edge_index: Tensor) -> Tensor:
        """Compute transport maps Q_uv."""
        if self.identity_transport:
            E = edge_index.shape[1]
            return torch.eye(
                self.hidden_dim, device=h.device, dtype=h.dtype
            ).unsqueeze(0).expand(E, -1, -1)
        _, Q = self.frame_builder(h, edge_index)
        return Q

    def _compute_disagreement(
        self, h: Tensor, Q: Tensor, edge_index: Tensor
    ) -> Tensor:
        """Compute sheaf disagreement D_uv(t) = ||h_u - Q_uv h_v||^2."""
        row, col = edge_index
        h_u = h[row]
        h_v = h[col]
        Qh_v = torch.bmm(Q, h_v.unsqueeze(-1)).squeeze(-1)
        diff = h_u - Qh_v
        return (diff * diff).sum(dim=-1).clamp(max=1e4)

    def _ode_integrate(self, h: Tensor) -> Tensor:
        """Integrate the ODE for one frame interval [0, 1]."""
        t_span = torch.tensor([0.0, 1.0], device=h.device, dtype=h.dtype)

        solver_kwargs = {"method": self.ode_solver}
        if self.ode_solver in ("dopri5", "dopri8", "bosh3", "adaptive_heun"):
            solver_kwargs["rtol"] = self.ode_rtol
            solver_kwargs["atol"] = self.ode_atol
        else:
            solver_kwargs["options"] = {"step_size": self.ode_step_size}

        solution = odeint(self.ode_func, h, t_span, **solver_kwargs)
        return solution[-1]  # Return final state [N, D]

    def forward(
        self,
        h_sequence: list[Tensor],
        edge_index_sequence: list[Tensor],
        edge_attr_sequence: list[Tensor] | None = None,
        num_nodes_per_frame: list[int] | None = None,
    ):
        """Process temporal sequence via piecewise continuous ODE.

        Same interface as TemporalSheafTransport.forward().
        """
        from tsnn.model.sheaf_transport import SheafTransportOutput

        T = len(h_sequence)
        h_prev = h_sequence[0]

        all_h = []
        all_disagreements = []
        all_Q = []

        # Static frames ablation
        if self.static_frames and not self.identity_transport:
            U_static, _ = self.frame_builder(h_prev, edge_index_sequence[0])

        for t in range(T):
            h_t = h_sequence[t]
            ei_t = edge_index_sequence[t]
            ea_t = edge_attr_sequence[t] if edge_attr_sequence else None

            # Fuse encoder output with temporal state
            h_fused = h_prev + h_t

            # Compute transport maps
            if self.static_frames and not self.identity_transport:
                from tsnn.model.householder import compute_transport_maps
                Q_t = compute_transport_maps(U_static, ei_t)
            else:
                Q_t = self._compute_transport(h_fused, ei_t)

            # Set graph structure on ODE function
            self.ode_func.set_graph(ei_t, ea_t, Q_t)

            # Continuous ODE integration instead of discrete GRU
            h_current = self._ode_integrate(h_fused)

            # Compute sheaf disagreement
            D_t = self._compute_disagreement(h_current, Q_t, ei_t)

            all_h.append(h_current)
            all_disagreements.append(D_t)
            all_Q.append(Q_t)

            h_prev = h_current

        return SheafTransportOutput(
            h_final=all_h[-1],
            h_sequence=all_h,
            disagreements=all_disagreements,
            transport_maps=all_Q,
        )
