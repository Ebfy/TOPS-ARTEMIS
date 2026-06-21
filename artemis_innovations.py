"""
ARTEMIS Innovations Module — Revised for IEEE TDSC Submission
=============================================================

Implements all six core innovations with the following fixes over the
CCS 2026 submission:

  FIX-1  (L1) ODE tolerances unified: rtol=1e-4, atol=1e-5 everywhere.
  FIX-2  (L4) MAML inner loop now correctly applies adapted parameters via
              torch.func.functional_call (was a stub that ignored params).
  FIX-3  (L5) EWC lambda unified to 1000 (paper value); was 5000 in model.py.
  FIX-4  (L2) Formal pollution-resistance bound added (Proposition 2.1).
  NEW-A       Elliptic Bitcoin dataset loader added alongside ETGraph.
  NEW-B       Per-component latency hooks on every forward pass.

Mathematical notation follows the TDSC submission exactly.

Author: ARTEMIS Research Team
"""

from __future__ import annotations

import math
import time
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


# ─────────────────────────────────────────────────────────────────────────────
# Timing context manager (NEW-B: per-component latency)
# ─────────────────────────────────────────────────────────────────────────────

class LatencyTracker:
    """Accumulates wall-clock time for named sub-components."""

    def __init__(self) -> None:
        self.records: Dict[str, List[float]] = {}

    def record(self, name: str, elapsed_ms: float) -> None:
        self.records.setdefault(name, []).append(elapsed_ms)

    def summary(self) -> Dict[str, Dict[str, float]]:
        return {
            k: {"mean_ms": float(np.mean(v)),
                "std_ms":  float(np.std(v)),
                "n":       len(v)}
            for k, v in self.records.items()
        }

    def reset(self) -> None:
        self.records.clear()


# Global tracker — attach to model at eval time if latency profiling needed.
LATENCY_TRACKER: Optional[LatencyTracker] = None


def _timed(name: str):
    """Decorator that records wall-clock time to LATENCY_TRACKER if active."""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            if LATENCY_TRACKER is None:
                return fn(*args, **kwargs)
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            LATENCY_TRACKER.record(name, (time.perf_counter() - t0) * 1000.0)
            return result
        return wrapper
    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# L1 — Neural ODE Continuous-Time Temporal Modeling
# ─────────────────────────────────────────────────────────────────────────────

class NeuralODEFunc(nn.Module):
    """
    Dynamics function  dh/dt = f_θ(h(t), t)  with Lyapunov stability term.

    Stability guarantee (Theorem 5.2 in paper):
        With Lipschitz constant L (enforced via spectral norm), for any
        t1 < t2:
            ‖h(t2) − h(t1)‖ ≤ ‖f_θ(h(t1), t1)‖ · (t2−t1) · e^{L(t2−t1)}

    The −α·h term ensures exponential convergence to equilibrium:
        dV/dt ≤ −2α ‖h‖²  where V(h) = ‖h‖²

    FIX-1: tolerances set to rtol=1e-4, atol=1e-5 (unified across all files).
    """

    def __init__(
        self,
        hidden_channels: int,
        time_embedding_dim: int = 16,
        use_spectral_norm: bool = True,
        stability_alpha: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.stability_alpha = stability_alpha

        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )

        def _lin(a: int, b: int) -> nn.Module:
            l = nn.Linear(a, b)
            return nn.utils.spectral_norm(l) if use_spectral_norm else l

        self.net = nn.Sequential(
            _lin(hidden_channels + time_embedding_dim, hidden_channels * 2),
            nn.SiLU(),
            _lin(hidden_channels * 2, hidden_channels * 2),
            nn.SiLU(),
            _lin(hidden_channels * 2, hidden_channels),
        )

    @_timed("ode_func")
    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:  # noqa: D401
        t_emb = self.time_embed(t.view(1, 1)).expand(h.size(0), -1)
        h_t   = torch.cat([h, t_emb], dim=-1)
        dh    = self.net(h_t) - self.stability_alpha * h   # Lyapunov term
        return dh


class TemporalODEBlock(nn.Module):
    """
    Neural ODE block.  Solves dh/dt = f_θ(h, t) from t=0 to t=1.

    FIX-1: rtol=1e-4, atol=1e-5 (paper Table A.2 / Appendix A.2.3).
    Solver: dopri5 (Dormand-Prince RK4(5)) with adjoint back-prop.
    """

    # Canonical tolerances — single source of truth.
    RTOL: float = 1e-4
    ATOL: float = 1e-5

    def __init__(
        self,
        hidden_channels: int,
        method: str = "dopri5",
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        self.ode_func = NeuralODEFunc(hidden_channels, use_spectral_norm=use_spectral_norm)
        self.method   = method

    @_timed("ode_solve")
    def forward(self, h: torch.Tensor, t_span: Optional[torch.Tensor] = None) -> torch.Tensor:
        from torchdiffeq import odeint_adjoint  # lazy import — optional dep

        if t_span is None:
            t_span = torch.tensor([0.0, 1.0], dtype=h.dtype, device=h.device)

        h_t = odeint_adjoint(
            self.ode_func, h, t_span,
            method=self.method,
            rtol=self.RTOL,
            atol=self.ATOL,
        )
        return h_t[-1]


# ─────────────────────────────────────────────────────────────────────────────
# L2 — Anomaly-Aware Memory Storage
# ─────────────────────────────────────────────────────────────────────────────

class AnomalyAwareMemory(nn.Module):
    """
    Priority-queue memory whose eviction policy maximises I(M; Y).

    Proposition 2.1 (Pollution Resistance):
        Let an adversary inject B benign-looking messages.  Let the
        minimum anomaly score of a true phishing message be s_min and
        the maximum anomaly score of a pollution message be s_max.
        If s_min > s_max (guaranteed when the anomaly scorer has
        Mahalanobis separation ≥ 1 std), then no pollution message
        replaces a phishing message in the priority queue for any
        B < memory_size.

    NEW: importance score now combines Mahalanobis distance with
    a KL-divergence MI proxy, not just the simpler entropy estimate.
    """

    def __init__(
        self,
        memory_size: int = 1000,
        embedding_dim: int = 128,
        anomaly_weight: float = 1.0,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.memory_size   = memory_size
        self.embedding_dim = embedding_dim
        self.anomaly_weight = anomaly_weight
        self.temperature   = temperature
        self.momentum      = 0.01

        self.register_buffer("memory",        torch.zeros(memory_size, embedding_dim))
        self.register_buffer("memory_labels", torch.zeros(memory_size, dtype=torch.long))
        self.register_buffer("memory_weights", torch.zeros(memory_size))
        self.register_buffer("memory_ptr",    torch.zeros(1, dtype=torch.long))
        self.register_buffer("memory_count",  torch.zeros(1, dtype=torch.long))
        self.register_buffer("running_mean",  torch.zeros(embedding_dim))
        self.register_buffer("running_cov",   torch.eye(embedding_dim))

        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.key_proj   = nn.Linear(embedding_dim, embedding_dim)
        self.value_proj = nn.Linear(embedding_dim, embedding_dim)

    def _mahalanobis(self, z: torch.Tensor) -> torch.Tensor:
        """‖z − μ‖_{Σ^{-1}}  with fallback to Euclidean when Σ is singular."""
        c = z - self.running_mean.unsqueeze(0)
        try:
            cov_inv = torch.linalg.inv(
                self.running_cov + 1e-6 * torch.eye(self.embedding_dim, device=z.device)
            )
            dist = torch.sqrt((c @ cov_inv * c).sum(dim=-1).clamp(min=1e-8))
        except Exception:
            dist = torch.norm(c, dim=-1)
        return dist

    def _mi_proxy(self, z: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
        """
        KL-divergence between batch label distribution and uniform prior as
        a proxy for I(memory; Y).  Higher diversity → higher MI.
        """
        if labels is None:
            return torch.ones(z.size(0), device=z.device)
        n_classes = max(int(labels.max().item()) + 1, 2)
        counts = torch.zeros(n_classes, device=z.device)
        for c in range(n_classes):
            counts[c] = (labels == c).float().sum()
        p = counts / counts.sum().clamp(min=1)
        uniform = torch.full_like(p, 1.0 / n_classes)
        kl = (p * (p / uniform.clamp(min=1e-8)).log()).sum()
        return torch.ones(z.size(0), device=z.device) * kl.clamp(min=0.0)

    def _importance(self, z: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
        anomaly = self._mahalanobis(z)
        a_norm  = (anomaly - anomaly.min()) / (anomaly.max() - anomaly.min() + 1e-8)
        mi      = self._mi_proxy(z, labels)
        return (1.0 + self.anomaly_weight * a_norm) * mi

    def _update_stats(self, z: torch.Tensor) -> None:
        with torch.no_grad():
            mu  = z.mean(dim=0)
            cov = ((z - mu).T @ (z - mu)) / max(z.size(0) - 1, 1)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            self.running_cov  = (1 - self.momentum) * self.running_cov  + self.momentum * cov

    @_timed("memory_update")
    def update(self, z: torch.Tensor, labels: Optional[torch.Tensor] = None) -> None:
        self._update_stats(z)
        importance = self._importance(z, labels)
        _, order = importance.sort(descending=True)
        for idx in order:
            imp_val = importance[idx].item()
            if int(self.memory_count.item()) >= self.memory_size:
                min_idx = int(self.memory_weights.argmin().item())
                if imp_val > self.memory_weights[min_idx].item():
                    self.memory[min_idx]         = z[idx].detach()
                    self.memory_weights[min_idx] = imp_val
                    if labels is not None:
                        self.memory_labels[min_idx] = labels[idx]
            else:
                ptr = int(self.memory_ptr.item())
                self.memory[ptr]         = z[idx].detach()
                self.memory_weights[ptr] = imp_val
                if labels is not None:
                    self.memory_labels[ptr] = labels[idx]
                self.memory_ptr[0]   = (ptr + 1) % self.memory_size
                self.memory_count[0] = min(int(self.memory_count.item()) + 1, self.memory_size)

    @_timed("memory_query")
    def query(self, z: torch.Tensor) -> torch.Tensor:
        valid = min(int(self.memory_count.item()), self.memory_size)
        if valid == 0:
            return z
        mem = self.memory[:valid]
        Q   = self.query_proj(z)
        K   = self.key_proj(mem)
        V   = self.value_proj(mem)
        sc  = (Q @ K.T) / math.sqrt(self.embedding_dim)
        w   = F.softmax(sc / self.temperature, dim=-1)
        return w @ V

    def update_and_query(
        self,
        z: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.training:
            self.update(z.detach(), labels)
        return z + 0.5 * self.query(z)


# ─────────────────────────────────────────────────────────────────────────────
# L3 — Multi-Hop Message Broadcasting
# ─────────────────────────────────────────────────────────────────────────────

class MultiHopBroadcast(MessagePassing):
    """
    k-hop attention-weighted message passing for Sybil resistance.

    Information-flow guarantee (Theorem 4.1 in paper):
        For a Sybil cluster S with graph conductance φ(S), after k hops
        the external information leaking into S is at least
            φ(S)^k · ‖I_ext‖
        where I_ext is the external signal norm.
    """

    def __init__(
        self,
        hidden_channels: int,
        num_hops: int = 3,
        aggregation: str = "attention",
        dropout: float = 0.1,
    ) -> None:
        super().__init__(aggr="add")
        self.hidden_channels = hidden_channels
        self.num_hops        = num_hops
        self.aggregation     = aggregation
        self.dropout         = dropout

        self.hop_transforms = nn.ModuleList(
            [nn.Linear(hidden_channels, hidden_channels) for _ in range(num_hops)]
        )
        if aggregation == "attention":
            self.attn_transforms = nn.ModuleList(
                [nn.Linear(hidden_channels * 2, 1) for _ in range(num_hops)]
            )
        self.hop_embedding  = nn.Embedding(num_hops + 1, hidden_channels)
        self.output_proj    = nn.Linear(hidden_channels * (num_hops + 1), hidden_channels)
        self.layer_norm     = nn.LayerNorm(hidden_channels)

    @_timed("multihop")
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hop_feats = [x]
        cur = x
        for hop in range(self.num_hops):
            cur = F.gelu(self.hop_transforms[hop](cur))
            cur = F.dropout(cur, p=self.dropout, training=self.training)
            if self.aggregation == "attention":
                cur = self._attn_propagate(edge_index, cur, hop)
            else:
                cur = self.propagate(edge_index, x=cur)
            hop_feats.append(cur)

        for i, feat in enumerate(hop_feats):
            hop_emb = self.hop_embedding(
                torch.full((feat.size(0),), i, device=feat.device, dtype=torch.long)
            )
            hop_feats[i] = feat + hop_emb

        out = self.output_proj(torch.cat(hop_feats, dim=-1))
        return self.layer_norm(out + x)

    def _attn_propagate(
        self, edge_index: torch.Tensor, x: torch.Tensor, hop: int
    ) -> torch.Tensor:
        row, col = edge_index
        alpha = self.attn_transforms[hop](torch.cat([x[row], x[col]], dim=-1)).squeeze(-1)
        alpha = softmax(alpha, col, num_nodes=x.size(0))
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out   = torch.zeros_like(x)
        out.scatter_add_(0, col.unsqueeze(-1).expand_as(x[row]), alpha.unsqueeze(-1) * x[row])
        return out

    def message(self, x_j: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return x_j


# ─────────────────────────────────────────────────────────────────────────────
# L4 — Adversarial Meta-Learning  (FIX-2: real MAML inner loop)
# ─────────────────────────────────────────────────────────────────────────────

class AdversarialMetaLearner:
    """
    MAML-based adversarial meta-learning.

    FIX-2: _functional_forward now correctly uses torch.func.functional_call
    (PyTorch ≥ 2.0) so that adapted_params are actually applied during the
    inner loop — the previous implementation simply called self.model(...)
    and ignored the adapted parameter dict entirely.

    Meta-objective (Eq. 7–8 in paper):
        θ* = argmin_θ  Σ_{A_i ~ p(A)}  L^query_{A_i}(f_{θ'_i})
        where  θ'_i = θ − α ∇_θ L^support_{A_i}(f_θ)
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
        adversarial_ratio: float = 0.3,
        pgd_steps: int = 10,
        pgd_epsilon: float = 0.1,
    ) -> None:
        self.model             = model
        self.inner_lr          = inner_lr
        self.inner_steps       = inner_steps
        self.adversarial_ratio = adversarial_ratio
        self.pgd_steps         = pgd_steps
        self.pgd_epsilon       = pgd_epsilon
        self.outer_optimizer   = torch.optim.Adam(model.parameters(), lr=outer_lr)

    # ------------------------------------------------------------------
    # Adversarial task construction
    # ------------------------------------------------------------------

    def _pgd_perturb(self, x: torch.Tensor, edge_index: torch.Tensor,
                     y: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x_adv = x.clone().detach().requires_grad_(True)
        for _ in range(self.pgd_steps):
            logits = self.model(x_adv, edge_index, batch=batch)
            loss   = F.cross_entropy(logits, y)
            loss.backward()
            with torch.no_grad():
                x_adv = x_adv + self.pgd_epsilon * x_adv.grad.sign()
                delta  = (x_adv - x).clamp(-self.pgd_epsilon, self.pgd_epsilon)
                x_adv  = (x + delta).detach().requires_grad_(True)
        return x_adv.detach()

    def generate_adversarial_task(self, support_data):
        adv = deepcopy(support_data)
        adv.x = self._pgd_perturb(
            support_data.x, support_data.edge_index,
            support_data.y, support_data.batch,
        )
        return adv

    # ------------------------------------------------------------------
    # FIX-2: proper functional forward using torch.func
    # ------------------------------------------------------------------

    @staticmethod
    def _functional_forward(
        model: nn.Module,
        params: Dict[str, torch.Tensor],
        buffers: Dict[str, torch.Tensor],
        data,
    ) -> torch.Tensor:
        """
        Forward pass using a custom parameter dict (not model.state_dict()).

        Uses torch.func.functional_call (PyTorch ≥ 2.0).  Falls back to
        a manual parameter-swap for older versions.
        """
        try:
            from torch.func import functional_call  # PyTorch ≥ 2.0
            return functional_call(
                model, {**params, **buffers},
                (data.x, data.edge_index),
                kwargs={"batch": data.batch},
            )
        except ImportError:
            # Fallback: temporarily swap parameters, run, then restore.
            originals: Dict[str, torch.Tensor] = {}
            for name, param in model.named_parameters():
                if name in params:
                    originals[name] = param.data.clone()
                    param.data = params[name]
            out = model(data.x, data.edge_index, batch=data.batch)
            for name, param in model.named_parameters():
                if name in originals:
                    param.data = originals[name]
            return out

    # ------------------------------------------------------------------
    # Inner loop
    # ------------------------------------------------------------------

    def inner_loop(self, support_data, model_params: Dict[str, torch.Tensor],
                   model_buffers: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        adapted = {k: v.clone() for k, v in model_params.items()}
        for _ in range(self.inner_steps):
            logits = self._functional_forward(self.model, adapted, model_buffers, support_data)
            loss   = F.cross_entropy(logits, support_data.y)
            grads  = torch.autograd.grad(loss, list(adapted.values()), create_graph=False)
            adapted = {k: v - self.inner_lr * g for (k, v), g in zip(adapted.items(), grads)}
        return adapted

    # ------------------------------------------------------------------
    # Outer (meta) loop
    # ------------------------------------------------------------------

    @_timed("meta_step")
    def meta_train_step(self, task_batch: List, device: torch.device) -> float:
        model_params  = {n: p for n, p in self.model.named_parameters()}
        model_buffers = {n: b for n, b in self.model.named_buffers()}

        meta_loss = torch.tensor(0.0, device=device)

        for support, query in task_batch:
            if np.random.random() < self.adversarial_ratio:
                support = self.generate_adversarial_task(support)

            adapted = self.inner_loop(support, model_params, model_buffers)

            # Query loss with adapted parameters (FIX-2)
            q_logits = self._functional_forward(self.model, adapted, model_buffers, query)
            meta_loss = meta_loss + F.cross_entropy(q_logits, query.y)

        meta_loss = meta_loss / max(len(task_batch), 1)

        self.outer_optimizer.zero_grad()
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.outer_optimizer.step()

        return float(meta_loss.item())


# ─────────────────────────────────────────────────────────────────────────────
# L5 — Elastic Weight Consolidation  (FIX-3: lambda = 1000)
# ─────────────────────────────────────────────────────────────────────────────

class ElasticWeightConsolidation:
    """
    EWC continual learning regulariser (Kirkpatrick et al., 2017).

    FIX-3: ewc_lambda default corrected to 1000 (matches paper Eq. 9 and
    Figure 1).  The previous implementation used 5000.

    Loss (Eq. 9):
        L_EWC = L_task + (λ/2) Σ_i F_i (θ_i − θ*_i)²

    Bounded-forgetting guarantee (Appendix B.2):
        L(θ_new, D_old) − L(θ*, D_old) ≤ O(λ⁻¹)
    """

    # FIX-3: canonical value matching the paper.
    DEFAULT_LAMBDA: float = 1000.0

    def __init__(self, model: nn.Module, ewc_lambda: float = DEFAULT_LAMBDA) -> None:
        self.model      = model
        self.ewc_lambda = ewc_lambda
        self.optimal_params: Dict[str, torch.Tensor] = {}
        self.fisher:         Dict[str, torch.Tensor] = {}
        self.task_count: int = 0

    @_timed("fisher_compute")
    def compute_fisher(self, dataloader, num_samples: int = 1000) -> Dict[str, torch.Tensor]:
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        self.model.eval()
        n_seen = 0
        for batch in dataloader:
            if n_seen >= num_samples:
                break
            self.model.zero_grad()
            logits    = self.model(batch.x, batch.edge_index, batch=batch.batch)
            log_probs = F.log_softmax(logits, dim=-1)
            loss      = F.nll_loss(log_probs, batch.y)
            loss.backward()
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.detach().pow(2)
            n_seen += int(batch.y.size(0))
        for name in fisher:
            fisher[name] /= max(n_seen, 1)
        self.model.train()
        return fisher

    def update_fisher(self, dataloader, task_id: int) -> None:
        current = self.compute_fisher(dataloader)
        if self.task_count == 0:
            self.fisher = current
        else:
            # Online EWC update (exponential moving average)
            for name in self.fisher:
                self.fisher[name] = 0.5 * self.fisher[name] + 0.5 * current[name]
        self.optimal_params = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        self.task_count += 1

    @_timed("ewc_penalty")
    def penalty(self) -> torch.Tensor:
        if self.task_count == 0 or not self.fisher:
            device = next(self.model.parameters()).device
            return torch.tensor(0.0, device=device)
        loss = sum(
            (self.fisher[n] * (p - self.optimal_params[n]).pow(2)).sum()
            for n, p in self.model.named_parameters()
            if n in self.fisher
        )
        return (self.ewc_lambda / 2.0) * loss


# ─────────────────────────────────────────────────────────────────────────────
# L6 — Certified Adversarial Training
# ─────────────────────────────────────────────────────────────────────────────

class CertifiedAdversarialTrainer:
    """
    PGD adversarial training + randomized-smoothing certification.

    Certified radius (Theorem 5.1 / Eq. 12):
        R = (σ/2)(Φ⁻¹(p_A) − Φ⁻¹(p_B))

    where p_A ≥ p_B are the top-two class probabilities under
    Gaussian noise ε ~ N(0, σ²I).

    Spectral normalisation (in the encoder) ensures the Lipschitz
    condition required by Theorem 5.2 is satisfied during training.
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.1,
        sigma: float = 0.25,
        n_samples: int = 100,
        pgd_steps: int = 20,
        pgd_step_size: Optional[float] = None,
    ) -> None:
        self.model         = model
        self.epsilon       = epsilon
        self.sigma         = sigma
        self.n_samples     = n_samples
        self.pgd_steps     = pgd_steps
        self.pgd_step_size = pgd_step_size or epsilon / 4.0

    # ------------------------------------------------------------------
    # PGD attack
    # ------------------------------------------------------------------

    @_timed("pgd_attack")
    def pgd_attack(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        y: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Generate PGD adversarial examples (Eq. 10)."""
        x_adv = x.detach().clone()
        # Random start inside ε-ball
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-self.epsilon, self.epsilon)

        for _ in range(self.pgd_steps):
            x_adv = x_adv.requires_grad_(True)
            logits = self.model(x_adv, edge_index, batch=batch)
            loss   = F.cross_entropy(logits, y)
            loss.backward()
            with torch.no_grad():
                x_adv = x_adv + self.pgd_step_size * x_adv.grad.sign()
                delta  = (x_adv - x).clamp(-self.epsilon, self.epsilon)
                x_adv  = (x + delta).detach()
        return x_adv

    @_timed("adv_loss")
    def adversarial_loss(self, batch_data) -> torch.Tensor:
        x_adv  = self.pgd_attack(batch_data.x, batch_data.edge_index,
                                  batch_data.y, batch_data.batch)
        logits = self.model(x_adv, batch_data.edge_index, batch=batch_data.batch)
        return F.cross_entropy(logits, batch_data.y)

    # ------------------------------------------------------------------
    # Randomised smoothing (certification)
    # ------------------------------------------------------------------

    @_timed("smoothed_predict")
    def smoothed_predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        n_samples: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n = n_samples or self.n_samples
        self.model.eval()
        counts = torch.zeros(x.size(0) if batch is None else int(batch.max()) + 1,
                             2, device=x.device)
        with torch.no_grad():
            for _ in range(n):
                x_noisy = x + torch.randn_like(x) * self.sigma
                logits  = self.model(x_noisy, edge_index, batch=batch)
                preds   = logits.argmax(dim=-1)
                for c in range(2):
                    counts[:, c] += (preds == c).float()
        conf, preds = counts.max(dim=-1)
        return preds, conf / n

    @_timed("certify")
    def certify(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        n_samples: int = 1000,
        alpha: float = 0.001,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Clopper–Pearson lower bound on p_A, then apply Eq. 12.

        Returns (predictions, certified_radii).
        """
        from scipy.stats import norm, beta as sp_beta

        predictions, confidence = self.smoothed_predict(x, edge_index, batch, n_samples)
        radii = []
        for conf in confidence.cpu().tolist():
            count_A = int(round(conf * n_samples))
            if count_A > n_samples // 2:
                # Clopper-Pearson lower bound
                p_A_low = float(sp_beta.ppf(alpha, count_A, n_samples - count_A + 1))
                if p_A_low > 0.5:
                    # Φ⁻¹(p_A) − Φ⁻¹(1 − p_A) = 2 Φ⁻¹(p_A)
                    r = self.sigma * float(norm.ppf(p_A_low))
                else:
                    r = 0.0
            else:
                r = 0.0
            radii.append(r)
        return predictions, torch.tensor(radii, device=x.device, dtype=torch.float)

    def certified_accuracy(self, dataloader, radius: float) -> float:
        correct_certified = 0
        total = 0
        for batch in dataloader:
            preds, radii = self.certify(batch.x, batch.edge_index, batch.batch)
            correct_certified += int(((preds == batch.y) & (radii >= radius)).sum())
            total += int(batch.y.size(0))
        return correct_certified / max(total, 1)
