"""
ARTEMIS Model — Revised for IEEE TDSC Submission
=================================================

Fixes from code review vs CCS 2026 submission
----------------------------------------------
FIX-1  ODE rtol=1e-4 / atol=1e-5  (unified; was inconsistent across files).
FIX-2  MAML inner loop uses torch.func.functional_call (see artemis_innovations).
FIX-3  EWC λ = 1000 (paper value; was mistakenly 5000 in the original model.py).
FIX-4  Problem formulation Eq. 1 restated: perturbation δ applies to the
       node FEATURE VECTOR x_v ∈ ℝ^d, not to the discrete address index v.
       The objective is:
           min_θ  E_{(x_v,y)~D}  max_{‖δ‖_∞≤ε}  L(f_θ(x_v + δ), y)
                                               + λ R(θ, θ*)
FIX-5  train/val/test split documented as 70/10/20 matching paper Section 6.1
       (the 70/15/15 in the old download_etgraph.py was wrong).

Author: ARTEMIS Research Team
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_max_pool, global_mean_pool

from artemis_innovations import (
    AnomalyAwareMemory,
    CertifiedAdversarialTrainer,
    ElasticWeightConsolidation,
    MultiHopBroadcast,
    TemporalODEBlock,
    _timed,
)


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _maybe_spectral(linear: nn.Linear, use: bool) -> nn.Module:
    return nn.utils.spectral_norm(linear) if use else linear


class SpectralNormLinear(nn.Module):
    """Linear with optional spectral normalisation (Lipschitz ≤ 1)."""

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, use_sn: bool = True) -> None:
        super().__init__()
        lin = nn.Linear(in_features, out_features, bias=bias)
        self.linear = nn.utils.spectral_norm(lin) if use_sn else lin

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.linear(x)


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Graph Encoder  (L1 + L3)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalGraphEncoder(nn.Module):
    """
    Encodes a temporal transaction graph into node embeddings.

    Pipeline:
        x_v  →  [input_proj]  →  GAT×2  →  [Neural ODE]  →  [Multi-Hop]  →  h_v

    FIX-1: ODE tolerances delegated to TemporalODEBlock.RTOL / ATOL constants
           (rtol=1e-4, atol=1e-5).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        ode_method: str = "dopri5",
        broadcast_hops: int = 3,
        use_spectral_norm: bool = True,
        use_ode: bool = True,
        use_multihop: bool = True,
    ) -> None:
        super().__init__()
        self.use_ode      = use_ode
        self.use_multihop = use_multihop

        self.input_proj = SpectralNormLinear(
            in_channels, hidden_channels, use_sn=use_spectral_norm
        )

        # Two-layer GAT
        head_dim = max(hidden_channels // num_heads, 1)
        self.gat1 = GATConv(hidden_channels, head_dim, heads=num_heads,
                            dropout=dropout, concat=True)
        self.gat2 = GATConv(hidden_channels, head_dim, heads=num_heads,
                            dropout=dropout, concat=True)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.drop  = nn.Dropout(dropout)

        # L1 — Neural ODE (FIX-1: tolerances inside TemporalODEBlock)
        if use_ode:
            self.ode_block = TemporalODEBlock(
                hidden_channels, method=ode_method,
                use_spectral_norm=use_spectral_norm,
            )

        # L3 — Multi-hop broadcast
        if use_multihop:
            self.broadcast = MultiHopBroadcast(
                hidden_channels, num_hops=broadcast_hops, aggregation="attention"
            )

    @_timed("encoder")
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = F.gelu(self.input_proj(x))

        h = self.norm1(F.gelu(self.gat1(h, edge_index)))
        h = self.drop(h)
        h = self.norm2(F.gelu(self.gat2(h, edge_index)))

        if self.use_ode:
            h = self.ode_block(h)

        if self.use_multihop:
            h = self.broadcast(h, edge_index, batch)

        return h


# ─────────────────────────────────────────────────────────────────────────────
# ARTEMIS full model
# ─────────────────────────────────────────────────────────────────────────────

class ARTEMIS(nn.Module):
    """
    ARTEMIS: Adversarial-Resistant Temporal Embedding Model for
    Intelligent Security in Blockchain Fraud Detection.

    Six innovations (L1–L6):
        L1  Neural ODE continuous-time temporal modelling
        L2  Anomaly-aware priority-queue memory
        L3  Multi-hop message broadcast (Sybil resistance)
        L4  Adversarial MAML meta-learning
        L5  Elastic Weight Consolidation  (λ=1000, FIX-3)
        L6  Certified adversarial training (randomised smoothing)

    Perturbation domain (FIX-4):
        Adversarial perturbation δ is applied to the node feature vector
        x_v ∈ ℝ^d (not to the discrete Ethereum address index).
        All PGD attacks and certified-smoothing calls operate in feature space.

    Data splits (FIX-5): train 70 % / val 10 % / test 20 %.
    """

    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.config = config

        # Dimensions
        in_ch  = config.get("in_channels", 32)
        hid_ch = config.get("hidden_channels", 128)
        n_cls  = config.get("num_classes", 2)

        # Innovation flags
        self.use_ode            = config.get("use_ode",            True)
        self.use_anomaly_memory = config.get("use_anomaly_memory", True)
        self.use_multihop       = config.get("use_multihop",       True)
        self.use_ewc            = config.get("use_ewc",            True)
        self.use_certified      = config.get("use_certified",      True)

        use_sn = self.use_certified   # spectral-norm only when certification needed

        # L1 + L3 inside encoder
        self.encoder = TemporalGraphEncoder(
            in_channels=in_ch,
            hidden_channels=hid_ch,
            num_heads=config.get("num_heads", 4),
            dropout=config.get("dropout", 0.1),
            ode_method=config.get("ode_method", "dopri5"),
            broadcast_hops=config.get("broadcast_hops", 3),
            use_spectral_norm=use_sn,
            use_ode=self.use_ode,
            use_multihop=self.use_multihop,
        )

        # L2 — anomaly memory
        self.memory = (
            AnomalyAwareMemory(
                memory_size=config.get("memory_size", 1000),
                embedding_dim=hid_ch,
                anomaly_weight=config.get("anomaly_weight", 1.0),
            )
            if self.use_anomaly_memory else None
        )

        # Readout MLP (mean + max pool → classification)
        self.readout = nn.Sequential(
            SpectralNormLinear(hid_ch * 2, hid_ch, use_sn=use_sn),
            nn.GELU(),
            nn.Dropout(config.get("dropout", 0.1)),
            SpectralNormLinear(hid_ch, hid_ch // 2, use_sn=use_sn),
            nn.GELU(),
        )
        self.classifier = SpectralNormLinear(hid_ch // 2, n_cls, use_sn=use_sn)

        # L5 — EWC (FIX-3: λ=1000)
        self.ewc = (
            ElasticWeightConsolidation(
                self, ewc_lambda=config.get("ewc_lambda", ElasticWeightConsolidation.DEFAULT_LAMBDA)
            )
            if self.use_ewc else None
        )

        # L6 — certified training
        self.cert_trainer = (
            CertifiedAdversarialTrainer(
                self,
                epsilon=config.get("adv_epsilon", 0.1),
                sigma=config.get("smoothing_sigma", 0.25),
                n_samples=config.get("smoothing_samples", 100),
                pgd_steps=config.get("pgd_steps", 20),
            )
            if self.use_certified else None
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode graph → node embeddings (h_v)."""
        h = self.encoder(x, edge_index, edge_attr, timestamps, batch)
        if self.memory is not None:
            h = self.memory.update_and_query(h, batch)
        return h

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    @_timed("forward")
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> torch.Tensor:
        """
        Graph-level classification forward pass.

        Args:
            x           : Node feature matrix [N, d].
                          FIX-4: adversarial δ is added here, not to address IDs.
            edge_index  : Edge index [2, E].
            edge_attr   : Optional edge features.
            timestamps  : Optional edge timestamps.
            batch       : Batch vector [N].
            return_embeddings: If True, also return graph-level embedding.

        Returns:
            logits [B, num_classes] or (logits, embedding) if return_embeddings.
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = self.encode(x, edge_index, edge_attr, timestamps, batch)

        h_mean  = global_mean_pool(h, batch)
        h_max   = global_max_pool(h, batch)
        h_graph = torch.cat([h_mean, h_max], dim=-1)

        h_out  = self.readout(h_graph)
        logits = self.classifier(h_out)

        return (logits, h_out) if return_embeddings else logits

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_loss(
        self, batch_data, task_id: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Total training loss:
            L = L_task + L_ewc + w_adv · L_adv
        """
        logits    = self(batch_data.x, batch_data.edge_index,
                        getattr(batch_data, "edge_attr",  None),
                        getattr(batch_data, "timestamps", None),
                        batch_data.batch)
        loss_task = F.cross_entropy(logits, batch_data.y)
        info      = {"task": float(loss_task)}
        total     = loss_task

        if self.ewc is not None and task_id is not None and task_id > 0:
            l_ewc = self.ewc.penalty()
            total = total + l_ewc
            info["ewc"] = float(l_ewc)

        if self.cert_trainer is not None and self.training:
            l_adv = self.cert_trainer.adversarial_loss(batch_data)
            total = total + self.config.get("adv_weight", 0.5) * l_adv
            info["adversarial"] = float(l_adv)

        info["total"] = float(total)
        return total, info

    # ------------------------------------------------------------------
    # Certification
    # ------------------------------------------------------------------

    def certify(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        n_samples: int = 1000,
        alpha: float = 0.001,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cert_trainer is None:
            raise ValueError("Certified training (L6) is not enabled.")
        return self.cert_trainer.certify(x, edge_index, batch, n_samples, alpha)

    def update_ewc(self, dataloader, task_id: int) -> None:
        if self.ewc is not None:
            self.ewc.update_fisher(dataloader, task_id)

    def get_lipschitz_bound(self) -> float:
        """Product of spectral norms across all spec-normalised layers."""
        bound = 1.0
        for m in self.modules():
            if hasattr(m, "weight_u"):
                with torch.no_grad():
                    sigma = torch.dot(m.weight_u, torch.mv(m.weight_orig, m.weight_v))
                bound *= float(sigma)
        return bound


# ─────────────────────────────────────────────────────────────────────────────
# Node-level variant
# ─────────────────────────────────────────────────────────────────────────────

class ARTEMISNodeClassifier(ARTEMIS):
    """Node-level phishing address classification (no graph pooling)."""

    @_timed("forward_node")
    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        target_nodes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.encode(x, edge_index, edge_attr, timestamps, batch)
        if target_nodes is not None:
            h = h[target_nodes]
        # Duplicate to match readout input width (expects h_mean ‖ h_max)
        h_cat  = torch.cat([h, h], dim=-1)
        h_out  = self.readout(h_cat)
        return self.classifier(h_out)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_artemis(config: Dict) -> ARTEMIS:
    """Build ARTEMIS (graph or node variant) from a config dict."""
    if config.get("model_type", "graph") == "node":
        return ARTEMISNodeClassifier(config)
    return ARTEMIS(config)
