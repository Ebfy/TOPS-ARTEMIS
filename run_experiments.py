#!/usr/bin/env python3
"""
ARTEMIS Unified Experiment Runner — IEEE TDSC Revision
=======================================================

Runs all experiments required for the TDSC submission in one script:

    Mode 1: --mode main       Table 3  — detection performance (both datasets)
    Mode 2: --mode ablation   Table 4  — component contribution
    Mode 3: --mode adversarial Table 5 — adversarial robustness
    Mode 4: --mode efficiency  Table 7 — computational overhead (per-component)
    Mode 5: --mode all         All of the above sequentially

Key fixes vs CCS submission
----------------------------
FIX-1  ODE tolerances unified (rtol=1e-4, atol=1e-5).
FIX-2  MAML inner loop uses torch.func.functional_call.
FIX-3  EWC lambda = 1000 everywhere.
FIX-4  Perturbation δ applied to node features x_v (clarified in output).
FIX-5  Train/val/test = 70/10/20 (temporal split, matches paper Section 6.1).
NEW-A  Elliptic Bitcoin dataset evaluated alongside ETGraph.
NEW-B  Per-component latency breakdown in efficiency table.

Usage
-----
  python run_experiments.py --mode all --dataset etgraph --data_dir ./data
  python run_experiments.py --mode main --dataset elliptic --data_dir ./data
  python run_experiments.py --mode ablation --dataset etgraph --quick
  python run_experiments.py --mode adversarial --epsilon 0.1 0.2

Author: ARTEMIS Research Team
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from scipy import stats

# ── src path resolution ──────────────────────────────────────────────────────
SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

from artemis_model import build_artemis
from artemis_innovations import (
    LATENCY_TRACKER, LatencyTracker,
    ElasticWeightConsolidation,
    CertifiedAdversarialTrainer,
)
from data_loader import load_dataset, get_num_features
from baseline_implementations import build_baseline


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray) -> Dict[str, float]:
    """Full metric suite matching TDSC paper tables."""
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())

    recall    = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-10)
    fpr       = fp / max(fp + tn, 1)
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = 0.5

    return {
        "recall":    recall,
        "precision": precision,
        "f1":        f1,
        "auc":       auc,
        "fpr":       fpr,
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def significance_test(a_scores: List[float],
                       b_scores: List[float]) -> Dict[str, float]:
    """Paired t-test + Wilcoxon + Cohen's d."""
    a, b   = np.array(a_scores), np.array(b_scores)
    t, p   = stats.ttest_rel(a, b)
    diff   = a - b
    cohens = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-10))
    try:
        _, wp = stats.wilcoxon(a, b)
    except Exception:
        wp = float("nan")
    return {"t": float(t), "p_value": float(p),
            "cohens_d": cohens, "wilcoxon_p": float(wp)}


# ─────────────────────────────────────────────────────────────────────────────
# Training / evaluation
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """Manages training, validation, evaluation for any model."""

    def __init__(self, model: nn.Module, config: Dict, device: torch.device) -> None:
        self.model  = model.to(device)
        self.config = config
        self.device = device

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get("learning_rate", 1e-3),
            weight_decay=config.get("weight_decay", 1e-2),
            betas=(config.get("beta1", 0.9), config.get("beta2", 0.999)),
        )
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt,
            T_max=config.get("epochs", 100),
            eta_min=config.get("min_lr", 1e-6),
        )
        self.scaler = (
            torch.cuda.amp.GradScaler()
            if config.get("mixed_precision", True) and torch.cuda.is_available()
            else None
        )

    def _step(self, batch):
        batch = batch.to(self.device)
        logits = self.model(
            batch.x, batch.edge_index,
            edge_attr=getattr(batch, "edge_attr",  None),
            timestamps=getattr(batch, "timestamps", None),
            batch=batch.batch,
        )
        return F.cross_entropy(logits, batch.y), logits

    def train_epoch(self, loader) -> float:
        self.model.train()
        total_loss, n = 0.0, 0
        for batch in loader:
            self.opt.zero_grad()
            if self.scaler:
                with torch.cuda.amp.autocast():
                    loss, _ = self._step(batch)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.get("clip_grad", 1.0))
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                loss, _ = self._step(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.get("clip_grad", 1.0))
                self.opt.step()
            total_loss += float(loss) * int(batch.y.size(0))
            n          += int(batch.y.size(0))
        self.sched.step()
        return total_loss / max(n, 1)

    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, float]:
        self.model.eval()
        y_true, y_pred, y_prob = [], [], []
        for batch in loader:
            batch  = batch.to(self.device)
            logits = self.model(
                batch.x, batch.edge_index,
                edge_attr=getattr(batch, "edge_attr",  None),
                timestamps=getattr(batch, "timestamps", None),
                batch=batch.batch,
            )
            prob = F.softmax(logits, dim=-1)[:, -1]
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(logits.argmax(-1).cpu().numpy())
            y_prob.extend(prob.cpu().numpy())
        return compute_metrics(
            np.array(y_true), np.array(y_pred), np.array(y_prob)
        )

    def fit(self, train_loader, val_loader) -> Dict:
        best_f1, patience_ctr, best_state = 0.0, 0, None
        epochs   = self.config.get("epochs", 100)
        patience = self.config.get("patience", 15)
        for epoch in range(1, epochs + 1):
            loss = self.train_epoch(train_loader)
            vm   = self.evaluate(val_loader)
            if vm["f1"] > best_f1 + 1e-5:
                best_f1      = vm["f1"]
                patience_ctr = 0
                best_state   = deepcopy(self.model.state_dict())
            else:
                patience_ctr += 1
            if patience_ctr >= patience:
                break
        if best_state:
            self.model.load_state_dict(best_state)
        return {"best_val_f1": best_f1, "epochs_trained": epoch}


def train_and_eval(
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    config: Dict,
    device: torch.device,
    seed: int = 42,
) -> Tuple[Dict[str, float], float, nn.Module]:
    """Train model and return (test_metrics, wall_clock_seconds, trained_model).

    The trained_model is always the unwrapped module (never DataParallel),
    so callers can safely call model.certify() and model.update_ewc().
    """
    set_seed(seed)
    trainer = Trainer(model, config, device)
    t0      = time.perf_counter()
    trainer.fit(train_loader, val_loader)
    train_s = time.perf_counter() - t0
    metrics = trainer.evaluate(test_loader)

    # Unwrap DataParallel so the caller always gets the plain module
    trained = trainer.model
    if isinstance(trained, nn.DataParallel):
        trained = trained.module

    return metrics, train_s, trained


# ─────────────────────────────────────────────────────────────────────────────
# Mode 1 — Main performance  (Table 3)
# ─────────────────────────────────────────────────────────────────────────────

def run_main(args, config: Dict, device: torch.device) -> Dict:
    """Compare ARTEMIS vs all baselines on the chosen dataset (5 seeds)."""
    print(f"\n{'='*65}")
    print(f"TABLE 3 — Main Performance  [{args.dataset.upper()}]")
    print(f"{'='*65}")

    train_l, val_l, test_l = load_dataset(
        args.dataset, args.data_dir, config["batch_size"], task_id=args.task
    )

    n_feat = get_num_features(args.dataset)
    config["in_channels"] = n_feat

    seeds    = [42, 123, 456, 789, 1011] if not args.quick else [42]
    methods  = {
        "ARTEMIS":    lambda: build_artemis(config),
        "2DynEthNet": lambda: build_baseline("2dynethnet", config),
        "GrabPhisher": lambda: build_baseline("grabphisher", config),
        "TGAT":       lambda: build_baseline("tgat",        config),
        "TGN":        lambda: build_baseline("tgn",         config),
        "GAT":        lambda: build_baseline("gat",         config),
        "GraphSAGE":  lambda: build_baseline("graphsage",   config),
    }

    all_results: Dict[str, List[Dict]] = {m: [] for m in methods}

    for name, model_fn in methods.items():
        print(f"\n  {name}:")
        for seed in seeds:
            m, _, _ = train_and_eval(model_fn(), train_l, val_l, test_l,
                                   config, device, seed)
            all_results[name].append(m)
            print(f"    seed={seed}  recall={m['recall']:.4f}  "
                  f"f1={m['f1']:.4f}  auc={m['auc']:.4f}")

    # Summary table
    print(f"\n{'─'*65}")
    print(f"{'Method':<18} {'Recall':<16} {'F1':<16} {'AUC':<16}")
    print(f"{'─'*65}")
    artemis_recalls = [r["recall"] for r in all_results["ARTEMIS"]]
    for name, runs in all_results.items():
        recalls = [r["recall"] for r in runs]
        f1s     = [r["f1"]     for r in runs]
        aucs    = [r["auc"]    for r in runs]
        sig     = significance_test(artemis_recalls, recalls) if name != "ARTEMIS" else {}
        p_str   = f"  p={sig['p_value']:.3f}  d={sig['cohens_d']:.2f}" if sig else ""
        print(
            f"  {name:<16} "
            f"{np.mean(recalls):.4f}±{np.std(recalls):.4f}  "
            f"{np.mean(f1s):.4f}±{np.std(f1s):.4f}  "
            f"{np.mean(aucs):.4f}±{np.std(aucs):.4f}"
            f"{p_str}"
        )
    print(f"{'─'*65}")
    print(f"\n[FIX-4] Perturbation δ applied to node feature vectors x_v ∈ ℝ^d")
    print(f"[FIX-5] Split: train 70% / val 10% / test 20% (temporal)\n")

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Mode 2 — Ablation  (Table 4)
# ─────────────────────────────────────────────────────────────────────────────

ABLATION_VARIANTS = {
    "full":              dict(use_ode=True,  use_anomaly_memory=True,  use_multihop=True,
                              use_ewc=True,  use_certified=True),
    "no_ode":            dict(use_ode=False, use_anomaly_memory=True,  use_multihop=True,
                              use_ewc=True,  use_certified=True),
    "no_anomaly_memory": dict(use_ode=True,  use_anomaly_memory=False, use_multihop=True,
                              use_ewc=True,  use_certified=True),
    "no_multihop":       dict(use_ode=True,  use_anomaly_memory=True,  use_multihop=False,
                              use_ewc=True,  use_certified=True),
    "no_ewc":            dict(use_ode=True,  use_anomaly_memory=True,  use_multihop=True,
                              use_ewc=False, use_certified=True),
    "no_certified":      dict(use_ode=True,  use_anomaly_memory=True,  use_multihop=True,
                              use_ewc=True,  use_certified=False),
}

_VARIANT_LABELS = {
    "full":              "Full ARTEMIS",
    "no_ode":            "w/o Neural ODE (L1)",
    "no_anomaly_memory": "w/o Anomaly Memory (L2)",
    "no_multihop":       "w/o Multi-Hop (L3)",
    "no_ewc":            "w/o EWC (L5)",
    "no_certified":      "w/o Certified (L6)",
}


def run_ablation(args, config: Dict, device: torch.device) -> Dict:
    print(f"\n{'='*65}")
    print(f"TABLE 4 — Ablation Study  [{args.dataset.upper()}]")
    print(f"{'='*65}")

    train_l, val_l, test_l = load_dataset(
        args.dataset, args.data_dir, config["batch_size"], task_id=args.task
    )
    n_feat = get_num_features(args.dataset)
    seeds  = [42, 123, 456] if not args.quick else [42]

    results: Dict[str, List[Dict]] = {}
    for key, flags in ABLATION_VARIANTS.items():
        cfg = deepcopy(config)
        cfg["in_channels"] = n_feat
        cfg.update(flags)

        results[key] = []
        print(f"\n  {_VARIANT_LABELS[key]}")
        for seed in seeds:
            m, _, _ = train_and_eval(build_artemis(cfg), train_l, val_l, test_l,
                                   cfg, device, seed)
            results[key].append(m)
            print(f"    seed={seed}  recall={m['recall']:.4f}  f1={m['f1']:.4f}  "
                  f"certified={m.get('certified', 'N/A')}")

    # Summary table
    full_recall = np.mean([r["recall"] for r in results["full"]])
    print(f"\n{'─'*65}")
    print(f"{'Variant':<30} {'Recall':<16} {'F1':<16} {'Δ Recall':>10}")
    print(f"{'─'*65}")
    for key, label in _VARIANT_LABELS.items():
        if key not in results:
            continue
        runs    = results[key]
        rec     = np.mean([r["recall"] for r in runs])
        rec_std = np.std([r["recall"]  for r in runs])
        f1      = np.mean([r["f1"]     for r in runs])
        f1_std  = np.std([r["f1"]      for r in runs])
        delta   = rec - full_recall if key != "full" else 0.0
        d_str   = f"{delta:+.4f}" if key != "full" else "—"
        print(f"  {label:<28} {rec:.4f}±{rec_std:.4f}  {f1:.4f}±{f1_std:.4f}  {d_str:>10}")

    print(f"{'─'*65}")
    print("\n[TDSC note] EWC / Certified gains are measured as dependability")
    print("  properties (forgetting resistance; certified accuracy) not F1.")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Mode 3 — Adversarial robustness  (Table 5)
# ─────────────────────────────────────────────────────────────────────────────

def _pgd_attack(model, batch, epsilon: float, steps: int, device: torch.device):
    """Standalone PGD for evaluation (does not require CertifiedAdversarialTrainer)."""
    model.eval()
    x_adv = batch.x.detach().clone().to(device)
    step_size = epsilon / 4.0
    for _ in range(steps):
        x_adv = x_adv.requires_grad_(True)
        logits = model(x_adv, batch.edge_index.to(device), batch=batch.batch.to(device))
        loss   = F.cross_entropy(logits, batch.y.to(device))
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + step_size * x_adv.grad.sign()
            delta  = (x_adv - batch.x.to(device)).clamp(-epsilon, epsilon)
            x_adv  = (batch.x.to(device) + delta).detach()
    return x_adv


@torch.no_grad()
def _eval_adv(model, loader, epsilon: float, steps: int, device: torch.device) -> Dict:
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for batch in loader:
        x_adv  = _pgd_attack(model, batch, epsilon, steps, device)
        logits = model(x_adv, batch.edge_index.to(device), batch=batch.batch.to(device))
        prob   = F.softmax(logits, dim=-1)[:, -1]
        y_true.extend(batch.y.cpu().numpy())
        y_pred.extend(logits.argmax(-1).cpu().numpy())
        y_prob.extend(prob.detach().cpu().numpy())
    return compute_metrics(np.array(y_true), np.array(y_pred), np.array(y_prob))


def run_adversarial(args, config: Dict, device: torch.device) -> Dict:
    print(f"\n{'='*65}")
    print(f"TABLE 5 — Adversarial Robustness  [{args.dataset.upper()}]")
    print(f"{'='*65}")

    train_l, val_l, test_l = load_dataset(
        args.dataset, args.data_dir, config["batch_size"], task_id=args.task
    )
    n_feat = get_num_features(args.dataset)
    config["in_channels"] = n_feat

    epsilons = args.epsilon or [0.05, 0.10, 0.15, 0.20]
    methods  = {
        "ARTEMIS":    build_artemis(deepcopy(config)),
        "2DynEthNet": build_baseline("2dynethnet", config),
    }

    # Train both models — store the returned trained (unwrapped) model
    trained: Dict[str, nn.Module] = {}
    for name, model in methods.items():
        print(f"\n  Training {name} …")
        _, _, trained_model = train_and_eval(model, train_l, val_l, test_l,
                                             config, device)
        trained[name] = trained_model

    all_results: Dict = {}
    for name, model in trained.items():
        all_results[name] = {}
        # Clean evaluation
        clean = Trainer(model, config, device).evaluate(test_l)
        all_results[name]["clean"] = clean
        # PGD attacks
        for eps in epsilons:
            for steps in [10, 20]:
                key = f"pgd{steps}_eps{eps}"
                m   = _eval_adv(model, test_l, eps, steps, device)
                all_results[name][key] = m
                print(f"  {name}  PGD-{steps} ε={eps}  "
                      f"recall={m['recall']:.4f}  f1={m['f1']:.4f}")

        # Certified accuracy (ARTEMIS only)
        if name == "ARTEMIS" and hasattr(model, "certify"):
            print(f"\n  Computing certified accuracy …")
            for eps in epsilons:
                n_cert, n_total = 0, 0
                for batch in test_l:
                    preds, radii = model.certify(
                        batch.x.to(device),
                        batch.edge_index.to(device),
                        batch.batch.to(device),
                        n_samples=100, alpha=0.001,
                    )
                    n_cert  += int(((preds.cpu() == batch.y) & (radii.cpu() >= eps)).sum())
                    n_total += int(batch.y.size(0))
                cert_acc = n_cert / max(n_total, 1)
                all_results[name][f"certified_eps{eps}"] = cert_acc
                print(f"  ARTEMIS  certified acc ε={eps}: {cert_acc:.4f}")

    # Summary
    print(f"\n{'─'*65}")
    for eps in epsilons:
        key20 = f"pgd20_eps{eps}"
        a_r   = all_results["ARTEMIS"].get(key20, {}).get("recall", 0)
        b_r   = all_results["2DynEthNet"].get(key20, {}).get("recall", 0)
        cert  = all_results["ARTEMIS"].get(f"certified_eps{eps}", "—")
        print(f"  ε={eps}  ARTEMIS recall={a_r:.4f}  "
              f"2DynEthNet recall={b_r:.4f}  "
              f"certified={cert if isinstance(cert, str) else f'{cert:.4f}'}")
    print(f"{'─'*65}")

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Mode 4 — Efficiency  (Table 7, per-component latency — NEW-B)
# ─────────────────────────────────────────────────────────────────────────────

def run_efficiency(args, config: Dict, device: torch.device) -> Dict:
    """
    Measures per-model AND per-component latency (NEW-B).

    Per-component breakdown (ARTEMIS only):
        ode_solve, memory_update, memory_query, multihop,
        pgd_attack, adv_loss, ewc_penalty, certify
    """
    import gc
    print(f"\n{'='*65}")
    print(f"TABLE 7 — Efficiency Analysis  [{args.dataset.upper()}]")
    print(f"{'='*65}")

    n_feat = get_num_features(args.dataset)
    config["in_channels"] = n_feat

    _, _, test_l = load_dataset(
        args.dataset, args.data_dir, config["batch_size"], task_id=args.task
    )

    methods: Dict[str, nn.Module] = {
        "ARTEMIS":    build_artemis(deepcopy(config)),
        "2DynEthNet": build_baseline("2dynethnet", config),
        "TGN":        build_baseline("tgn",         config),
        "TGAT":       build_baseline("tgat",        config),
        "GAT":        build_baseline("gat",         config),
        "GraphSAGE":  build_baseline("graphsage",   config),
    }

    n_warmup = 5
    n_runs   = 50 if not args.quick else 10

    all_eff: Dict = {}

    for name, model in methods.items():
        model.to(device).eval()
        params = sum(p.numel() for p in model.parameters())

        # Warm up
        for i, batch in enumerate(test_l):
            if i >= n_warmup:
                break
            with torch.no_grad():
                _ = model(batch.x.to(device), batch.edge_index.to(device),
                          batch=batch.batch.to(device))

        # Timed inference
        latencies_ms = []
        for i, batch in enumerate(test_l):
            if i >= n_runs:
                break
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(batch.x.to(device), batch.edge_index.to(device),
                          batch=batch.batch.to(device))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        # GPU memory
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            for i, batch in enumerate(test_l):
                if i >= 3:
                    break
                _ = model(batch.x.to(device), batch.edge_index.to(device),
                          batch=batch.batch.to(device))
            mem_mb = torch.cuda.max_memory_allocated(device) / 1e6
        else:
            mem_mb = 0.0

        eff = {
            "params_M":    params / 1e6,
            "latency_mean_ms": float(np.mean(latencies_ms)),
            "latency_std_ms":  float(np.std(latencies_ms)),
            "throughput_per_s": 1000.0 / max(np.mean(latencies_ms), 1e-3),
            "peak_mem_mb": mem_mb,
        }
        all_eff[name] = eff

        print(f"  {name:<16}  params={params/1e6:.2f}M  "
              f"latency={eff['latency_mean_ms']:.2f}±{eff['latency_std_ms']:.2f} ms  "
              f"mem={mem_mb:.0f} MB")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Per-component ARTEMIS breakdown (NEW-B via LatencyTracker)
    import artemis_innovations as _ai_module
    tracker = LatencyTracker()
    _ai_module.LATENCY_TRACKER = tracker

    print(f"\n  --- Per-component latency (ARTEMIS) ---")
    art = build_artemis(deepcopy(config)).to(device).eval()
    for i, batch in enumerate(test_l):
        if i >= n_runs:
            break
        with torch.no_grad():
            _ = art(batch.x.to(device), batch.edge_index.to(device),
                    batch=batch.batch.to(device))

    comp_summary = tracker.summary()
    for comp, stats_c in comp_summary.items():
        print(f"    {comp:<22} {stats_c['mean_ms']:.3f} ± {stats_c['std_ms']:.3f} ms")
    all_eff["ARTEMIS"]["per_component_ms"] = {k: v["mean_ms"] for k, v in comp_summary.items()}

    _ai_module.LATENCY_TRACKER = None   # disable tracker

    # Summary table
    print(f"\n{'─'*65}")
    print(f"{'Method':<18} {'Params(M)':<12} {'Latency(ms)':<18} {'Peak Mem(MB)'}")
    print(f"{'─'*65}")
    for name, e in all_eff.items():
        print(f"  {name:<16} {e.get('params_M',0):<12.2f} "
              f"{e['latency_mean_ms']:.2f}±{e['latency_std_ms']:.2f}  "
              f"{e.get('peak_mem_mb',0):.0f}")
    print(f"{'─'*65}")

    return all_eff


# ─────────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────────

def _serialise(obj):
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialise(v) for v in obj]
    return obj


def save_results(results: Dict, output_dir: str, tag: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(output_dir) / f"{tag}_{ts}.json"
    with open(path, "w") as f:
        json.dump(_serialise(results), f, indent=2)
    print(f"\n  Results saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_default_config(args) -> Dict:
    """Build a sensible default config that can be overridden by a YAML file."""
    cfg: Dict = {
        # Architecture
        "in_channels":      32,
        "hidden_channels":  128,
        "num_classes":      2,
        "num_heads":        4,
        "dropout":          0.1,
        "broadcast_hops":   3,
        "memory_size":      1000,
        # ODE (FIX-1: canonical values)
        "ode_method":       "dopri5",
        # EWC (FIX-3: matches paper λ=1000)
        "ewc_lambda":       1000,
        # Adversarial
        "adv_epsilon":      0.1,
        "smoothing_sigma":  0.25,
        "smoothing_samples": 100,
        "pgd_steps":        20,
        "adv_weight":       0.5,
        # Training
        "learning_rate":    1e-3,
        "weight_decay":     1e-2,
        "beta1":            0.9,
        "beta2":            0.999,
        "min_lr":           1e-6,
        "epochs":           5 if getattr(args, "quick", False) else 100,
        "patience":         15,
        "clip_grad":        1.0,
        "batch_size":       32,
        "mixed_precision":  True,
    }
    # Load YAML overrides
    cfg_path = Path(getattr(args, "config", "configs/default.yaml"))
    if cfg_path.exists():
        import yaml
        with open(cfg_path) as f:
            override = yaml.safe_load(f) or {}
        if "model" in override:
            override = {**override, **override["model"]}
        cfg.update(override)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ARTEMIS Unified Experiment Runner (IEEE TDSC revision)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode",     choices=["main", "ablation", "adversarial",
                                               "efficiency", "all"],
                        default="all")
    parser.add_argument("--dataset",  choices=["etgraph", "elliptic", "synthetic"],
                        default="etgraph",
                        help="Dataset to evaluate on (use both for TDSC submission)")
    parser.add_argument("--data_dir", default="./data",   help="Root data directory")
    parser.add_argument("--output",   default="./results", help="Results output directory")
    parser.add_argument("--config",   default="configs/default.yaml")
    parser.add_argument("--task",     type=int, default=1, help="ETGraph task ID (1-6)")
    parser.add_argument("--quick",    action="store_true", help="Smoke-test (minimal data)")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--epsilon",  type=float, nargs="+",
                        default=[0.05, 0.10, 0.15, 0.20],
                        help="Perturbation budgets for adversarial eval")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = build_default_config(args)

    print("=" * 65)
    print("ARTEMIS — IEEE TDSC Experiment Runner")
    print(f"  Dataset : {args.dataset.upper()}")
    print(f"  Mode    : {args.mode}")
    print(f"  Device  : {device}")
    print(f"  Quick   : {args.quick}")
    print(f"  FIX-1   : ODE rtol=1e-4 atol=1e-5")
    print(f"  FIX-3   : EWC λ={config['ewc_lambda']}")
    print(f"  FIX-5   : Split 70/10/20 (temporal)")
    print("=" * 65)

    modes_to_run = (
        ["main", "ablation", "adversarial", "efficiency"]
        if args.mode == "all" else [args.mode]
    )

    all_outputs: Dict = {}
    for mode in modes_to_run:
        cfg = deepcopy(config)
        if mode == "main":
            out = run_main(args, cfg, device)
        elif mode == "ablation":
            out = run_ablation(args, cfg, device)
        elif mode == "adversarial":
            out = run_adversarial(args, cfg, device)
        elif mode == "efficiency":
            out = run_efficiency(args, cfg, device)
        else:
            continue
        all_outputs[mode] = out
        save_results(out, args.output, f"{mode}_{args.dataset}")

    print("\n✓  All requested experiments complete.")


if __name__ == "__main__":
    main()
