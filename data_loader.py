"""
ARTEMIS Data Loader — Revised for IEEE TDSC Submission
=======================================================

Supports two datasets (addresses Reviewer B's generalisability concern):

  1. ETGraph  — Ethereum phishing detection (original paper dataset).
               847 M transactions, 12.4 M addresses, 23 847 phishing labels.
               Source: https://xblock.pro/#/dataset/68

  2. Elliptic Bitcoin Dataset — illicit Bitcoin transaction detection.
               203 769 transactions (nodes), 234 355 edges, 49 classes
               collapsed to binary (illicit / licit).
               Source: https://www.kaggle.com/ellipticco/elliptic-data-set
               Reference: Weber et al. (2019) arXiv:1908.02591

Split (FIX-5, matching paper Section 6.1):
    Train 70 % | Validation 10 % | Test 20 %
    Temporal ordering is preserved — future data is never seen during training.

Synthetic fallback:
    If neither dataset is present, a structurally similar synthetic graph
    is generated so the pipeline can be smoke-tested without downloads.

Author: ARTEMIS Research Team
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Split constants  (FIX-5)
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_FRAC = 0.70   # Paper Section 6.1 — temporal split
VAL_FRAC   = 0.10
TEST_FRAC  = 0.20


# ─────────────────────────────────────────────────────────────────────────────
# Generic helpers
# ─────────────────────────────────────────────────────────────────────────────

def _temporal_split(graphs: List[Data]) -> Tuple[List[Data], List[Data], List[Data]]:
    """
    Temporally ordered 70/10/20 split.
    Graphs are assumed to be sorted by their first timestamp (if available)
    or by list order (proxy for temporal order).
    """
    n     = len(graphs)
    t_val  = int(n * TRAIN_FRAC)
    t_test = int(n * (TRAIN_FRAC + VAL_FRAC))
    return graphs[:t_val], graphs[t_val:t_test], graphs[t_test:]


def _make_loaders(
    train: List[Data],
    val:   List[Data],
    test:  List[Data],
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True,  drop_last=False),
        DataLoader(val,   batch_size=batch_size, shuffle=False, drop_last=False),
        DataLoader(test,  batch_size=batch_size, shuffle=False, drop_last=False),
    )


# ─────────────────────────────────────────────────────────────────────────────
# ETGraph loader
# ─────────────────────────────────────────────────────────────────────────────

class ETGraphLoader:
    """
    Loads pre-processed ETGraph data from disk.

    Expected directory layout (produced by download_etgraph.py):
        <data_dir>/
            task<i>/
                processed/
                    graphs.pt          # list[Data], each graph = transaction ego-net
                    node_labels.json   # {address: label}  0=benign, 1=phishing
                splits.json           # {"train": [...], "val": [...], "test": [...]}

    Node feature dimension: 16 (as per ETGraph paper).
    Phishing label count (ground truth): 23 847  (FIX-4 in data notes).
    """

    N_PHISHING_GROUNDTRUTH = 23_847   # correct figure, Section 6.1

    def __init__(self, data_dir: str, task_id: int = 1, batch_size: int = 32) -> None:
        self.data_dir   = Path(data_dir)
        self.task_id    = task_id
        self.batch_size = batch_size

    def _task_dir(self) -> Path:
        return self.data_dir / f"task{self.task_id}"

    def load(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        task_dir = self._task_dir()
        graphs_path = task_dir / "processed" / "graphs.pt"

        if not graphs_path.exists():
            warnings.warn(
                f"ETGraph task {self.task_id} not found at {graphs_path}. "
                "Falling back to synthetic data.  Run download_etgraph.py first."
            )
            return SyntheticLoader(num_features=16, batch_size=self.batch_size).load()

        graphs: List[Data] = torch.load(str(graphs_path))

        # Check for pre-computed split file
        splits_path = task_dir / "splits.json"
        if splits_path.exists():
            with open(splits_path) as f:
                splits = json.load(f)
            train = [graphs[i] for i in splits["train"]]
            val   = [graphs[i] for i in splits["val"]]
            test  = [graphs[i] for i in splits["test"]]
        else:
            train, val, test = _temporal_split(graphs)

        print(
            f"[ETGraph task {self.task_id}]  "
            f"train={len(train):,}  val={len(val):,}  test={len(test):,}"
        )
        return _make_loaders(train, val, test, self.batch_size)

    @staticmethod
    def num_features() -> int:
        return 16


# ─────────────────────────────────────────────────────────────────────────────
# Elliptic Bitcoin Dataset loader  (NEW-A: second dataset for TDSC)
# ─────────────────────────────────────────────────────────────────────────────

class EllipticLoader:
    """
    Loads the Elliptic Bitcoin Dataset and constructs temporal subgraphs.

    Dataset statistics:
        - 203 769 transactions (nodes), 234 355 payment edges.
        - 49 time steps (each ~2 weeks of Bitcoin history).
        - Labels: 2 = illicit (phishing / scam / darknet), 1 = licit, 3 = unknown.
        - We map: illicit→1 (positive), licit→0 (negative), unknown nodes dropped.
        - Node features: 166 dims (93 local + 72 aggregated neighbourhood).

    Expected directory layout:
        <data_dir>/elliptic/
            elliptic_txs_features.csv   # (203769, 167): [txId, f1..f166]
            elliptic_txs_edgelist.csv   # (234355, 2): [txId1, txId2]
            elliptic_txs_classes.csv    # (203769, 2): [txId, class]

    Download: https://www.kaggle.com/ellipticco/elliptic-data-set

    Graph construction:
        One Data object per time step (49 total).
        Each time step's transactions become nodes; edges restricted to
        same-time-step pairs.  Temporal ordering preserved for 70/10/20 split.

    Reference: Weber et al. (2019). Anti-Money Laundering in Bitcoin:
        Experimenting with Graph Convolutional Networks for Financial Forensics.
        arXiv:1908.02591.
    """

    ILLICIT_LABEL   = 2
    LICIT_LABEL     = 1
    UNKNOWN_LABEL   = 3
    NUM_TIME_STEPS  = 49
    NUM_FEATURES    = 166

    def __init__(self, data_dir: str, batch_size: int = 32,
                 drop_unknown: bool = True) -> None:
        self.data_dir     = Path(data_dir) / "elliptic"
        self.batch_size   = batch_size
        self.drop_unknown = drop_unknown

    def _check_files(self) -> bool:
        required = [
            "elliptic_txs_features.csv",
            "elliptic_txs_edgelist.csv",
            "elliptic_txs_classes.csv",
        ]
        return all((self.data_dir / f).exists() for f in required)

    def _load_raw(self):
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("pandas is required for Elliptic loading: pip install pandas") from e

        feat_path  = self.data_dir / "elliptic_txs_features.csv"
        edge_path  = self.data_dir / "elliptic_txs_edgelist.csv"
        label_path = self.data_dir / "elliptic_txs_classes.csv"

        # Features: first column is txId, second is time step, rest are features
        feat_df  = pd.read_csv(feat_path, header=None)
        edge_df  = pd.read_csv(edge_path)
        label_df = pd.read_csv(label_path)

        # CSV layout: txId | time_step | f1 … f166  (168 columns total)
        # range(1, NUM_FEATURES+1) gives labels f1..f166 — do NOT use range(NUM_FEATURES-1)
        feat_df.columns  = ["txId", "time_step"] + [f"f{i}" for i in range(1, self.NUM_FEATURES + 1)]
        label_df.columns = ["txId", "class"]

        merged = feat_df.merge(label_df, on="txId", how="left")
        merged["class"] = merged["class"].fillna(str(self.UNKNOWN_LABEL)).astype(str)

        return merged, edge_df

    def _build_time_step_graph(
        self, step_df, edge_df
    ) -> Optional[Data]:
        """Build one PyG Data object for a single Elliptic time step."""
        try:
            import pandas as pd
        except ImportError:
            raise

        # Filter to this time step
        tx_ids = set(step_df["txId"].astype(str).tolist())

        # Build local index
        local_idx = {tx: i for i, tx in enumerate(sorted(tx_ids))}
        n = len(local_idx)
        if n == 0:
            return None

        # Node features
        feat_cols = [c for c in step_df.columns if c.startswith("f")]
        x = torch.tensor(
            step_df.set_index("txId")[feat_cols].values, dtype=torch.float
        )

        # Labels (drop unknown if requested)
        label_col = step_df["class"].astype(str)
        labels = []
        valid_mask = []
        for _, row in step_df.iterrows():
            cls = str(row["class"])
            if cls == str(self.ILLICIT_LABEL):
                labels.append(1)
                valid_mask.append(True)
            elif cls == str(self.LICIT_LABEL):
                labels.append(0)
                valid_mask.append(True)
            else:
                labels.append(-1)
                valid_mask.append(not self.drop_unknown)

        if self.drop_unknown:
            valid_idx = [i for i, v in enumerate(valid_mask) if v]
            if len(valid_idx) == 0:
                return None
            x = x[valid_idx]
            labels = [labels[i] for i in valid_idx]
            # Remap local indices
            remap = {old: new for new, old in enumerate(valid_idx)}
        else:
            remap = {i: i for i in range(n)}

        y = torch.tensor(labels, dtype=torch.long)

        # Edges restricted to this time step
        src_ids = edge_df.iloc[:, 0].astype(str)
        dst_ids = edge_df.iloc[:, 1].astype(str)
        edge_rows = [
            (local_idx.get(s), local_idx.get(d))
            for s, d in zip(src_ids, dst_ids)
            if s in local_idx and d in local_idx
        ]
        if edge_rows:
            src_list = [remap.get(e[0], -1) for e in edge_rows]
            dst_list = [remap.get(e[1], -1) for e in edge_rows]
            valid_edges = [(s, d) for s, d in zip(src_list, dst_list) if s >= 0 and d >= 0]
            if valid_edges:
                src_t = torch.tensor([e[0] for e in valid_edges], dtype=torch.long)
                dst_t = torch.tensor([e[1] for e in valid_edges], dtype=torch.long)
                edge_index = torch.stack([src_t, dst_t], dim=0)
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        # Graph-level label = 1 if any illicit node present
        graph_label = int((y == 1).any().item())

        return Data(x=x, edge_index=edge_index, y=torch.tensor([graph_label]))

    def load(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        if not self._check_files():
            warnings.warn(
                f"Elliptic dataset not found at {self.data_dir}. "
                "Falling back to synthetic data.  "
                "Download from https://www.kaggle.com/ellipticco/elliptic-data-set"
            )
            return SyntheticLoader(
                num_features=self.NUM_FEATURES, batch_size=self.batch_size
            ).load()

        print("[Elliptic] Loading raw CSV files …")
        merged, edge_df = self._load_raw()

        graphs: List[Data] = []
        for step in range(1, self.NUM_TIME_STEPS + 1):
            step_df = merged[merged["time_step"] == step].copy()
            if step_df.empty:
                continue
            g = self._build_time_step_graph(step_df, edge_df)
            if g is not None:
                graphs.append(g)

        train, val, test = _temporal_split(graphs)

        illicit_total = sum(int(g.y.item()) for g in graphs)
        print(
            f"[Elliptic]  time_steps={len(graphs)}  "
            f"illicit_graphs={illicit_total}  "
            f"train={len(train)}  val={len(val)}  test={len(test)}"
        )
        return _make_loaders(train, val, test, self.batch_size)

    @staticmethod
    def num_features() -> int:
        return EllipticLoader.NUM_FEATURES


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic fallback loader
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticLoader:
    """
    Generates structurally similar synthetic graphs for smoke-testing
    the full pipeline without downloading either real dataset.

    Graph properties match ETGraph statistics:
        - 50–200 nodes per graph
        - ~10 % phishing rate (class imbalance)
        - Random temporal edges
    """

    def __init__(
        self,
        num_graphs: int = 1000,
        num_features: int = 16,
        batch_size: int = 32,
        phishing_rate: float = 0.10,
        seed: int = 42,
    ) -> None:
        self.num_graphs    = num_graphs
        self.num_features  = num_features
        self.batch_size    = batch_size
        self.phishing_rate = phishing_rate
        self.seed          = seed

    def _make_graph(self, rng: np.random.Generator) -> Data:
        n_nodes = int(rng.integers(50, 201))
        n_edges = int(rng.integers(n_nodes, n_nodes * 4))
        x       = torch.from_numpy(rng.standard_normal((n_nodes, self.num_features)).astype(np.float32))
        src     = torch.from_numpy(rng.integers(0, n_nodes, n_edges).astype(np.int64))
        dst     = torch.from_numpy(rng.integers(0, n_nodes, n_edges).astype(np.int64))
        y       = torch.tensor([1 if rng.random() < self.phishing_rate else 0])
        return Data(x=x, edge_index=torch.stack([src, dst]), y=y)

    def load(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        rng    = np.random.default_rng(self.seed)
        graphs = [self._make_graph(rng) for _ in range(self.num_graphs)]
        train, val, test = _temporal_split(graphs)
        print(
            f"[Synthetic]  graphs={self.num_graphs}  features={self.num_features}  "
            f"train={len(train)}  val={len(val)}  test={len(test)}"
        )
        return _make_loaders(train, val, test, self.batch_size)


# ─────────────────────────────────────────────────────────────────────────────
# Unified entry point
# ─────────────────────────────────────────────────────────────────────────────

DATASET_REGISTRY: Dict[str, type] = {
    "etgraph":  ETGraphLoader,
    "elliptic": EllipticLoader,
    "synthetic": SyntheticLoader,
}


def load_dataset(
    name: str,
    data_dir: str = "./data",
    batch_size: int = 32,
    task_id: int = 1,
    **kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Unified dataset loader.

    Args:
        name       : "etgraph", "elliptic", or "synthetic".
        data_dir   : Root directory where datasets are stored / will be stored.
        batch_size : Mini-batch size.
        task_id    : ETGraph task index (1–6); ignored for other datasets.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    name = name.lower()
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(DATASET_REGISTRY)}")

    if name == "etgraph":
        loader_obj = ETGraphLoader(data_dir, task_id=task_id, batch_size=batch_size)
    elif name == "elliptic":
        loader_obj = EllipticLoader(data_dir, batch_size=batch_size)
    else:
        loader_obj = SyntheticLoader(batch_size=batch_size, **kwargs)

    return loader_obj.load()


def get_num_features(dataset_name: str) -> int:
    """Return the expected node feature dimensionality for each dataset."""
    mapping = {
        "etgraph":   ETGraphLoader.num_features(),
        "elliptic":  EllipticLoader.num_features(),
        "synthetic": 16,
    }
    return mapping.get(dataset_name.lower(), 16)
