# TOPS-ARTEMIS
# ARTEMIS+P — ACM TOPS Artifact

Reference implementation for *“ARTEMIS: A Certifiably Robust and Privacy-Preserving
Framework for Ethereum Phishing Detection with Dual Adversarial and Local
Differential Privacy Guarantees.”*

This README is written so a fresh machine can install, smoke-test, and (given the
datasets and the base modules) reproduce the tables in the paper.

---

## 1. What is in this bundle — and what is **not**

**Included (the TOPS privacy layer + runners):**

| File | Role |
|------|------|
| `run_all_experiments_tops.py` | Master runner — Tables 4–11 (main, ablation, adversarial, efficiency, continual, dependability, privacy). |
| `run_privacy_evaluation.py` | Privacy-only runner — Tables P1–P5 and the privacy figures. |
| `artemis_model_tops.py` | `ARTEMISPlus` / `build_artemis_tops` — the privacy-augmented model (P1, P2). |
| `artemis_privacy.py` | P1 DP memory, P2 dual smoothing, P3 Rényi accountant, P4 MI defence, P5 PUR analyser. |
| `data_loader_tops.py` | Loaders for ETGraph / MulDiGraph / MDBChain + synthetic fallback. |
| `tops_config.yaml` | All hyperparameters (Table 3). |

**NOT included — required before anything runs.** The code imports three base
(“TDSC”) modules that are not in this bundle:

| Missing module | Symbols imported by this bundle |
|----------------|---------------------------------|
| `artemis_model.py` | `build_artemis`, `ARTEMIS`, `SpectralNormLinear`, `TemporalGraphEncoder` |
| `artemis_innovations.py` | `LATENCY_TRACKER`, `LatencyTracker`, `ElasticWeightConsolidation`, `CertifiedAdversarialTrainer` |
| `baseline_implementations.py` | baseline model builders (2DynEthNet, GrabPhisher, JODIE, TGN, TGAT, GAT, GraphSAGE) |

These are imported at module load time:

* `run_all_experiments_tops.py` imports `artemis_model` and `artemis_innovations` directly.
* `run_privacy_evaluation.py` imports `build_artemis_tops`, which lives in
  `artemis_model_tops.py`, which itself does
  `from artemis_model import SpectralNormLinear, TemporalGraphEncoder` and
  `from artemis_innovations import ...`.

**Consequence:** without the three modules above, *both* runners fail at import with
`ModuleNotFoundError`. Place them on the `PYTHONPATH` (same directory is simplest)
before testing. Everything below assumes they are present.

---

## 2. Requirements

* **Python** 3.9+ (the paper used 3.9.18)
* **PyTorch** 2.1.x (CUDA 11.8 build for GPU; CPU build works for smoke tests)
* **PyTorch Geometric** (`torch_geometric`) matching your torch/CUDA
* **NumPy**, **SciPy**, **scikit-learn**, **PyYAML**

The uploaded files import only `torch`, `torch_geometric`, `numpy`, `scipy`,
`sklearn`, and `yaml`. The **base modules additionally require** (per the paper’s
§5.3): **`torchdiffeq`** (L1 Neural-ODE solver) and **`statsmodels`**
(Clopper–Pearson intervals for L6 certification). Install them too.

### Suggested install

```bash
conda create -n artemis python=3.9 -y
conda activate artemis

# Torch + PyG: pick the wheels matching your CUDA (example: CUDA 11.8)
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric

# Scientific stack + paper-specific deps
pip install numpy scipy scikit-learn pyyaml torchdiffeq statsmodels
```

CPU-only quick test: install the CPU torch wheel
(`pip install torch==2.1.0`) instead; PyG has CPU wheels too.

A minimal `requirements.txt` you can drop next to the code:

```
torch==2.1.0
torch_geometric
numpy
scipy
scikit-learn
pyyaml
torchdiffeq
statsmodels
```

---

## 3. Fastest path: synthetic smoke test (no datasets needed)

If no dataset files are present, the loader generates a **synthetic fallback** that
matches each dataset’s feature dimensionality, phishing rate, and label counts. This
is for **end-to-end plumbing checks only**.

```bash
# Smoke-test everything on synthetic data (1 seed, few epochs, small sample counts)
python run_all_experiments_tops.py --mode all --quick --data_dir ./data
```

The runner prints, for every dataset:

```
DATA SOURCE: SYNTHETIC (no real data found — synthetic fallback active)
⚠️  Results from SYNTHETIC data must NOT be reported in the paper.
```

and tags every JSON record with `"data_source": "synthetic"`. **Do not put any
synthetic number in the manuscript.** A run is “successful” if each mode finishes
and writes `results_tops/table_<mode>.json` plus `results_tops/all_tables.json`
without an `"error"` field.

> Baselines are **skipped** under synthetic/quick runs unless you pass
> `--allow_stubs` (see §6), because the real baseline code lives in
> `baseline_implementations.py`. Stub baselines are valid for plumbing only.

---

## 4. Real-data runs (for paper numbers)

Place each dataset under `--data_dir` (default `./data`) using the layout the loader
expects. If a `raw/` or `processed/` folder exists **and is non-empty**, the runner
reports `DATA SOURCE: REAL`.

### Directory layout

```
data/
  etgraph/
    raw/
      graphs_task1.pt ... graphs_task6.pt   # list[Data] per temporal task
    processed/                               # optional pre-split cache
      task1_train.pt  task1_val.pt  task1_test.pt   ... (per task)
  muldigraph/
    raw/
      node_features.pt   # FloatTensor [N,18]
      edge_index.pt      # LongTensor  [2,E]
      labels.pt          # LongTensor  [N]  (0=benign,1=phishing,-1=unknown)
      timestamps.pt      # FloatTensor [E]  (normalised edge times)
    processed/
      graphs.pt          # list[Data] temporal subgraphs
      splits.json        # {"train":[...],"val":[...],"test":[...]}
  mdbchain/
    raw/
      node_features.pt   # [66402,18]
      edge_index.pt      # [2,~1.2M]
      labels.pt          # [66402]  (0/1/-1)
      timestamps.pt
    processed/
      graphs.pt
      splits.json
```

### Dataset sources (from `data_loader_tops.py`)

* **ETGraph** — https://xblock.pro/#/dataset/68
* **MulDiGraph** — IET Blockchain paper `doi:10.1049/blc2.12031` (and the
  MulDiGraph-Ethereum repository)
* **MDBChain** — authors of Sensors 2024 `doi:10.3390/s24124022` (or the DA-HGNN repo)

You must convert each source into the `.pt` tensors above (18-D node features in the
schema documented at the top of `data_loader_tops.py`). The `processed/splits.json`
cache, if present, is used directly and bypasses re-splitting.

### Full reproduction commands

```bash
# All tables, all three datasets, 5 seeds, 100 epochs (paper config)
python run_all_experiments_tops.py --mode all --dataset all --data_dir ./data \
    --allow_stubs            # only if you lack real baseline code; results then NOT reportable

# Individual tables
python run_all_experiments_tops.py --mode main        --dataset etgraph
python run_all_experiments_tops.py --mode ablation     --dataset etgraph
python run_all_experiments_tops.py --mode adversarial  --dataset etgraph --epsilon 0.05 0.10 0.20
python run_all_experiments_tops.py --mode efficiency   --dataset etgraph
python run_all_experiments_tops.py --mode continual    --dataset etgraph
python run_all_experiments_tops.py --mode dependability --dataset etgraph
python run_all_experiments_tops.py --mode privacy      --dataset etgraph

# Privacy tables P1–P5 (separate runner)
python run_privacy_evaluation.py --config tops_config.yaml --train --epochs 100
```

---

## 5. Command reference

### `run_all_experiments_tops.py`

| Flag | Default | Meaning |
|------|---------|---------|
| `--mode` | `all` | `main`, `ablation`, `adversarial`, `efficiency`, `continual`, `dependability`, `privacy`, `all` |
| `--dataset` | `all` | `etgraph`, `muldigraph`, `mdbchain`, `all` |
| `--data_dir` | `./data` | Root directory for dataset files |
| `--output` | `./results_tops` | JSON output directory |
| `--config` | `tops_config.yaml` | Hyperparameter file |
| `--quick` | off | Smoke test: 1 seed, ≤3 epochs, small sample counts |
| `--seed` | `42` | Global seed (full runs use seeds 42, 123, 456, 789, 1011) |
| `--epsilon` | `0.05 0.10 0.20` | PGD ℓ₂ budgets for adversarial mode |
| `--allow_stubs` | off | Permit stub baselines when real baseline code is absent (**stub results are not reportable**) |

### `run_privacy_evaluation.py`

| Flag | Default | Meaning |
|------|---------|---------|
| `--config` | `tops_config.yaml` | Hyperparameter file |
| `--device` | `cuda` if available else `cpu` | Compute device |
| `--seed` | `42` | Random seed |
| `--tables` | `p1 p2 p3 p4 p5` | Subset of privacy tables to generate |
| `--train` | off | Run private training first — **required** for P4 (empirical MI) and P5 (measured Pareto); without it P4 shows bounds only and P5 warns |
| `--epochs` | `5` | Training epochs when `--train` is set (use `100` for paper results) |

---

## 6. Outputs

Written under `output:` in the config (default `./results_tops`):

```
results_tops/
  table_main.json  table_ablation.json  table_adversarial.json
  table_efficiency.json  table_continual.json  table_dependability.json
  table_privacy.json
  all_tables.json
  privacy/         # P1–P5 reports from run_privacy_evaluation.py
  figures/         # figures, if generated
  checkpoints/     # model checkpoints
```

Every record carries `"data_source": "real"` or `"synthetic"` — check this field
before using any number.

---

## 7. Notes and caveats relevant to testing

These are things that will bite you during a test run, so they are stated plainly:

1. **Base modules first.** Nothing imports without `artemis_model.py`,
   `artemis_innovations.py`, and `baseline_implementations.py` on the path (§1).

2. **Synthetic ≠ reportable.** The fallback is for plumbing. Results from it must not
   appear in the paper; the runner says so on every dataset.

3. **Baselines need real code.** 2DynEthNet, GrabPhisher, and JODIE are skipped unless
   `--allow_stubs` is set; stub numbers are smoke-test only. For paper comparisons,
   supply the authors’ implementations in `baseline_implementations.py`.

4. **GPU memory.** The paper used 4× RTX 3090 (24 GB). Batch 32 fits on one card;
   batch 256 needs 4 GPUs (Table 11c). Reduce `training.batch_size` for smaller cards.

5. **Config vs. paper label count.** `tops_config.yaml` sets
   `etgraph_phishing_labels: 23847`, while `data_loader_tops.py` `DATASET_STATS` and
   Table 2 of the paper report **9,032** for ETGraph. The privacy accountant’s step
   count (`dp_sample_rate: 0.01`, commented `32/3200`) and `n_steps = 52,100` should
   be re-derived from whichever figure is correct before quoting privacy budgets from
   a real run. Reconcile this prior to reproducing Tables 10/P1–P5.

6. **`target_epsilon`.** The accountant stops training when ε exceeds
   `privacy_accountant.target_epsilon` (8.0 in the config); the paper reports the
   crossing at ε = 8.006, epoch 30 (σ_DP = 1.1).

7. **Determinism.** Seeds and cuDNN deterministic flags are set, but the adaptive ODE
   solver under multi-GPU all-reduce is not bit-for-bit reproducible across runs
   (paper §8.2). Expect small run-to-run variation.

---

## 8. Five-minute verification checklist

```bash
# 1. Deps import
python -c "import torch, torch_geometric, numpy, scipy, sklearn, yaml; print('deps OK')"

# 2. Base modules import (must succeed before any run)
python -c "import artemis_model, artemis_innovations, baseline_implementations; print('base OK')"

# 3. Project modules import
python -c "import artemis_model_tops, artemis_privacy, data_loader_tops; print('project OK')"

# 4. Synthetic end-to-end (fast)
python run_all_experiments_tops.py --mode privacy --dataset mdbchain --quick

# 5. Confirm output + provenance
python -c "import json; d=json.load(open('results_tops/table_privacy.json')); print('wrote results;', 'error' not in d)"
```

If all five pass, the pipeline is wired correctly; switch to real data (§4) for any
number you intend to publish.
