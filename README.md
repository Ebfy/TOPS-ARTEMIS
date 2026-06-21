# ARTEMIS+P: Testing and Experiment Guide

This repository contains the privacy-augmented ARTEMIS implementation and experiment runners prepared for an ACM Transactions on Privacy and Security (TOPS) evaluation.

The code supports:

- Cross-dataset fraud-detection experiments on ETGraph, MulDiGraph, and MDBChain.
- Ablation, adversarial robustness, efficiency, continual-learning, and dependability evaluations.
- Privacy Tables P1–P5 covering DP anomaly memory, dual-purpose randomized smoothing, Rényi DP accounting, membership-inference risk, and privacy–utility–robustness trade-offs.
- Synthetic fallback datasets for software testing when the real datasets are unavailable.

> **Important:** Synthetic results are only for debugging and smoke testing. They must not be reported as experimental results in a paper.

## 1. Required project files

Place the following files in the same project directory:

```text
artemis-tops/
├── artemis_model.py                 # Required base ARTEMIS module
├── artemis_innovations.py           # Required base innovation components
├── artemis_model_tops.py            # ARTEMIS+P model
├── artemis_privacy.py               # Privacy mechanisms P1–P5
├── data_loader_tops.py              # Real/synthetic dataset loaders
├── run_all_experiments_tops.py      # Master experiment runner
├── run_privacy_evaluation.py        # Standalone privacy evaluator
├── tops_config.yaml                 # Hyperparameter configuration
├── requirements_tops.txt            # Python dependencies
└── data/                             # Optional real datasets
```

The supplied TOPS files are **not fully standalone**. The following two modules must be copied from the original ARTEMIS/TDSC codebase before the complete model can run:

```text
artemis_model.py
artemis_innovations.py
```

`baseline_implementations.py` is optional. When it is absent, built-in implementations are used for GAT, GraphSAGE, TGN, and TGAT. The 2DynEthNet, GrabPhisher, and JODIE baselines are unavailable unless their real implementations are supplied.

## 2. Environment setup

Python 3.10 or 3.11 is recommended.

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements_tops.txt
```

### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements_tops.txt
```

For GPU execution, install a PyTorch build compatible with the installed CUDA driver before installing PyTorch Geometric. Verify the environment with:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
```

## 3. Preflight tests

Run all commands from the directory containing the Python files.

### 3.1 Check that required files exist

Linux/macOS:

```bash
ls artemis_model.py artemis_innovations.py artemis_model_tops.py \
   artemis_privacy.py data_loader_tops.py run_all_experiments_tops.py \
   run_privacy_evaluation.py tops_config.yaml
```

Windows PowerShell:

```powershell
Get-Item artemis_model.py, artemis_innovations.py, artemis_model_tops.py, `
         artemis_privacy.py, data_loader_tops.py, run_all_experiments_tops.py, `
         run_privacy_evaluation.py, tops_config.yaml
```

### 3.2 Compile the uploaded Python files

This catches syntax and indentation errors without starting training:

```bash
python -m py_compile \
  artemis_model_tops.py \
  artemis_privacy.py \
  data_loader_tops.py \
  run_all_experiments_tops.py \
  run_privacy_evaluation.py
```

No output means compilation succeeded.

### 3.3 Check third-party imports

```bash
python -c "import numpy, scipy, sklearn, yaml, torch, torch_geometric; print('Third-party imports: OK')"
```

### 3.4 Check project imports

```bash
python -c "import artemis_model, artemis_innovations, artemis_privacy, data_loader_tops, artemis_model_tops; print('ARTEMIS imports: OK')"
```

A `ModuleNotFoundError` for `artemis_model` or `artemis_innovations` means the original base modules have not yet been copied into the project directory.

## 4. Fastest confirmed CPU test

The following command tests the privacy calculations without training a model or loading a dataset:

```bash
python run_privacy_evaluation.py \
  --config tops_config.yaml \
  --device cpu \
  --tables p2 p3
```

It should generate:

```text
results_tops/privacy/table_p2_dual_smoothing.json
results_tops/privacy/table_p3_budget.json
```

Table P2 evaluates the randomized-smoothing privacy/robustness trade-off. Table P3 evaluates cumulative Rényi DP budget consumption.

## 5. Privacy evaluation commands

### Run one privacy table

```bash
python run_privacy_evaluation.py --config tops_config.yaml --device cpu --tables p2
```

Valid table names are:

```text
p1  p2  p3  p4  p5
```

### Run the non-training privacy calculations

```bash
python run_privacy_evaluation.py \
  --config tops_config.yaml \
  --device cpu \
  --tables p1 p2 p3
```

Table P1 can be noticeably slower on CPU because it repeatedly updates and projects covariance matrices and evaluates privacy-accounting terms.

### Run all privacy tables without training

```bash
python run_privacy_evaluation.py \
  --config tops_config.yaml \
  --device cpu \
  --tables p1 p2 p3 p4 p5
```

Without `--train`, P4 reports only theoretical membership-inference information, and P5 warns that measured Pareto points are unavailable.

### One-epoch privacy training smoke test

```bash
python run_privacy_evaluation.py \
  --config tops_config.yaml \
  --device cpu \
  --train \
  --epochs 1 \
  --tables p4
```

Use a GPU for the measured P5 sweep because P5 trains a separate model for every value in `pur_analysis.noise_grid`.

### Paper-scale privacy training

```bash
python run_privacy_evaluation.py \
  --config tops_config.yaml \
  --device cuda \
  --train \
  --epochs 100 \
  --tables p1 p2 p3 p4 p5
```

## 6. Master experiment runner

Display the command-line options:

```bash
python run_all_experiments_tops.py --help
```

Available modes are:

```text
main
ablation
adversarial
efficiency
continual
dependability
privacy
all
```

### Small main-experiment smoke test

```bash
python run_all_experiments_tops.py \
  --mode main \
  --dataset etgraph \
  --quick \
  --data_dir ./data \
  --output ./results_tops_smoke
```

In quick mode, the runner uses one seed and three epochs. When the ETGraph files are missing, the loader automatically uses a calibrated synthetic dataset.

### Include unavailable baseline stubs for pipeline testing

```bash
python run_all_experiments_tops.py \
  --mode main \
  --dataset etgraph \
  --quick \
  --allow_stubs \
  --data_dir ./data \
  --output ./results_tops_smoke
```

`--allow_stubs` permits placeholder versions of 2DynEthNet, GrabPhisher, and JODIE. Their outputs are **not valid baseline results** and must never be included in a publication.

Without `--allow_stubs`, these unavailable baselines are skipped with a warning.

### Run a selected experiment

```bash
python run_all_experiments_tops.py --mode ablation --quick
python run_all_experiments_tops.py --mode adversarial --quick --epsilon 0.05 0.10 0.20
python run_all_experiments_tops.py --mode efficiency --quick
python run_all_experiments_tops.py --mode continual --quick
python run_all_experiments_tops.py --mode dependability --quick
python run_all_experiments_tops.py --mode privacy --quick
```

### Full cross-dataset experiment

```bash
python run_all_experiments_tops.py \
  --mode main \
  --dataset all \
  --data_dir ./data \
  --config tops_config.yaml \
  --output ./results_tops
```

### Run every experiment

```bash
python run_all_experiments_tops.py \
  --mode all \
  --dataset all \
  --data_dir ./data \
  --config tops_config.yaml \
  --output ./results_tops
```

The complete run trains many models and should be executed on a CUDA-capable system with sufficient memory.

## 7. Dataset directory layouts

The loaders apply a chronological 70/10/20 train/validation/test split.

### ETGraph

```text
data/etgraph/
├── raw/
│   ├── graphs_task1.pt
│   ├── graphs_task2.pt
│   ├── ...
│   └── graphs_task6.pt
└── processed/
    ├── task1_train.pt
    ├── task1_val.pt
    └── task1_test.pt
```

Either the raw task file or all three processed split files may be supplied.

### MulDiGraph

```text
data/muldigraph/
├── raw/
│   ├── node_features.pt
│   ├── edge_index.pt
│   ├── labels.pt
│   └── timestamps.pt
└── processed/
    └── graphs.pt
```

### MDBChain

```text
data/mdbchain/
├── raw/
│   ├── node_features.pt
│   ├── edge_index.pt
│   ├── labels.pt
│   └── timestamps.pt
└── processed/
    └── graphs.pt
```

Expected tensors:

```text
node_features.pt : FloatTensor [N, 18]
edge_index.pt    : LongTensor  [2, E]
labels.pt        : LongTensor  [N], using 0=benign, 1=phishing, -1=unknown
timestamps.pt    : FloatTensor [E]
```

## 8. Output files

### Master runner

The master runner creates timestamped JSON files in the directory passed through `--output`, for example:

```text
results_tops/
├── table_main_YYYYMMDD_HHMMSS.json
├── table_ablation_YYYYMMDD_HHMMSS.json
├── table_adversarial_YYYYMMDD_HHMMSS.json
├── table_efficiency_YYYYMMDD_HHMMSS.json
├── table_continual_YYYYMMDD_HHMMSS.json
├── table_dependability_YYYYMMDD_HHMMSS.json
├── table_privacy_YYYYMMDD_HHMMSS.json
└── all_tables_YYYYMMDD_HHMMSS.json
```

### Standalone privacy runner

The privacy runner writes fixed-name JSON files to `output.privacy_dir` from `tops_config.yaml`:

```text
results_tops/privacy/
├── table_p1_dp_memory.json
├── table_p2_dual_smoothing.json
├── table_p3_budget.json
├── table_p4_mi_summary.json
├── table_p5_pur.json
├── training_privacy_log.json
└── full_privacy_report.json
```

The uploaded implementation currently writes JSON tables. Although the privacy runner's module description mentions Figures P1–P3, figure-generation code is not present in the supplied files.

## 9. Recommended testing sequence

Use this order to isolate failures efficiently:

1. Run `py_compile` on all Python files.
2. Check third-party imports.
3. Check the two required base ARTEMIS modules.
4. Run privacy Tables P2 and P3 on CPU.
5. Run a one-epoch privacy-training test.
6. Run the master runner in `--quick` mode on one dataset.
7. Add the real datasets and confirm the log prints `DATA SOURCE: REAL`.
8. Run the full experiments on a GPU.

## 10. Troubleshooting

### `ModuleNotFoundError: No module named 'artemis_model'`

Copy `artemis_model.py` from the original ARTEMIS codebase into the same directory as the runners.

### `ModuleNotFoundError: No module named 'artemis_innovations'`

Copy `artemis_innovations.py` from the original ARTEMIS codebase into the same directory.

### `ModuleNotFoundError: No module named 'torch_geometric'`

Install PyTorch Geometric in the active virtual environment:

```bash
pip install torch-geometric
```

If compiled-extension errors occur, install PyG wheels matching the exact PyTorch and CUDA versions.

### CUDA is unavailable

Check:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.__version__, torch.version.cuda)"
```

A CPU-only PyTorch build or an incompatible CUDA driver prevents GPU execution.

### The loader says `SYNTHETIC FALLBACK`

The expected real dataset files were not found under `./data/<dataset>/`. Verify the directory structure and the `--data_dir` argument.

### Out-of-memory error

Reduce values in `tops_config.yaml`, especially:

```yaml
model:
  hidden_channels: 64
  memory_size: 500

training:
  batch_size: 8
  mixed_precision: true
```

Also reduce randomized-smoothing samples during debugging.

### P1 runs slowly on CPU

P1 performs repeated private covariance updates, positive-semidefinite projections, and privacy-accounting calculations. Run P2/P3 first for a quick sanity check, and use a GPU or reduced experimental grid for P1 debugging.

### Results differ between runs

The runners set random seeds, but exact reproducibility can still depend on GPU hardware, CUDA, PyTorch, and PyG versions. Record the successful environment using:

```bash
pip freeze > environment-lock.txt
```
