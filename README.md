# DeepfakeDetection

Configuration-driven toolkit for deepfake detection research and benchmarking.

The repository contains:
- multiple detector implementations (CNNSpot, UnivFD, NPR, FreqNet, Latte, DIRE, FatFormer, AIDE, TruFor)
- reusable dataset loaders for images, videos, CSV, FaceForensics++, Hugging Face, and combined datasets
- scripts for training, evaluation, robustness testing, and FiftyOne-based dataset inspection

## Repository overview

- `deepfake_detection/`: core package
  - `data/`: dataset abstractions, loaders, sampling/filtering/mapping utilities
  - `models/`: detector wrappers, training utilities, custom networks, weights folder
  - `evaluation/`: evaluator and metrics tooling
  - `analysis/` and `visualization/`: signal analysis and embedding visualization helpers
  - `utils/`: config loading, IO, serialization, hashing helpers
- `configs/`: example YAML configurations
  - `configs/datasets/`: dataset config examples
  - `configs/training/`: per-model training config examples
- `train.py`: model training entrypoint (Hugging Face `Trainer`)
- `evaluate.py`: evaluation and metrics export entrypoint
- `display.py`: launch a FiftyOne app for dataset inspection
- `robustness.py`: JPEG-compression robustness benchmark
- `tests/`: unit and integration tests
- `scripts/train_sbatch.sh`: SLURM helper script

## Requirements

- Python `3.10.x` (project is pinned to `==3.10.*`)
- CUDA-enabled PyTorch is recommended for training speed

## Installation

From the repository root:

```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux/macOS
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install .
```

### Optional: development tools

```bash
pip install -e ".[test]"
```

Or install direct test tooling:

```bash
pip install pytest ruff
```

## Configuration style

Most workflows are YAML-driven. Classes are loaded dynamically from fully-qualified import paths.

### Dataset config (minimal)

```yaml
class: deepfake_detection.data.datasets.FileImageDataset
params:
  dataset_name: TrainDataset
  path: /path/to/train/
```

### Dataset config (advanced)

The loader supports:
- `map`: transform labels/instances via configured functions
- `filter`: keep samples by logical predicates (`and`, `or`, `not`)
- `sample`: downsample/rebalance datasets

See `configs/datasets/example_dataset_full.yaml` for a full end-to-end example.

### Training config

Training configs in `configs/training/*.yaml` include:
- `model`: class + params
- `training`: label setup, optimizer, augmentations, scheduler, output path, optional MLflow settings

Examples:
- `configs/training/cnnspot.yaml`
- `configs/training/univfd.yaml`
- `configs/training/aide.yaml`

## Usage

## 1) Train a model

```bash
python train.py \
  --config configs/training/cnnspot.yaml \
  --train-dataset configs/datasets/example_dataset.yaml \
  --val-dataset configs/datasets/example_dataset.yaml
```

Notes:
- `--val-dataset` is optional
- mixed precision is controlled by `training.fp16` in the training config
- checkpoints are saved under `training.output_dir`
- for AIDE reproduction, set `model.params.resnet_ckpt` and `model.params.convnext_ckpt` in `configs/training/aide.yaml`

## 2) Evaluate a model

```bash
python evaluate.py \
  --dataset-config configs/datasets/example_dataset.yaml \
  --model-config path/to/model_config.yaml \
  --label authenticity \
  --output-dir results
```

Optional flags:
- `--predictions-file`: reuse precomputed predictions JSON
- `--subset-labels`: compute per-subset metrics
- `--neg-label`: required for some binary per-subset metrics
- `--device`: override inference device (`cuda`/`cpu`)

Outputs:
- predictions JSON (if generated)
- metrics CSV in `output-dir`

## 3) Display dataset in FiftyOne

```bash
python display.py \
  --dataset configs/datasets/example_dataset.yaml \
  --cache-dir .cache
```

Optional:
- `--model path/to/model_config.yaml` to compute embeddings

## 4) Run JPEG robustness evaluation

```bash
python robustness.py \
  --dataset configs/datasets/example_dataset.yaml \
  --model path/to/model_config.yaml \
  --output-dir results
```

This evaluates performance under varying JPEG quality factors and exports a CSV.

## Running tests and linting

```bash
ruff check .
pytest -s -vv --import-mode=append tests/
```

If using PDM scripts:

```bash
pdm run lint
pdm run test
```

## Available detectors

The package currently includes:
- `CNNSpot`
- `DIRE`
- `FreqNet`
- `Latte`
- `NPR`
- `UnivFD`
- `FatFormer`
- `AIDE`
- `TruFor`

Implementations are under `deepfake_detection/models/detection/`.

## Weights

Pretrained weights are expected in `deepfake_detection/models/weights/`.
There are model-specific notes in `deepfake_detection/models/weights/README.md`.

## MLflow tracking (optional)

`train.py` supports MLflow logging via the `training.mlflow` section in config:

```yaml
mlflow:
  tracking_uri: mlflow.db
  experiment_name: deepfake_training
  run_name: cnnspot_run1
```

If `tracking_uri` is a local `.db` path, it is normalized automatically to a SQLite URI.
