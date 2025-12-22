# Motor Fault Detection

This repository contains a complete workflow for detecting electric motor faults from time-series data. Two datasets with different motor configurations are provided along with training scripts, utility modules and Jupyter notebooks for exploration.

## Repository Layout

```
├── dataset/              # Raw measurements and metadata
│   ├── engine_1/         # First motor dataset (see `dataset/engine_1/README.md`)
│   └── engine_2/         # Second motor dataset (see `dataset/engine_2/README.md`)
├── docs/                 # Additional documentation and notes
├── experiments/          # Training and analysis scripts
├── notebooks/            # Step‑by‑step exploratory notebooks
├── src/                  # Library code (dataset prep, models, training utils)
├── training_configs/     # YAML files describing training parameters
└── res/                  # Output directory for experiment results
```

## Installation

A Conda environment definition is provided. The included `setup.sh` script creates and activates the environment automatically:

```bash
conda env create -f environment.yaml
conda activate py311_mfd
```

After activating the environment, install the project in editable mode:

```bash
pip install -e .
```

## Datasets

Two datasets of real motor measurements are located under the `dataset/` directory. Each dataset contains multiple experiments with accompanying metadata files:

- **Engine 1** – Collected from a motor model RA180L4/2У3. Measurements include various load conditions and induced faults. Detailed descriptions are available in `dataset/engine_1/README.md`.
- **Engine 2** – Load variation experiments on four identical motors with different defects. See `dataset/engine_2/README.md` for folder structure and acquisition notes.

Each dataset folder also contains an `engine.yml` file describing the motor configuration (power, RPM, sampling rate, etc.).

## Training

Model training is performed using the script `experiments/train.py`. Training behavior is controlled through YAML configuration files in `training_configs/`. A configuration specifies the data source, the type of task (binary or multiclass classification) and all preprocessing and model parameters.

Example command for Engine 1 binary classification:

```bash
cd experiments
python train.py --cfg train_engine-2.yml
```

During training an experiment directory is created inside `res/` containing checkpoints, logs and predictions. The script also supports logging to [Comet ML](https://www.comet.com/) if API credentials are provided in `cfg.yaml`.

## Inference

The `src/inference.py` module provides a helper function to run a trained model on a `DataLoader`. Predictions on the test set are automatically generated at the end of training and saved alongside other results in the experiment directory.
