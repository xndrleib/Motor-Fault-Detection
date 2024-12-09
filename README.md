# Motor-Fault-Detection

## Overview

The **Motor Fault Detection** project is designed to detect anomalies in motor operations using advanced signal processing and machine learning techniques. This repository implements a pipeline for preprocessing time-series data, training machine learning models, and detecting anomalies based on processed signals. The system includes a ResNet-based deep learning model and a Variational Autoencoder (VAE) for generating synthetic peak segments.

## Directory Structure

```
├── .gitignore
├── README.md                       # Project documentation
├── data/                           # Data directory
│   ├── raw/                        # Raw data files
│   └── processed/                  # Processed data files
├── environment.yml                 # Conda environment dependencies
├── notebooks/                      # Jupyter notebooks
│   ├── 01_Data-Preprocessing.ipynb # Data preprocessing steps
│   ├── 02_VAE-Training-and-Synthetic-Peak-Generation.ipynb
│   └── 03_ResNet-Training.ipynb    # ResNet training steps
├── setup.py                        # Setup script for Python package
├── setup.sh                        # Environment setup shell script
└── src/                            # Source code
    ├── __init__.py                 # Package initialization
    ├── data_preprocessing.py       # Data preprocessing utilities
    ├── models.py                   # Model definitions (ResNet, VAE)
    ├── train.py                    # Training utilities
    └── utils.py                    # Helper utilities
```

## Features

1. **Data Preprocessing**:
   - Converts raw oscilloscope data to FFT-based frequency data.
   - Extracts and normalizes frequency peaks for model input.

2. **Synthetic Data Generation**:
   - Uses a Variational Autoencoder (VAE) to generate synthetic peak data for anomaly augmentation.

3. **Model Training**:
   - Implements a ResNet architecture for time-series classification.
   - Trains on both real and synthetic data for binary and multi-class classification tasks.

4. **Anomaly Detection**:
   - Predicts motor faults by classifying time-series data into normal or anomalous categories.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/Motor-Fault-Detection.git
   cd Motor-Fault-Detection
   ```

2. Install dependencies using Conda:

   ```bash
   bash setup.sh
   conda activate py311_mfd
   ```

3. Install the project package:

   ```bash
   pip install .
   ```

## Usage

### 1. Preprocess Data

Run the `01_Data-Preprocessing.ipynb` notebook to:
- Read raw oscilloscope data.
- Perform FFT transformations.
- Normalize frequency peaks.

### 2. Generate Synthetic Data

Use `02_VAE-Training-and-Synthetic-Peak-Generation.ipynb` to:
- Train a Variational Autoencoder (VAE) on extracted peaks.
- Generate synthetic data for anomaly augmentation.

### 3. Train ResNet Model

Train the ResNet model using `03_ResNet-Training.ipynb`:
- Load preprocessed and synthetic data.
- Train for binary or multi-class classification.

## Key Files

- **`data_preprocessing.py`**: Utilities for transforming and preprocessing time-series data.
- **`models.py`**: Definitions of ResNet and VAE architectures.
- **`train.py`**: Functions for training models and generating synthetic peaks.
- **`utils.py`**: Helper functions for file management and data visualization.
