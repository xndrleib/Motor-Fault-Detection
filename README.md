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
   git clone https://github.com/xndrleib/Motor-Fault-Detection.git
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

### **Data Description**

The dataset consists of oscilloscope data files that capture time-series measurements of motor operations under various conditions.

**1. `20_01_1460_об_мин_с_дисбалансом_ротора.txt`**
- **Description**: Data from a motor running at 1460 RPM with rotor imbalance.
- **Experiment Time**: 20th January 2023, 12:49:36.
- **Module**: E-2010B (8T773102).
- **Number of Frames**: 100,000.
- **Input Rate**: 10 kHz.
- **Channel**: 2.
- **Time Markers Scale**: Seconds.

**2. `21 02 дисбаланс макс обороты 100%.txt`**
- **Description**: Data from a motor operating at maximum RPM with imbalance.
- **Experiment Time**: 21st February 2023, 10:04:13.
- **Module**: E-2010B (8T773112).
- **Number of Frames**: 50,000.
- **Input Rate**: 10 kHz.
- **Channel**: 1.
- **Time Markers Scale**: Seconds.

**3. `0702  межвитковые фазы A ток фазы А.txt`**
- **Description**: Data related to phase A currents with inter-turn faults.
- **Experiment Time**: 7th February 2023, 11:11:56.
- **Module**: E-2010B (8T773112).
- **Number of Frames**: 50,000.
- **Input Rate**: 10 kHz.
- **Channel**: 1.
- **Time Markers Scale**: Milliseconds.

**4. `0702  перекос фазы A ток фазы А.txt`**
- **Description**: Data from phase A currents with phase imbalance.
- **Experiment Time**: 7th February 2023, 10:49:31.
- **Module**: E-2010B (8T773112).
- **Number of Frames**: 50,000.
- **Input Rate**: 10 kHz.
- **Channel**: 1.
- **Time Markers Scale**: Milliseconds.

**5. `0702 без перекосов фаза A.txt`**
- **Description**: Data from phase A currents without any imbalances.
- **Experiment Time**: 7th February 2023, 10:21:48.
- **Module**: E-2010B (8T773112).
- **Number of Frames**: 100,000.
- **Input Rate**: 10 kHz.
- **Channel**: 1.
- **Time Markers Scale**: Milliseconds.

**6. `ВКЛ-ВЫКЛ нагрузка 100% 11 01.txt`**
- **Description**: Data of motor operation under 100% load with switching on/off.
- **Experiment Time**: 11th January 2023, 12:05:10.
- **Module**: E-2010B (8T773102).
- **Number of Frames**: 600,000.
- **Input Rate**: 10 kHz.
- **Channel**: 2.
- **Time Markers Scale**: Seconds.

**7. `нагрузка 100% 11 01.txt`**
- **Description**: Data of motor operating at 100% load.
- **Experiment Time**: 11th January 2023, 12:02:45.
- **Module**: E-2010B (8T773102).
- **Number of Frames**: 600,000.
- **Input Rate**: 10 kHz.
- **Channel**: 2.
- **Time Markers Scale**: Seconds.

**8. `холостой ход 11 01.txt`**
- **Description**: Data from a motor running at no load (idle).
- **Experiment Time**: 11th January 2023, 11:59:08.
- **Module**: E-2010B (8T773102).
- **Number of Frames**: 600,000.
- **Input Rate**: 10 kHz.
- **Channel**: 2.
- **Time Markers Scale**: Seconds.
