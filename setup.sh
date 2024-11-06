#!/bin/bash

# Check if py311_mfd.yml exists
if [[ ! -f "py311_mfd.yml" ]]; then
    echo "Error: py311_mfd.yml file not found!"
    exit 1
fi

# Check if requirements.txt exists
if [[ ! -f "requirements.txt" ]]; then
    echo "Error: requirements.txt file not found!"
    exit 1
fi

# Initialize Conda for the current shell session
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not available in this shell. Please ensure Conda is installed."
    exit 1
fi

eval "$(conda shell.bash hook)"

# Create the conda environment from the YAML file
echo "Creating conda environment from py311_mfd.yml..."
conda env create -f py311_mfd.yml

# Extract the environment name from the YAML file
ENV_NAME=$(grep 'name:' py311_mfd.yml | awk '{print $2}')

# Activate the environment
echo "Activating environment: $ENV_NAME"
conda activate "$ENV_NAME"

# Check if we're on Linux for GPU-compatible PyTorch installation
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected Linux OS. Installing GPU-compatible PyTorch..."
    pip install torch --index-url https://download.pytorch.org/whl/cu118
else
    echo "Non-Linux OS detected. Installing CPU-only PyTorch..."
    pip install torch
fi

echo "Setup complete. The $ENV_NAME environment is ready."
