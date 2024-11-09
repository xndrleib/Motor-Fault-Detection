#!/bin/bash

# Check if environment.yml exists
if [[ ! -f "environment.yml" ]]; then
    echo "Error: environment.yml file not found!"
    exit 1
fi

# Initialize Conda for the current shell session
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not available in this shell. Please ensure Conda is installed."
    exit 1
fi

eval "$(conda shell.bash hook)"

# Create the conda environment from the YAML file
echo "Creating conda environment from environment.yml..."
conda env create -f environment.yml

# Extract the environment name from the YAML file
ENV_NAME=$(grep 'name:' environment.yml | awk '{print $2}')

# Activate the environment
echo "Activating environment: $ENV_NAME"
conda activate "$ENV_NAME"

# Check if we're on Linux for GPU-compatible PyTorch installation
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected Linux OS. Installing GPU-compatible PyTorch..."
    conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y
else
    echo "Non-Linux OS detected. Installing CPU-only PyTorch..."
    conda install pytorch::pytorch -c pytorch -y
fi

echo "Setup complete. The $ENV_NAME environment is ready."
