#!/bin/bash

# Check if environment.yaml exists
if [[ ! -f "environment.yaml" ]]; then
    echo "Error: environment.yaml file not found!"
    exit 1
fi

# Initialize Conda for the current shell session
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not available in this shell. Please ensure Conda is installed."
    exit 1
fi

eval "$(conda shell.bash hook)"

# Create the conda environment from the YAML file
echo "Creating conda environment from environment.yaml..."
conda env create -f environment.yaml

# Extract the environment name from the YAML file
ENV_NAME=$(grep 'name:' environment.yaml | awk '{print $2}')

# Activate the environment
echo "Activating environment: $ENV_NAME"
conda activate "$ENV_NAME"

echo "Setup complete. The $ENV_NAME environment is ready."

pip install -e .