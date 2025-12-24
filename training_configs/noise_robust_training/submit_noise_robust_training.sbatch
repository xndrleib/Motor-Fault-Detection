#!/bin/bash
#SBATCH --job-name=sgda_noise_robust_train
#SBATCH --array=1-10%10
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --output=task-logs/noise_robust_%A_%a_out.txt
#SBATCH --error=task-logs/noise_robust_%A_%a_err.txt
#SBATCH --partition=normal

module purge
module load Python

conda deactivate
conda activate py311_mfd

echo "Running on node $(hostname)"

# Repo root (adjust if needed)
REPO_DIR="/home/akhizhik/Motor-Fault-Detection"
cd "${REPO_DIR}/experiments"

# Declare an array of all config filenames
CONFIGS=(
  '../training_configs/noise_robust_training/train_noiserobust_binary_seed42_1.yml'
  '../training_configs/noise_robust_training/train_noiserobust_binary_seed43_2.yml'
  '../training_configs/noise_robust_training/train_noiserobust_binary_seed44_3.yml'
  '../training_configs/noise_robust_training/train_noiserobust_binary_seed45_4.yml'
  '../training_configs/noise_robust_training/train_noiserobust_binary_seed46_5.yml'
  '../training_configs/noise_robust_training/train_noiserobust_multiclass_seed42_6.yml'
  '../training_configs/noise_robust_training/train_noiserobust_multiclass_seed43_7.yml'
  '../training_configs/noise_robust_training/train_noiserobust_multiclass_seed44_8.yml'
  '../training_configs/noise_robust_training/train_noiserobust_multiclass_seed45_9.yml'
  '../training_configs/noise_robust_training/train_noiserobust_multiclass_seed46_10.yml'
)

idx=$SLURM_ARRAY_TASK_ID
echo "This is array job number $idx"
CONFIG_FILE="${CONFIGS[$((idx-1))]}"
echo "Config file: ${CONFIG_FILE}"

srun python -u train.py --cfg "${CONFIG_FILE}" --exp-name sgda_noise_robust_train
