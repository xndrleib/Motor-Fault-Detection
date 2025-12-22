#!/bin/bash
#SBATCH --job-name=sgda_peak_ablation
#SBATCH --array=1-10%10
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --output=task-logs/peak_ablation_%A_%a_out.txt
#SBATCH --error=task-logs/peak_ablation_%A_%a_err.txt
#SBATCH --partition=normal

module purge
module load Python

conda deactivate
conda activate py311_mfd

echo "Running on node $(hostname)"

# Repo root
REPO_DIR="/home/akhizhik/Motor-Fault-Detection"
cd "${REPO_DIR}/experiments"

# Declare an array of all config filenames
CONFIGS=(
  '../training_configs/sgda_peak_ablation/train_peakablation_mcsa_seed1_1.yml'
  '../training_configs/sgda_peak_ablation/train_peakablation_mcsa_seed2_2.yml'
  '../training_configs/sgda_peak_ablation/train_peakablation_mcsa_seed3_3.yml'
  '../training_configs/sgda_peak_ablation/train_peakablation_mcsa_seed4_4.yml'
  '../training_configs/sgda_peak_ablation/train_peakablation_mcsa_seed5_5.yml'
  '../training_configs/sgda_peak_ablation/train_peakablation_random_seed1_6.yml'
  '../training_configs/sgda_peak_ablation/train_peakablation_random_seed2_7.yml'
  '../training_configs/sgda_peak_ablation/train_peakablation_random_seed3_8.yml'
  '../training_configs/sgda_peak_ablation/train_peakablation_random_seed4_9.yml'
  '../training_configs/sgda_peak_ablation/train_peakablation_random_seed5_10.yml'
)

idx=$SLURM_ARRAY_TASK_ID
echo "This is array job number $idx"
CONFIG_FILE="${CONFIGS[$((idx-1))]}"
echo "Config file: ${CONFIG_FILE}"

srun python -u train.py --cfg "${CONFIG_FILE}" --exp-name sgda_peak_ablation
