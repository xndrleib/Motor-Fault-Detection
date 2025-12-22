# AWGN Robustness Experiment

## Goal
Evaluate model robustness by injecting **additive white Gaussian noise (AWGN)** into
time-domain signals *before* FFT/spectral transforms. The experiment sweeps SNR levels
and measures metric degradation for both binary and multiclass tasks.

## Design
- **Noise injection:** AWGN is added per time-domain segment to reach a target SNR (dB).
- **SNR sweep:** [30, 20, 10, 5, 0] dB by default.
- **Preprocessing:** The script replays the training preprocessing (segmentation, FFT,
  filtering by class/load/phase) using the same config from each run.
- **Evaluation:** Best checkpoints (`best_{task}.pth`) are evaluated on noisy test sets.
- **Aggregation:** Results are aggregated across all seeds/runs for each task,
  producing mean and standard deviation per metric.

## Metrics
Computed per task and SNR:
- Accuracy
- Macro Precision / Recall / F1
- ROC AUC
  - Binary: standard ROC AUC on anomalous probability
  - Multiclass: macro OVR ROC AUC

## Outputs
Saved under `<runs_dir>/awgn_robustness` (or `--out_dir`):
- `results.csv` with columns: `snr_db`, `task_type`, `metrics`, `n_samples`
- `results.json` with full config, per-run metrics, and aggregated stats
- Degradation plots:
  - `degradation_macro_f1.png`
  - `degradation_accuracy.png`
  - `degradation_roc_auc.png`

## Usage
From the repo root:
```bash
cd experiments
python awgn_robustness.py \
  --runs_dir ../res/runs/exp9 \
  --snr_list 30 20 10 5 0 \
  --seed 42 \
  --data_root ../dataset
```

## Notes
- The script assumes all runs per task share the same preprocessing/model settings.
- Test indices and normalizers are loaded from each run to keep the evaluation consistent.
