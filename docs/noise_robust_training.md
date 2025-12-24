# Noise-Robust Training (Time-Domain AWGN)

This experiment trains models with **additional time-domain AWGN** (incremental
noise) and then evaluates robustness using the existing AWGN grid test.

## Methodology (detailed)
The goal is to improve noise robustness without changing model architecture or
the SGDA augmentation logic.

### 1) Noise definition (incremental)
We add **incremental noise** to the recorded waveform:

```
x_train = x_recorded + n1
```

Noise power is defined **relative to the recorded signal**:

```
SNR_add (dB) = 10 * log10(Px / Pn1)
Px = mean(x^2)
```

This matches evaluation and avoids estimating a latent clean signal.

### 2) Per-example noise policy
Noise is sampled **per example**:
- With probability `p_clean`: keep the sample clean.
- Otherwise: add AWGN at a sampled SNR in dB.

Currently, **AWGN is the only supported noise type** in code (additional
types can be added later).

### 3) Curriculum schedule
To avoid training collapse, SNR ranges can be scheduled over epochs
(high → low SNR):

```
Epochs 0–20%:  SNR 30–40 dB
Epochs 20–60%: SNR 20–35 dB
Epochs 60–100%: SNR 10–30 dB
```

### 4) Training parity
All non-noise settings remain fixed:
- same model, data splits, and SGDA parameters
- attention masks disabled (for parity)
- early stopping optionally disabled

### 5) Evaluation
After training, run the AWGN robustness grid to compare degradation curves.

## Configuration knobs
Add this to `processing_parameters`:

```yaml
train_noise_policy:
  enabled: true
  p_clean: 0.3
  snr_db:
    distribution: "uniform"   # uniform | mixture | bin_weighted
    snr_min: 20
    snr_max: 40
    knee_range: [18, 22]       # used only for mixture
    knee_weight: 0.5
  curriculum:
    enabled: true
    schedule:
      - {start_pct: 0.0, end_pct: 0.2, snr_min: 30, snr_max: 40}
      - {start_pct: 0.2, end_pct: 0.6, snr_min: 20, snr_max: 35}
      - {start_pct: 0.6, end_pct: 1.0, snr_min: 10, snr_max: 30}
  epsilon: 1e-12
```

## Code-level walkthrough (key excerpts)

### A) Time-domain AWGN is injected before FFT
Noise is applied on raw windows, then the FFT is computed:

```python
# src/noise_policy.py
power = mean(signal ** 2)
noise_power = power / (10 ** (snr_db / 10))
noise = rng.normal(0, sqrt(noise_power), size=signal.shape)
signal = signal + noise
```

```python
# src/datasets.py (HybridAugFaultDataset)
time_seg, _ = noise_policy.apply(time_seg, rng=..., epoch=..., total_epochs=...)
seg, _ = time_to_freq_transform(time_seg, f_sampling=..., cutoff_freq=..., db=...)
```

### B) Policy sampling and curriculum scheduling
SNR ranges can change by epoch; sampling is per-example:

```python
# src/noise_policy.py
snr_min, snr_max = policy._snr_bounds(epoch, total_epochs)
snr_db = rng.uniform(snr_min, snr_max)  # or mixture / bin_weighted
```

### C) Training pipeline wiring
`experiments/train.py` passes the policy and time-domain windows to the dataset:

```python
# experiments/train.py
noise_policy = NoisePolicy.from_config(train_noise_cfg)
datasets = create_balanced_datasets(
    ..., time_segments=time_segments, train_dataset_kwargs={
        "noise_policy": noise_policy,
        "noise_total_epochs": num_epochs,
        "noise_fft_params": {...},
    }
)
```

## Run the experiment

### Single run
```cli
python experiments/train.py \
  --cfg training_configs/train_engine-2.yml \
  --exp-name sgda_noise_robust_train
```

### Generate configs (recommended for multi-seed runs)
```cli
python training_configs/noise_robust_training/make_configs.py \
  --base-cfg training_configs/train_engine-2.yml \
  --out-dir training_configs/noise_robust_training \
  --seeds 42 43 44 45 46 \
  --p-clean 0.3 \
  --tasks binary multiclass \
  --snr-min 20 --snr-max 40
```

### SLURM array (optional)
```cli
sbatch training_configs/noise_robust_training/submit_noise_robust_training.sbatch
```

## Evaluate robustness (AWGN grid)
```cli
cd experiments
python awgn_robustness.py \
  --runs_dir ../res/runs/sgda_noise_robust_train \
  --data_root ../dataset \
  --snr_list 30 20 10 5 0 \
  --seed 42
```

Outputs are written under:
```
res/runs/sgda_noise_robust_train/awgn_robustness/
```
including aggregated metrics and degradation plots.
