# SGDA Peak-Mode Ablation

This ablation isolates the effect of **peak-location selection** in SGDA by
comparing physics-guided (MCSA) peak centers against uniformly random peak
centers while keeping everything else fixed.

## What is held constant
- Number of peaks per fault type (random uses the same count as MCSA).
- Peak shape parameters (amplitude, sigma, bandwidth, sign).
- Synthetic budget (K/R/p\_inject) and training schedule.
- Random seeds for all other randomness.

## Configuration knobs
Add these to `processing_parameters` (defaults shown):
```yaml
peak_mode: "mcsa"          # "mcsa" or "random"
peak_location_seed: null   # optional; if null uses seed + offset
```

Notes:
- Random peaks are sampled **uniformly** from FFT bins with a margin of
  `peak_segment` (in bins) to avoid truncated peak windows.
- Peak locations are **fixed per run** (no per-epoch resampling), matching the
  baseline behavior.
- Attention masks are disabled for this ablation.

## Run the ablation (5 seeds × 2 modes)
```cli
python experiments/run_peak_ablation.py \
  --cfg training_configs/train_engine-2.yml \
  --exp-name sgda_peak_ablation
```

## Parallel SLURM workflow (preferred)

1) Generate configs (writes `config_index.csv`):
```cli
python training_configs/sgda_peak_ablation/make_configs.py \
  --base-cfg training_configs/train_engine-2.yml \
  --out-dir training_configs/sgda_peak_ablation \
  --seeds 1 2 3 4 5 \
  --modes mcsa random \
  --disable-early-stopping
```

2) Update the array in `training_configs/sgda_peak_ablation/submit_peak_ablation.sbatch`
   to match the generated filenames (or keep the defaults if names match).

3) Submit the job array:
```cli
sbatch training_configs/sgda_peak_ablation/submit_peak_ablation.sbatch
```

Optional:
- Override seeds: `--seeds 11 12 13 14 15`
- Limit modes: `--modes mcsa` or `--modes random`
- Dry-run: `--dry-run`

All runs are written under:
```
res/runs/sgda_peak_ablation/
```
with a per-run config copy and a `ablation_manifest.json`.

## Aggregate results across seeds
Once runs are available (locally or on the cluster), aggregate metrics and
compare MCSA vs random with mean ± 3 std bands:

```cli
python experiments/peak_ablation_results.py \
  --runs-dir res/runs/sgda_peak_ablation \
  --plot-loss
```

Outputs are written to:
```
res/runs/sgda_peak_ablation/peak_ablation_results/
```
including per-run metrics, aggregated summaries, and comparison plots.
