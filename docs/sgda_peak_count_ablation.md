# SGDA Peak-Count Ablation

This ablation isolates the effect of **how many** SGDA peaks are injected per
fault type by keeping everything else fixed and randomizing the peak count in
`random` mode.

## Methodology (detailed)
The experiment changes only the **number of injected peaks** in random mode,
while retaining the same model, preprocessing, and training schedule.

### 1) Base configuration and fixed settings
Each run starts from a common training config (e.g. `training_configs/train_engine-2.yml`):
- Same dataset, preprocessing, and SGDA injection settings.
- Same training schedule and optimizer settings.
- Same model architecture (ResNet) and **attention masks disabled** for all runs.

Only the peak-selection mode and seed differ between runs.

### 2) Peak-selection modes
- **MCSA (physics-guided)**: uses existing MCSA peak centers and their count.
- **Random (random count)**: uses random peak centers, **and** samples the number
  of peaks per fault uniformly from a configured range.

For example, with `random_peak_count_range: [1, 10]`, each fault gets a random
integer count between 1 and 10 (inclusive) for that run.

### 3) Randomness and reproducibility
Each run uses a fixed training seed (`seed`) and a separate peak-location seed
(`peak_location_seed`):
- If `peak_location_seed` is not specified, it is derived as
  `seed + DEFAULT_PEAK_SEED_OFFSET` (see `src/sgda_peak_selection.py`).
- Random peak selection uses a **dedicated NumPy RNG** seeded with
  `peak_location_seed`, so it does **not** consume the global RNG stream used
  elsewhere.

### 4) Training schedule parity
To ensure identical compute budgets across modes, early stopping is effectively
disabled by setting:
- `training_parameters.patience >= num_epochs + 1`

### 5) Outputs and evaluation
Each run is trained and evaluated with the standard `experiments/train.py`
pipeline, producing prediction CSVs, checkpoints, loss history, and metrics.
Aggregation across seeds is performed afterwards using
`experiments/peak_ablation_results.py` (the same aggregator used for peak-mode
ablation).

## Configuration knobs
Add these to `processing_parameters` (defaults shown):
```yaml
peak_mode: "mcsa"                # "mcsa" or "random"
peak_location_seed: null         # optional; if null uses seed + offset
random_peak_count_range: null    # [min, max] inclusive; used only in random mode
```

## Code-level walkthrough (key excerpts)

### A) Random peak counts are sampled per fault
When `random_peak_count_range` is provided and `peak_mode="random"`, the helper
draws a count per fault and then samples that many locations:

```python
# src/sgda_peak_selection.py
if random_peak_count_range is not None:
    min_peaks, max_peaks = ...
    n_peaks = rng.integers(min_peaks, max_peaks + 1)
else:
    n_peaks = len(freqs_list)
```

This guarantees that **random mode is the only mode with randomized peak counts**.

### B) The training pipeline passes the range through
`experiments/train.py` validates and passes the range to the selector:

```python
# experiments/train.py
random_peak_count_range = inj_cfg.get("random_peak_count_range", None)
fault_freqs_inject = select_peak_frequencies(
    fault_freqs_physics,
    freqs,
    peak_mode,
    rng_seed=peak_seed,
    margin_bins=peak_segment_bins,
    random_peak_count_range=random_peak_count_range,
)
```

### C) Run generation for the ablation
The ablation runner sets the range only for random mode:

```python
# experiments/run_peak_count_ablation.py
if mode == "random":
    proc_cfg["random_peak_count_range"] = [min_count, max_count]
else:
    proc_cfg.pop("random_peak_count_range", None)
```

## Run the ablation (5 seeds × 2 modes)
```cli
python experiments/run_peak_count_ablation.py \
  --cfg training_configs/train_engine-2.yml \
  --exp-name sgda_peak_count_ablation \
  --random-peak-count-range 1 10
```

## Parallel SLURM workflow (preferred)

1) Generate configs (writes `config_index.csv`):
```cli
python training_configs/sgda_peak_count_ablation/make_configs.py \
  --base-cfg training_configs/train_engine-2.yml \
  --out-dir training_configs/sgda_peak_count_ablation \
  --seeds 42 43 44 45 46 \
  --modes mcsa random \
  --random-peak-count-range 1 10 \
  --disable-early-stopping
```

2) Update the array in `training_configs/sgda_peak_count_ablation/submit_peak_count_ablation.sbatch`
   to match the generated filenames (or keep the defaults if names match).

3) Submit the job array:
```cli
sbatch training_configs/sgda_peak_count_ablation/submit_peak_count_ablation.sbatch
```

All runs are written under:
```
res/runs/sgda_peak_count_ablation/
```
with a per-run config copy and a `ablation_manifest.json`.

## Aggregate results across seeds
```cli
python experiments/peak_ablation_results.py \
  --runs-dir res/runs/sgda_peak_count_ablation \
  --plot-loss
```

Outputs are written to:
```
res/runs/sgda_peak_count_ablation/peak_ablation_results/
```
including per-run metrics, aggregated summaries, and comparison plots.
