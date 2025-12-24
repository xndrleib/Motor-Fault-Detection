# SGDA Peak-Mode Ablation

This ablation isolates the effect of **peak-location selection** in SGDA by
comparing physics-guided (MCSA) peak centers against uniformly random peak
centers while keeping everything else fixed.

## What is held constant
- Number of peaks per fault type (random uses the same count as MCSA).
- Peak shape parameters (amplitude, sigma, bandwidth, sign).
- Synthetic budget (K/R/p\_inject) and training schedule.
- Random seeds for all other randomness.

## Mathematical formulation (multiclass)

### Frequency-domain representation
Each time segment $x(t)$ is transformed into a magnitude spectrum:
$$
\mathbf{s} \in \mathbb{R}^L,\quad \mathbf{s}[i] = \lvert \mathrm{FFT}(x)(f_i) \rvert,
$$
with FFT frequency bins $\{f_i\}_{i=0}^{L-1}$, where $f_i \in [0, f_{\max}]$.

### Fault classes
Let the synthetic fault classes be
$$
\mathcal{T} = \{\text{ITSC},\ \text{RBD}\},
$$
with **Normal** being unmodified segments. Each synthetic sample is labeled
$$
y \in \{\text{Normal},\ \text{ITSC},\ \text{RBD}\}.
$$

### Anchor selection
For each fault type $t \in \mathcal{T}$:

**MCSA (SGDA)** uses physics-guided anchors
$$
\nu(\theta, t) = \{\nu_1, \ldots, \nu_{M_t}\},\quad M_t = |\nu(\theta, t)|.
$$

**Random Peaks (RP)** replaces anchors with random locations:
$$
\tilde{\nu}(t) = \{\tilde{f}_1, \ldots, \tilde{f}_{M_t}\},\quad \tilde{f}_j \sim \text{Uniform}(\{f_i: i \in \mathcal{I}_{\text{valid}}\}),
$$
where the valid index set avoids edge truncation for the Gaussian window:
$$
\mathcal{I}_{\text{valid}} = \{m, m+1, \ldots, L-1-m\}.
$$
Here $m$ is the **margin** in bins derived from `peak_segment`.

If `random_peak_count_range = [a,b]` is set, then
$$
M_t \sim \text{Uniform}\{a,\ldots,b\}\quad \text{(per fault)}.
$$
Otherwise $M_t$ is fixed to the MCSA count.

### Gaussian peak injection (shared)
For each anchor frequency $f^*$:

1) **Map to the closest bin**
$$
i^* = \arg\min_i |f_i - f^*|.
$$

2) **Define a window** of half-width $p$ (in bins):
$$
W = \{i^*-p, \ldots, i^*+p\},\quad
p = \left\lceil \frac{\texttt{peak_segment (Hz)}}{\Delta f} \right\rceil.
$$

3) **Sample Gaussian parameters**
$$
A \sim \text{Uniform}(A_{\min}, A_{\max}),\quad
\sigma \sim \text{Uniform}(\sigma_{\min}, \sigma_{\max}).
$$
If `random_peak_position=True`, the center is sampled uniformly within the window,
$$
\mu \sim \text{Uniform}\{0,\ldots,|W|-1\},
$$
otherwise $\mu = \lfloor |W|/2 \rfloor$. If `include_negative_peaks=True`, a sign
$s \in \{-1, +1\}$ is sampled.

4) **Inject into the spectrum**
$$
g(k) = A \exp\left(-\frac{(k-\mu)^2}{2\sigma^2}\right),
\quad k \in \{0,\ldots,|W|-1\},
$$
and
$$
\mathbf{s}'[W] \leftarrow \mathbf{s}[W] + s \cdot g.
$$

### What the ablation isolates
Everything is held fixed except **anchor selection**:
$$
\textbf{SGDA: } \nu(\theta, t)\quad \text{vs}\quad \textbf{RP: } \tilde{\nu}(t).
$$
This isolates whether gains come from **physics-guided frequency placement** rather than
generic spectral peak injection.

### Fixed vs per-sample random anchors
This matters only in `peak_mode="random"`:

- `random_peak_sampling: fixed` samples $\tilde{\nu}(t)$ once per run and reuses it.
- `random_peak_sampling: per_sample` resamples $\tilde{\nu}(t)$ for each synthetic window.
  With `cache=true`, resampling happens once per epoch (cached dataset); with
  `cache=false`, resampling happens per access.

## Configuration knobs
Add these to `processing_parameters` (defaults shown):
```yaml
peak_mode: "mcsa"          # "mcsa" or "random"
peak_location_seed: null   # optional; if null uses seed + offset
random_peak_sampling: fixed  # fixed or per_sample (random mode only)
```

Notes:
- Random peaks are sampled **uniformly** from FFT bins with a margin of
  `peak_segment` (in bins) to avoid truncated peak windows.
- With `random_peak_sampling: per_sample`, random peak locations are resampled
  for every injected synthetic window.
- Attention masks are disabled for this ablation.

## Run the ablation (5 seeds × 2 modes)
```cli
python experiments/run_peak_ablation.py \
  --cfg training_configs/train_engine-2.yml \
  --exp-name sgda_peak_ablation
```

## Multiclass peak ablation (Normal / ITSC / RBD)
Use the same base config and override the task when generating configs:
```cli
python training_configs/sgda_peak_ablation/make_configs.py \
  --base-cfg training_configs/train_engine-2.yml \
  --out-dir training_configs/sgda_peak_ablation_multiclass \
  --task multiclass \
  --seeds 42 43 44 45 46 \
  --modes mcsa random \
  --disable-early-stopping
```

Then submit as usual (see SLURM workflow) or run a single config locally:
```cli
cd experiments
python train.py --cfg ../training_configs/sgda_peak_ablation/<generated>.yml --exp-name sgda_peak_ablation_multiclass
```

## Parallel SLURM workflow (preferred)

1) Generate configs (writes `config_index.csv`):
```cli
python training_configs/sgda_peak_ablation/make_configs.py \
  --base-cfg training_configs/train_engine-2.yml \
  --out-dir training_configs/sgda_peak_ablation \
  --seeds 42 43 44 45 46 \
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
