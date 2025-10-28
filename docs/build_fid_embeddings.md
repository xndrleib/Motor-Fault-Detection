# `embed_fid_from_run.py` — Documentation

## 1) Purpose

Produce fixed-length **embeddings** for every windowed FFT spectrum saved for FID evaluation. The script:

* Reconstructs your model using the run’s `training_config.yaml`.
* Restores **weights**, **normalizer**, and (for multiclass) **label encoder**.
* Applies the model to windows from `res/fid_inputs/`.
* Captures the **penultimate** feature vector (input to the final classifier).
* Saves embeddings to `res/fid_embs/<run_id>/`, one file per source `.npy`.

These embeddings are then ready for downstream FID computation (e.g., per cluster, per group).

---

## 2) Quick start

```bash
python tools/embed_fid_from_run.py \
  --run-dir res/runs/2025-06-03_02-14-40_train_full-data-removeES-42-16 \
  --fid-dir res/fid_inputs \
  --out-root res/fid_embs \
  --batch-size 2048 \
  --device auto
```

On success you’ll see lines like:

```
[✓]        synth_100_RBD.npy → (N, D) saved at res/fid_embs/<run_id>/synth_100_RBD_embs.npy
```

---

## 3) Inputs & expected structure

### 3.1 Run directory (required)

`--run-dir` must point to a **single training run** folder that contains the artifacts below:

```
res/runs/<run_id>/
├── checkpoints/
│   └── best_multiclass.pth        # or best_binary.pth / .pt (auto-detected)
├── training_config.yaml           # REQUIRED (drives model & data params)
├── normalizer_multiclass.json     # REQUIRED for mode=global (auto-detected)
├── label_encoder_multiclass.pkl   # REQUIRED for multiclass (joblib-compressed)
└── ... (logs, figs, indices, etc.)
```

Notes:

* **Checkpoint**: the script prefers `best_multiclass.pth` or `best_binary.pth`, then falls back to the first `*.pth` / `*.pt` in `checkpoints/`.
* **Normalizer**: discovered as `normalizer_multiclass.json` (or `normalizer.json`) and loaded via your `src.normalization.Normalizer`. For `mode=global`, this file is **required**.
* **Label encoder** (multiclass only): discovered as `label_encoder_multiclass.pkl` (joblib-compressed). Loaded with `joblib.load` and must expose `classes_`.

### 3.2 FID windows directory (required)

`--fid-dir` points to the folder of window arrays:

```
res/fid_inputs/
├── real_<LOAD>_<FAULTCODE>.npy      # e.g., real_40_RBD.npy
├── synth_<LOAD>_<FAULTCODE>.npy     # e.g., synth_100_ITSC.npy
└── engine_<ID>_freqs.npy            # optional, not used by this script
```

Each `real_*.npy` / `synth_*.npy` must be a 2-D array of shape:

```
(num_windows, segment_length)
```

The script probes the **first** file to determine `segment_length` (L) and checks all others for consistency.

> Empty arrays are allowed but **skipped** with a clear message.

---

## 4) What the script does

1. **Parse config** from `training_config.yaml` to determine:

   * `task`: `binary` or `multiclass`
   * `model`: `ResNet` / `CNN` / `MLP` (mapped internally to `resnet18`, `cnn`, `mlp`)
   * `model_parameters.dropout` and `model_parameters.attention_module`
   * Dataset normalization: `dataset_parameters.normalization_method` (`min-max` / `z-score`) and `dataset_parameters.normalization_mode` (`global` / `per`)
   * Default `batch_size` (can be overridden by CLI)

2. **Instantiate the model**:

   * `CNN` or `ResNet(18)` produces a 512-dimensional penultimate vector by design; `MLP` uses the last hidden dimension (default 128).
   * If `attention_module: true`, a **neutral** prior-attention block is constructed (zero mask with unit weights) so checkpoints load safely without changing behavior.

3. **Load artifacts**:

   * Checkpoint: various formats supported (`state_dict`, `model_state_dict`, whole module).
   * Normalizer: global/per behavior is preserved; for `global`, the saved stats file is required.
   * Label encoder (multiclass): loaded via `joblib` and used to infer `num_classes`.

4. **Discover input files** in `--fid-dir` matching `real_*.npy` and `synth_*.npy`.

5. **Embed**:

   * Applies normalizer (global/per) to each batch.
   * Uses a **classifier-swap** context so `model(x)` returns the penultimate features (no forward hooks).
   * Streams batches on CPU/GPU as requested.

6. **Save outputs** to `--out-root/<run_id>/` with `_embs.npy` suffix.

---

## 5) How embeddings are extracted (no hooks)

To avoid the global-state pitfalls of forward hooks, the script temporarily replaces the model’s final classifier (`.fc` or `.classifier`) with `nn.Identity()` inside a context manager. This makes the forward pass return the **pre-classifier features** directly (the penultimate vector), then restores the original layer. It’s simple, robust, and production-friendly.

---

## 6) CLI arguments

* `--run-dir` (**required**): path to a run folder under `res/runs/…`.
* `--fid-dir` (default `res/fid_inputs`): where `real_*.npy` / `synth_*.npy` live.
* `--out-root` (default `res/fid_embs`): root output folder; embeddings go into `<out-root>/<run_id>/`.
* `--batch-size` (optional): override; defaults to training `batch_size` or `1024`.
* `--device` (`auto` | `cpu` | `cuda`; default `auto`): choose compute device. `auto` uses `cuda` if available.

---

## 7) Outputs

For every input `*.npy`:

```
res/fid_embs/<run_id>/<stem>_embs.npy
```

Examples:

* `res/fid_embs/2025-06-03_02-14-40_train_full-data-removeES-42-16/synth_100_RBD_embs.npy`
* `res/fid_embs/2025-06-03_02-14-40_train_full-data-removeES-42-16/real_40_RBD_embs.npy`

Each output file is a `float32` NumPy array of shape:

```
(num_windows, D)
```

Where **D** is the penultimate dimensionality:

* `CNN` / `ResNet18`: typically **512**
* `MLP`: last hidden size (default **128**, given `[512, 256, 128]`)

> The script prints the shape `(N, D)` for each saved file.

---

## 8) Behavior on empty or malformed inputs

* If an input file is **empty** (0 rows) or has unexpected shape, it is **skipped** and a message is printed, e.g.:

  * `[skip] synth_60_ITSC.npy is empty (0 windows).`
  * `[skip] real_80_RBD.npy has shape (L,), expected (N, L).`

* The embedder also early-returns on empty arrays, so even direct calls won’t crash.

---

## 9) Normalization details

* The script rebuilds your `Normalizer(method, mode)` from the config.
* `mode='global'`: the saved stats JSON (e.g., `normalizer_multiclass.json`) is **required** and applied to each batch.
* `mode='per'`: per-window normalization is applied on the fly.
* Normalization is performed **before** feeding the batch to the model.

---

## 10) Model & checkpoint loading

* Architecture is derived from `training_config.yaml` (`model: ResNet/CNN/MLP`).
* `num_classes`:

  * `binary` ⇒ 2.
  * `multiclass` ⇒ inferred from the joblib-loaded `label_encoder_*` via `len(classes_)`.
* Weights are restored from common patterns:

  * `state_dict`, `model_state_dict`, whole module, or a dict containing such.
* Loading uses `strict=False` to tolerate benign differences (e.g., presence/absence of prior attention).

---

## 11) Device & performance notes

* `--device auto` prefers GPU if available; otherwise CPU.
* `--batch-size` defaults to the run’s original batch size (or 1024 if not present). You can bump this up/down depending on GPU memory.
* All tensors are `float32`, and inference runs with `torch.no_grad()` in `eval()` mode.

---

## 12) Errors & troubleshooting

You’ll get clear exceptions for missing/invalid artifacts:

* **Missing config**: `Missing training_config.yaml at <path>`
* **Missing checkpoint**: `No checkpoint found in <.../checkpoints>`
* **Global normalization but no stats file**: `Global normalization selected but stats file not found`
* **Multiclass but no label encoder** (or malformed): a `FileNotFoundError` or `AttributeError` explaining what’s missing.

Run-time skips:

* **Empty file**: prints `[skip] <name> is empty (0 windows).`
* **Wrong shape**: prints `[skip] <name> has shape <...>; expected (*, L).`

---

## 13) Assumptions & conventions

* FID windows follow the naming convention:

  * `real_<LOAD>_<FAULTCODE>.npy` (e.g., `real_40_RBD.npy`)
  * `synth_<LOAD>_<FAULTCODE>.npy` (e.g., `synth_100_ITSC.npy`)
* Arrays are 2-D: `(num_windows, segment_length)`; consistent `segment_length` across all files in `--fid-dir`.
* The run’s normalizer reflects how the model was trained; the script does **not** recompute any stats.

---

## 14) Minimal example end-to-end

```bash
# 1) Ensure run artifacts exist:
#    res/runs/<run_id>/training_config.yaml
#    res/runs/<run_id>/checkpoints/best_multiclass.pth
#    res/runs/<run_id>/normalizer_multiclass.json
#    res/runs/<run_id>/label_encoder_multiclass.pkl   # joblib-compressed

# 2) Ensure FID inputs exist:
#    res/fid_inputs/real_40_RBD.npy
#    res/fid_inputs/synth_100_ITSC.npy
#    ... (2D arrays shaped (N, L))

# 3) Run:
python tools/embed_fid_from_run.py \
  --run-dir res/runs/<run_id> \
  --fid-dir res/fid_inputs \
  --out-root res/fid_embs \
  --device auto

# 4) Outputs:
#    res/fid_embs/<run_id>/real_40_RBD_embs.npy   # (N, D)
#    res/fid_embs/<run_id>/synth_100_ITSC_embs.npy
```
