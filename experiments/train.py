# Code for training models
"""End-to-end training & evaluation script with artifact saving.

This module orchestrates data preparation, model training/validation, test-time
inference, and artifact logging. It augments the test predictions CSV with
per-class softmax scores and persists arrays (logits, probabilities, embeddings,
labels, predictions, and test indices) to compressed ``.npz`` files.

Outputs
-------
In the run directory (``exp_dir``), this script saves:
- ``segments_metadata_test_[binary|multiclass]_pred.csv`` : predictions + scores
- ``test_arrays_[binary|multiclass].npz`` : probs/logits/embeddings/labels/preds/index
- ``test_embeddings_[binary|multiclass].npy`` : embeddings only
- ``test_logits_[binary|multiclass].npy`` : logits only
- ``test_probs_[binary|multiclass].npy`` : softmax probabilities only
- Confusion matrix figures (PNG/PDF) and optional importance mask figure

Directory layout
----------------
- Default: results go to ``<repo_root>/res/runs/{run-name}/`` where ``run-name`` is
  derived from the config base name + timestamp.
- If ``--exp-name`` is provided: ``<repo_root>/res/runs/{exp-name}/{run-name}/``.

Notes
-----
- ``inference_model`` is invoked with ``return_extra=True`` to collect
  logits/probabilities/embeddings.
- Embeddings come from ``model.forward_features`` if implemented.
"""
from __future__ import annotations

import argparse
import datetime
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from src.anomaly_injector import (
    GaussianPeakInjector,
    NoiseInjector,
    CompositeAnomalyInjector,
)
from src.data_pipeline import preprocessing, make_importance_mask, filter_segments
from src.datasets import (
    create_balanced_datasets,
    FaultInjectionDataset,
    AugmentedPoolDataset,
    HybridAugFaultDataset,
)
from src.electrical_signature_frequencies import ANOMALY_FREQS
from src.evaluation import calculate_metrics
from src.inference import inference_model
from src.models import ResNet, ResidualBlock, CNN
from src.normalization import Normalizer
from src.train import train_epoch_cached
from src.sgda_peak_selection import (
    RandomPeakSampler,
    build_random_peak_samplers,
    resolve_peak_location_seed,
    select_peak_frequencies,
)
from src.noise_policy import NoisePolicy
from src.utils import set_all_seeds, load_yaml
from src.experiment_logging import LocalExperiment


# ---- Paths & logging ---------------------------------------------------------


def _repo_root() -> Path:
    """Return repository root assuming this file is at <repo_root>/experiments/train.py."""
    return Path(__file__).resolve().parents[1]


def setup_logger(log_dir: str, log_file: str = "training.log") -> None:
    """Configure root logger to write to file and console.

    Parameters
    ----------
    log_dir : str
        Directory where the log file will be created.
    log_file : str, default="training.log"
        Log filename inside ``log_dir``.

    Returns
    -------
    None
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    # Configure root logger
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Avoid duplicate handlers in repeated runs
    for h in list(root.handlers):
        root.removeHandler(h)

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    root.addHandler(ch)

    root.info(f"Logging to file: {log_path}")


def load_configurations() -> dict:
    """Load general experiment configuration from ``<repo_root>/cfg.yaml``.

    Returns
    -------
    dict
        Parsed YAML configuration.
    """
    cfg_path = _repo_root() / "cfg.yaml"
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)
    logging.info(f"Loaded main configuration from {cfg_path}.")
    return config


def start_experiment(
    config: dict, online: bool = True, name: str | None = None, local_dir: Path | None = None
) -> Tuple[comet_ml.CometExperiment, str]:
    """Start a Comet ML experiment and log source code.

    Parameters
    ----------
    config : dict
        Dictionary with keys ``API_KEY``, ``PROJECT_NAME``, and ``WORKSPACE``.
    online : bool, default=True
        If ``False``, runs offline (local logging only).
    name :


    Returns
    -------
    (Experiment, str)
        The Comet experiment handle and the timestamped run name.
    """
    if not online:
        return LocalExperiment(Path(local_dir) / "experiment.jsonl"), name or "offline"
    import comet_ml
    experiment = comet_ml.start(
        api_key=config["API_KEY"],
        project_name=config["PROJECT_NAME"],
        workspace=config["WORKSPACE"],
        online=online,
    )
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    experiment.set_name(name if name else f"Script Run: {now}")
    experiment.log_code()
    logging.info(f"Started Comet ML experiment at {now}")
    return experiment, now


def prepare_directories(
    engine: str,
    runs_dir: str | Path | None = None,
    exp_name: str | None = None,
    run_name: str | None = None,
    dataset_dir: str | Path | None = None,
):
    """Create output directories for experiment artifacts.

    Behavior
    --------
    - Root runs dir defaults to ``<repo_root>/res/runs``.
    - If ``exp_name`` is None → ``exp_root = runs_dir/`` else ``runs_dir/exp_name/``.
    - ``exp_dir`` (the run folder) = ``exp_root/run_name/``.
    """
    repo = _repo_root()
    runs_root = Path(runs_dir) if runs_dir is not None else (repo / "res" / "runs")
    runs_root.mkdir(parents=True, exist_ok=True)

    if not run_name:
        raise ValueError("prepare_directories requires a non-empty run_name.")

    exp_root = runs_root / exp_name if exp_name else runs_root
    exp_dir = exp_root / run_name
    exp_dir.mkdir(parents=True, exist_ok=False)

    # dataset base (under repo root)
    base_dir = Path(dataset_dir).resolve() if dataset_dir else repo / "dataset" / engine.replace("-", "_")

    # standard subfolders
    indices_dir = exp_dir / "indices"
    checkpoints_dir = exp_dir / "checkpoints"
    fig_dir = exp_dir / "figs"
    log_dir = exp_dir / "logs"
    for d in [indices_dir, checkpoints_dir, fig_dir, log_dir]:
        d.mkdir(exist_ok=True, parents=True)

    logging.info(f"Run directory created at {exp_dir.resolve()}")
    return base_dir, exp_dir, indices_dir, checkpoints_dir, fig_dir, log_dir


# ---- Helpers -----------------------------------------------------------------


def save_label_encoder(enc, exp_dir: str | Path) -> None:
    """Persist label encoder used for multiclass tasks."""
    enc_path = Path(exp_dir) / "label_encoder_multiclass.pkl"
    joblib.dump(enc, enc_path, compress=3)
    logging.info(f"Saved LabelEncoder to {enc_path.resolve()}")


def plot_and_save_confusion_matrix(
    cm, labels, fig_dir: Path, fname_prefix: str, experiment
) -> None:
    """Plot, save, and log a confusion matrix."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    fpath_png = fig_dir / f"{fname_prefix}.png"
    fpath_pdf = fig_dir / f"{fname_prefix}.pdf"
    plt.savefig(fpath_png, dpi=300)
    plt.savefig(fpath_pdf, dpi=300)
    experiment.log_figure(figure=fig, figure_name=f"{fname_prefix}.png")
    plt.close(fig)
    logging.info(f"Confusion matrix plotted and saved: {fpath_png}")


def save_configs(
    train_cfg_path: str | Path, engine_cfg_path: str | Path, exp_dir: str | Path
) -> None:
    """Copy training and engine configuration files into the run folder."""
    train_cfg_path = Path(train_cfg_path)
    engine_cfg_path = Path(engine_cfg_path)
    exp_dir = Path(exp_dir)

    if not train_cfg_path.is_file():
        raise FileNotFoundError(f"Training config not found: {train_cfg_path}")
    if not engine_cfg_path.is_file():
        raise FileNotFoundError(f"Engine config not found:   {engine_cfg_path}")

    exp_dir.mkdir(parents=True, exist_ok=True)
    dest_train = exp_dir / "training_config.yaml"
    dest_engine = exp_dir / "engine_config.yaml"

    shutil.copy2(train_cfg_path, dest_train)
    logging.info(f"Copied training config to {dest_train}")

    shutil.copy2(engine_cfg_path, dest_engine)
    logging.info(f"Copied engine   config to {dest_engine}")


# ---- Train & Eval ------------------------------------------------------------


def train_and_eval(
    model,
    datasets,
    seg_meta_df: pd.DataFrame,
    val_loader,
    test_loader,
    device: torch.device,
    train_params: dict,
    checkpoints_dir: Path,
    exp_dir: Path,
    fig_dir: Path,
    experiment,
    task: str,
) -> None:
    """Train the model, evaluate on test set, and save artifacts."""
    logging.info(f"Training model for {task} classification...")

    num_epochs = (
        train_params["training_parameters"]["num_epochs_binary"]
        if task == "binary"
        else train_params["training_parameters"]["num_epochs_multi"]
    )

    # ── Training ───────────────────────────────────────────────────────────────
    train_start = time.perf_counter()
    model = train_epoch_cached(
        model=model,
        cached_dataset=datasets["train"],
        val_loader=val_loader,
        device=device,
        num_epochs=num_epochs,
        batch_size=train_params["data_parameters"]["batch_size"],
        initial_lr=train_params["training_parameters"]["initial_lr"],
        patience=train_params["training_parameters"]["patience"],
        checkpoint_path=checkpoints_dir,
        history_path=exp_dir / "loss_history.csv",
    )
    train_time = time.perf_counter() - train_start
    logging.info(f"Training complete. Time elapsed: {train_time:.2f} seconds.")
    experiment.log_metric(f"{task}_train_time_seconds", train_time)

    # ── Inference ──────────────────────────────────────────────────────────────
    logging.info("Starting inference on test set.")
    inference_start = time.perf_counter()
    res = inference_model(model, test_loader, device=device, return_extra=True)
    true_labels = res["labels"]
    predictions = res["preds"]
    probs = res["probs"]  # (N, C)
    logits = res["logits"]  # (N, C)
    embeddings = res["embeddings"]  # (N, D) or None
    inference_time = time.perf_counter() - inference_start
    logging.info(f"Inference complete. Time elapsed: {inference_time:.2f} seconds.")
    experiment.log_metric(f"{task}_inference_time_seconds", inference_time)

    # ── Metrics & confusion matrix ────────────────────────────────────────────
    if task == "binary":
        target_names = ["Normal", "Anomalous"]
    else:
        short_names = {
            "bearing defect": "BD",
            "inter-turn short circuits": "ITSC",
            "rotor bar defect": "RBD",
            "normal": "Normal",
        }
        target_names = [
            short_names[name] for name in datasets["label_encoder"].classes_
        ]

    cm, acc, prec, rec, f1 = calculate_metrics(true_labels, predictions)
    report = classification_report(
        true_labels,
        predictions,
        target_names=target_names,
        output_dict=(task == "multiclass"),
    )
    if task == "multiclass":
        experiment.log_metrics({f"multiclass_{k}": v for k, v in report.items()})

    logging.info(
        f"{task.capitalize()} Classification Report:\n"
        f"{classification_report(true_labels, predictions, target_names=target_names)}"
    )
    logging.info(
        f"{task.capitalize()} Test Set Accuracy: {acc:.4f} | "
        f"Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}"
    )
    plot_and_save_confusion_matrix(
        cm, target_names, fig_dir, f"confusion_matrix_{task}", experiment
    )

    # ── Build prediction CSV with scores ──────────────────────────────────────
    predictions = np.asarray(predictions, dtype=np.int64)
    test_idx = datasets["test_idx"]
    meta_test = seg_meta_df.loc[test_idx].copy()

    if task == "binary":
        meta_test["binary_prediction"] = predictions
        meta_test["binary_prediction_state"] = np.where(
            predictions == 0, "normal", "anomalous"
        )
        meta_test["score_normal"] = probs[:, 0]
        meta_test["score_anomalous"] = probs[:, 1]
        meta_test["score_max"] = probs.max(axis=1)

        out_path = exp_dir / "segments_metadata_test_binary_pred.csv"
        model_name = "bin_model"
        model_file = checkpoints_dir / "best_binary.pth"
    else:
        # Save encoder for downstream analysis
        save_label_encoder(datasets["label_encoder"], exp_dir)

        meta_test["multiclass_prediction"] = predictions
        inv = datasets["label_encoder"].inverse_transform(predictions)
        meta_test["multiclass_prediction_state"] = inv

        # Per-class probability columns in encoder order
        short_names = {
            "bearing defect": "BD",
            "inter-turn short circuits": "ITSC",
            "rotor bar defect": "RBD",
            "normal": "Normal",
        }
        class_names = list(datasets["label_encoder"].classes_)
        for c_idx, cname in enumerate(class_names):
            col = f"score_{short_names.get(cname, cname)}"
            meta_test[col] = probs[:, c_idx]
        meta_test["score_max"] = probs.max(axis=1)

        out_path = exp_dir / "segments_metadata_test_multiclass_pred.csv"
        model_name = "multi_model"
        model_file = checkpoints_dir / "best_multiclass.pth"

    meta_test.to_csv(out_path, index=True)
    logging.info(f"{task.capitalize()} test predictions (+scores) saved to: {out_path}")

    # ── Persist arrays (npz + convenience npy) ────────────────────────────────
    npz_path = exp_dir / f"test_arrays_{task}.npz"
    np.savez_compressed(
        npz_path,
        probs=probs,
        logits=logits,
        embeddings=(embeddings if embeddings is not None else np.empty((0,))),
        labels=true_labels,
        preds=predictions,
        test_index=meta_test.index.values,
    )
    np.save(
        exp_dir / f"test_embeddings_{task}.npy",
        embeddings if embeddings is not None else np.empty((0,)),
    )
    np.save(exp_dir / f"test_logits_{task}.npy", logits)
    np.save(exp_dir / f"test_probs_{task}.npy", probs)
    logging.info(f"Saved logits/probs/embeddings to: {npz_path}")

    # Log artifacts to Comet (best-effort)
    try:
        experiment.log_asset(str(npz_path), file_name=npz_path.name)
        experiment.log_asset(str(out_path), file_name=out_path.name)
    except Exception as e:
        logging.warning(f"Could not log assets to Comet: {e}")

    # ── Log model checkpoint path to Comet ────────────────────────────────────
    experiment.log_model(name=model_name, file_or_folder=model_file)
    logging.info(f"{task.capitalize()} model logged to experiment: {model_file}")


# ---- Main --------------------------------------------------------------------


def main() -> None:
    """CLI entry-point: load configs, prepare data, train/eval, and log artifacts."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate model. Reads 'engineLabel' and 'task' from training config."
    )
    parser.add_argument(
        "--task",
        choices=["binary", "multiclass"],
        help="[optional] Override: Task to train: binary or multiclass",
    )
    parser.add_argument(
        "--cfg", type=str, required=True, help="Path to training config YAML"
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Optional experiment name. If set, results go to <repo_root>/res/runs/{exp-name}/{run-name}.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name. Defaults to <cfg basename>_<timestamp>.",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=None,
        help="Override runs root directory. Default is <repo_root>/res/runs.",
    )
    parser.add_argument("--offline", action="store_true", help="Local logs only; no Comet configuration or network.")
    parser.add_argument("--dataset-dir", type=str, help="Directory containing engine.yml and metadata.csv.")
    parser.add_argument("--path-base", type=str, help="Base for relative metadata file_path values.")
    parser.add_argument("--prepared-dir", type=str, help="Verified output of reporting.cli prepare.")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], help="Explicit execution device.")
    args = parser.parse_args()

    # Load training configuration
    train_params = load_yaml(args.cfg)
    engine = train_params.get("engineLabel", None)
    task_from_cfg = train_params.get("task", None)
    task = args.task or task_from_cfg
    if args.offline:
        train_params["comet_online"] = False
    train_params["task"] = task
    if not task:
        raise RuntimeError(
            "Task must be specified in training configuration (task: binary|multiclass) or via --task CLI argument."
        )

    # Derive a unique run name if not provided: <cfg stem>_<YYYY-mm-dd_HH-MM-SS>
    cfg_stem = Path(args.cfg).stem
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = args.run_name if args.run_name else f"{cfg_stem}_{timestamp}"

    # Prepare directories & logger
    base_dir, exp_dir, indices_dir, checkpoints_dir, fig_dir, log_dir = prepare_directories(
        engine=engine, runs_dir=args.runs_dir, exp_name=args.exp_name, run_name=run_name,
        dataset_dir=args.dataset_dir,
    )
    save_configs(args.cfg, base_dir / "engine.yml", exp_dir)
    (exp_dir / "training_config.yaml").write_text(yaml.safe_dump(train_params, sort_keys=False))
    setup_logger(str(log_dir))
    logging.info(
        f"Starting Training/Evaluation Script for engine '{engine}', task '{task}'"
    )
    logging.info(f"Runs root: {(Path(args.runs_dir) if args.runs_dir else (_repo_root() / 'res' / 'runs')).resolve()}")
    logging.info(f"Experiment: {args.exp_name or '(none)'} | Run: {run_name}")

    # Start Comet and set seeds/device
    online = train_params.get("comet_online", True)
    config = load_configurations() if online else {}
    comet_name = f"{args.exp_name}/{run_name}" if args.exp_name else run_name
    experiment, _ = start_experiment(
        config, online, name=comet_name, local_dir=exp_dir
    )

    set_all_seeds(train_params.get("seed", 42))
    device_str = args.device or (
        "cuda"
        if torch.cuda.is_available()
        else (
            "mps"
            if getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
            else "cpu"
        )
    )
    experiment.log_parameter("device", device_str)
    device = torch.device(device_str)
    logging.info(f"Using device: {device}")

    experiment.log_parameter("exp_dir", str(exp_dir))
    experiment.log_parameter("engine", engine)
    experiment.log_parameter("task", task)
    experiment.log_parameter("exp_name", args.exp_name or "")
    experiment.log_parameter("run_name", run_name)

    # Load engine config & metadata
    engine_config = load_yaml(base_dir / "engine.yml")
    experiment.log_parameters(engine_config)
    metadata_df = pd.read_csv(base_dir / "metadata.csv").set_index("measurement_id")
    path_base = Path(args.path_base).resolve() if args.path_base else _repo_root() / "experiments"
    metadata_df["file_path"] = [str((path_base / str(p)).resolve()) for p in metadata_df["file_path"]]
    experiment.log_parameters(
        {"num_of_measurements": metadata_df["state"].value_counts().to_dict()}
    )
    experiment.log_parameters(train_params)

    # Frequencies for augmentation
    fault_types_to_use = train_params["processing_parameters"]["fault_types_to_use"]
    MCSA_cfg = {
        "rotor bar defect": {
            "engine_config": engine_config["mcsa"],
            "n_range": range(1, 4),
        },
        "inter-turn short circuits": {
            "engine_config": engine_config["mcsa"],
            "k_range": range(1, 4, 2),
            "m_range": range(0, 2),
        },
        "bearing defect": {"engine_config": engine_config["mcsa"]},
    }
    fault_freqs_physics = {
        ft: np.asarray(ANOMALY_FREQS[ft](**MCSA_cfg[ft]), dtype=float)
        for ft in fault_types_to_use
    }
    experiment.log_parameters(
        {
            f"mcsa_freqs_{ft}": fault_freqs_physics[ft].tolist()
            for ft in fault_freqs_physics
        }
    )
    logging.info(f"Fault frequencies calculated (MCSA): {fault_freqs_physics}")

    # Preprocessing (segmentation + spectrum)
    inj_cfg = train_params["processing_parameters"]
    test_run = train_params.get("test_run", False)
    if test_run:
        indices = np.random.choice(
            metadata_df.shape[0], size=int(metadata_df.shape[0] * 0.05), replace=False
        )
    else:
        indices = np.arange(metadata_df.shape[0])

    train_noise_cfg = inj_cfg.get("train_noise_policy", {}) or {}
    noise_enabled = bool(train_noise_cfg.get("enabled", False))

    if noise_enabled:
        logging.info("Training noise policy enabled.")
        noise_policy = NoisePolicy.from_config(train_noise_cfg)
        experiment.log_parameter("train_noise_policy_enabled", True)
        logging.info("Train noise policy: %s", train_noise_cfg)
    else:
        noise_policy = None
        experiment.log_parameter("train_noise_policy_enabled", False)

    if args.prepared_dir:
        from reporting.core import load_prepared
        if noise_enabled:
            raise ValueError("Prepared spectra do not contain time windows for the noise policy.")
        prepared_x, prepared_meta, prepared_freqs, prepared_manifest = load_prepared(args.prepared_dir)
        if prepared_manifest["processing_parameters"] != inj_cfg:
            raise ValueError("Prepared processing parameters differ from the training config.")
        prep_out = (prepared_x, prepared_meta, prepared_freqs)
        shutil.copy2(Path(args.prepared_dir) / "manifest.json", exp_dir / "prepared_manifest.json")
    else:
        prep_out = preprocessing(
        metadata_df=metadata_df.iloc[indices],
        out_dir=exp_dir / "preprocessed",
        segment_length=inj_cfg["segment_length"],
        step=inj_cfg["shift"],
        f_sampling=inj_cfg["f_sampling"],
        cutoff_freq=inj_cfg["cutoff_freq"],
        apply_window=False,
        db=inj_cfg["db"],
        return_time_segments=noise_enabled,
    )
    if noise_enabled:
        segments, seg_meta_df, freqs, time_segments = prep_out
    else:
        segments, seg_meta_df, freqs = prep_out
        time_segments = None

    # Data injectors
    freq_resolution = freqs[1] - freqs[0]
    peak_segment_bins = int(np.ceil(inj_cfg["peak_segment"] / freq_resolution))
    logging.info(
        f"Converted peak segment from Hz to bins: {inj_cfg['peak_segment']} Hz -> {peak_segment_bins}"
    )

    peak_mode = str(inj_cfg.get("peak_mode", "mcsa")).lower()
    random_peak_sampling = str(inj_cfg.get("random_peak_sampling", "fixed")).lower()
    if random_peak_sampling in {"per-sample", "per_sample"}:
        random_peak_sampling = "per_sample"
    if random_peak_sampling not in {"fixed", "per_sample"}:
        raise ValueError(
            "processing_parameters.random_peak_sampling must be 'fixed' or 'per_sample'."
        )
    random_peak_count_range = inj_cfg.get("random_peak_count_range", None)
    if random_peak_count_range is not None:
        if (
            not isinstance(random_peak_count_range, (list, tuple))
            or len(random_peak_count_range) != 2
        ):
            raise ValueError(
                "processing_parameters.random_peak_count_range must be a 2-element list/tuple."
            )
        random_peak_count_range = (
            int(random_peak_count_range[0]),
            int(random_peak_count_range[1]),
        )
        if random_peak_count_range[0] < 1 or random_peak_count_range[1] < random_peak_count_range[0]:
            raise ValueError(
                "processing_parameters.random_peak_count_range must be >= 1 and min <= max."
            )
        if peak_mode != "random":
            logging.info(
                "random_peak_count_range is set but peak_mode=%s; value will be ignored.",
                peak_mode,
            )
    if peak_mode != "random" and random_peak_sampling != "fixed":
        logging.info(
            "random_peak_sampling=%s ignored because peak_mode=%s.",
            random_peak_sampling,
            peak_mode,
        )
    peak_seed = resolve_peak_location_seed(
        train_params.get("seed", 42),
        inj_cfg.get("peak_location_seed", None),
    )
    if peak_mode == "random" and peak_seed is None:
        raise ValueError("peak_mode='random' requires a deterministic peak_location_seed.")

    if peak_mode == "random" and random_peak_sampling == "per_sample":
        fault_freqs_inject = build_random_peak_samplers(
            fault_freqs_physics,
            rng_seed=peak_seed,
            margin_bins=peak_segment_bins,
            random_peak_count_range=random_peak_count_range,
        )
    else:
        fault_freqs_inject = select_peak_frequencies(
            fault_freqs_physics,
            freqs,
            peak_mode,
            rng_seed=peak_seed,
            margin_bins=peak_segment_bins,
            random_peak_count_range=random_peak_count_range,
        )
    experiment.log_parameter("peak_mode", peak_mode)
    experiment.log_parameter("peak_location_seed", peak_seed)
    experiment.log_parameter("random_peak_sampling", random_peak_sampling)
    peak_ranges = {}
    peak_counts = {}
    for ft, freqs_list in fault_freqs_inject.items():
        if isinstance(freqs_list, RandomPeakSampler):
            if freqs_list.peak_count_range is not None:
                peak_ranges[ft] = list(freqs_list.peak_count_range)
            if freqs_list.peak_count is not None:
                peak_counts[ft] = int(freqs_list.peak_count)
        else:
            peak_counts[ft] = len(freqs_list)
    if peak_counts:
        experiment.log_parameters(
            {f"n_peaks_{ft}": count for ft, count in peak_counts.items()}
        )
    if peak_ranges:
        experiment.log_parameters(
            {f"n_peaks_range_{ft}": rng for ft, rng in peak_ranges.items()}
        )
    logging.info(
        "Peak selection mode: %s | peak_location_seed=%s", peak_mode, peak_seed
    )
    logging.info("Per-fault peak counts: %s", peak_counts)
    if peak_ranges:
        logging.info("Per-fault peak count ranges: %s", peak_ranges)

    gaussian_injector = GaussianPeakInjector(
        peak_segment=peak_segment_bins,
        amplitude_range=tuple(inj_cfg["amplitude_range"]),
        sigma_range=tuple(inj_cfg["sigma_range"]),
        negative=inj_cfg["include_negative_peaks"],
        random_peak_position=inj_cfg["random_peak_position"],
    )
    noise_injector = NoiseInjector(noise_factor=inj_cfg["noise_factor"])
    composite_injector = CompositeAnomalyInjector(
        {"peak-anomaly": gaussian_injector, "noise": noise_injector}
    )
    logging.info("Data injectors instantiated.")

    # Filtering
    logging.info(
        f"Applying filtering to segments: {segments.shape}, and metadata: {seg_meta_df.shape}"
    )
    training_classes = inj_cfg["fault_types_to_use"] + ["normal"]
    loads_to_use = inj_cfg["loads_to_use"]
    phases_to_use = inj_cfg["phases_to_use"]

    if noise_enabled:
        seg_meta_df, segments, mask = filter_segments(
            seg_meta_df,
            segments,
            training_classes,
            loads_to_use,
            phases_to_use,
            return_mask=True,
        )
        time_segments = time_segments[mask] if time_segments is not None else None
    else:
        seg_meta_df, segments = filter_segments(
            seg_meta_df, segments, training_classes, loads_to_use, phases_to_use
        )
    logging.info(
        f"After filtering → segments: {segments.shape}, Metadata: {seg_meta_df.shape}"
    )

    # Optional shrink for test runs
    if test_run:
        indices = np.random.choice(
            seg_meta_df.shape[0], size=int(seg_meta_df.shape[0] * 0.05), replace=False
        )
    else:
        indices = np.arange(seg_meta_df.shape[0])
    if noise_enabled and time_segments is not None:
        time_segments = time_segments[indices]

    # Normalizer
    normalizer = Normalizer(
        method=train_params["dataset_parameters"]["normalization_method"],
        mode=train_params["dataset_parameters"]["normalization_mode"],
    )
    logging.info(
        f"Normalizer initialized: method={normalizer.method}, mode={normalizer.mode}"
    )

    # Datasets & loaders
    train_dataset_cls = {
        "FaultInjectionDataset": FaultInjectionDataset,
        "AugmentedPoolDataset": AugmentedPoolDataset,
        "HybridAugFaultDataset": HybridAugFaultDataset,
    }[train_params["dataset_parameters"]["train_dataset_cls"]]
    train_dataset_kwargs = dict(train_params["dataset_parameters"]["train_dataset_kwargs"])
    batch_size = train_params["data_parameters"]["batch_size"]

    if noise_enabled and train_dataset_cls is not HybridAugFaultDataset:
        raise ValueError(
            "train_noise_policy is only supported with HybridAugFaultDataset."
        )

    if train_dataset_cls is HybridAugFaultDataset:
        if task == "binary":
            train_dataset_kwargs["K"] = 1 + train_dataset_kwargs["R"]
        else:
            train_dataset_kwargs["K"] = len(inj_cfg["fault_types_to_use"]) * (
                1 + train_dataset_kwargs["R"]
            )
        logging.info(
            f"From one normal window: {train_dataset_kwargs['R']} noise copies "
            f"and {train_dataset_kwargs['K']} augmentations are created"
        )

    logging.info(f"Preparing {task} dataset and model.")
    num_epochs = (
        train_params["training_parameters"]["num_epochs_binary"]
        if task == "binary"
        else train_params["training_parameters"]["num_epochs_multi"]
    )
    if noise_enabled:
        train_dataset_kwargs.update(
            {
                "noise_policy": noise_policy,
                "noise_total_epochs": int(num_epochs),
                "noise_fft_params": {
                    "f_sampling": inj_cfg["f_sampling"],
                    "cutoff_freq": inj_cfg["cutoff_freq"],
                    "db": inj_cfg["db"],
                },
                "noise_rng_seed": train_params.get("seed", 42) + 777,
            }
        )

    datasets = create_balanced_datasets(
        segments=segments[indices],
        seg_meta_df=seg_meta_df.iloc[indices],
        freqs=freqs,
        fault_freqs=fault_freqs_inject,
        mode=task,
        test_size=train_params["dataset_parameters"]["test_size"],
        val_size=train_params["dataset_parameters"]["val_size"],
        real_fault_train=train_params["dataset_parameters"].get("real_fault_train", 0),
        anomaly_injector=composite_injector,
        return_indices=True,
        save_indices=True,
        indices_dir=indices_dir,
        normalizer=normalizer,
        train_dataset_cls=train_dataset_cls,
        train_dataset_kwargs=train_dataset_kwargs,
        time_segments=time_segments,
    )
    datasets["normalizer"].save(exp_dir / f"normalizer_{task}.json")
    val_loader = DataLoader(datasets["val"], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False)

    # Classes
    num_classes = 2 if task == "binary" else len(datasets["label_encoder"].classes_)

    # Optional spectral prior attention
    if train_params["model_parameters"]["attention_module"]:
        mask = make_importance_mask(
            freqs, fault_freqs_physics, delta_hz=inj_cfg["peak_segment"]
        )
        mask_torch = torch.tensor(mask, dtype=torch.float32)[None, None, :].to(device)
        torch.save(mask_torch, exp_dir / "mask.pt")
        plt.figure()
        plt.plot(freqs, mask)
        plt.xlabel("Hz")
        plt.title("Importance Mask")
        plt.grid()
        plt.savefig(fig_dir / "importance_mask.png")
        experiment.log_figure(figure=plt.gcf(), figure_name="importance_mask.png")
        plt.close()
        logging.info("Importance mask computed and logged.")

        prior_kwargs = dict(
            mask=mask_torch,
            w_in_init=1.0,
            w_out_init=0.0,
            learnable_in=True,
            learnable_out=False,
        )
    else:
        prior_kwargs = None

    # Model
    model_type = train_params["model"]
    if model_type == "ResNet":
        model = ResNet(
            ResidualBlock,
            [2, 2, 2, 2],
            num_classes=num_classes,
            dropout_rate=train_params["model_parameters"]["dropout"],
            prior_kwargs=prior_kwargs,
        ).to(device)
    elif model_type == "CNN":
        model = CNN(
            num_classes=num_classes,
            dropout_rate=train_params["model_parameters"]["dropout"],
            prior_kwargs=prior_kwargs,
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {model_type}")

    logging.info(f"{task.capitalize()} {model_type} model instantiated.")

    # Train & evaluate
    train_and_eval(
        model=model,
        datasets=datasets,
        seg_meta_df=seg_meta_df,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        train_params=train_params,
        checkpoints_dir=checkpoints_dir,
        exp_dir=exp_dir,
        fig_dir=fig_dir,
        experiment=experiment,
        task=task,
    )

    experiment.end()
    logging.info("Training and evaluation complete. Experiment ended.")


if __name__ == "__main__":
    main()
