# Code for training models
"""End-to-end training & evaluation script with artifact saving.

This module orchestrates data preparation, model training/validation, test-time
inference, and artifact logging. It augments the test predictions CSV with
per-class softmax scores and persists arrays (logits, probabilities, embeddings,
labels, predictions, and test indices) to compressed ``.npz`` files.

Outputs
-------
In the experiment directory (``exp_dir``), this script saves:
- ``segments_metadata_test_[binary|multiclass]_pred.csv`` : predictions + scores
- ``test_arrays_[binary|multiclass].npz`` : probs/logits/embeddings/labels/preds/index
- ``test_embeddings_[binary|multiclass].npy`` : embeddings only
- ``test_logits_[binary|multiclass].npy`` : logits only
- ``test_probs_[binary|multiclass].npy`` : softmax probabilities only
- Confusion matrix figures (PNG/PDF) and optional importance mask figure

Notes
-----
- ``inference_model`` is invoked with ``return_extra=True`` to collect
  logits/probabilities/embeddings.
- Embeddings come from ``model.forward_features`` if implemented.
"""
import argparse
import datetime
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Tuple

import comet_ml  # import comet_ml before the following modules: torch.
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
from src.utils import set_all_seeds, create_experiment_folder, load_yaml


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
    """Load general experiment configuration from ``../cfg.yaml``.

    Returns
    -------
    dict
        Parsed YAML configuration.
    """
    with open("../cfg.yaml", "r") as f:
        config = yaml.safe_load(f)
    logging.info("Loaded main configuration from ../cfg.yaml.")
    return config


def start_experiment(
    config: dict, online: bool = True
) -> Tuple[comet_ml.CometExperiment, str]:
    """Start a Comet ML experiment and log source code.

    Parameters
    ----------
    config : dict
        Dictionary with keys ``API_KEY``, ``PROJECT_NAME``, and ``WORKSPACE``.
    online : bool, default=True
        If ``False``, runs offline (local logging only).

    Returns
    -------
    (Experiment, str)
        The Comet experiment handle and the timestamped run name.
    """
    experiment = comet_ml.start(
        api_key=config["API_KEY"],
        project_name=config["PROJECT_NAME"],
        workspace=config["WORKSPACE"],
        online=online,
    )
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    experiment.set_name(f"Script Run: {now}")
    experiment.log_code()
    logging.info(f"Started Comet ML experiment at {now}")
    return experiment, now


def prepare_directories(engine: str, config_name: str, res_dir: str = "../res"):
    """Create output directories for experiment artifacts.

    Parameters
    ----------
    engine : str
        Engine label from configuration.
    config_name : str
        Training configuration filename (used to name the experiment folder).
    res_dir : str, default="../res"
        Root results directory.

    Returns
    -------
    tuple
        ``(base_dir, exp_dir, indices_dir, checkpoints_dir, fig_dir, log_dir)``
        as ``pathlib.Path`` objects.
    """
    base_dir = Path(f"../dataset/{engine.replace('-', '_')}")
    res_dir = Path(res_dir)

    # Properly strip the YAML suffix if present
    if config_name and config_name.endswith((".yml", ".yaml")):
        config_name = config_name.rsplit(".", 1)[0]

    exp_dir = create_experiment_folder(res_dir, config_name)
    indices_dir = exp_dir / "indices"
    checkpoints_dir = exp_dir / "checkpoints"
    fig_dir = exp_dir / "figs"
    log_dir = exp_dir / "logs"
    for d in [indices_dir, checkpoints_dir, fig_dir, log_dir]:
        d.mkdir(exist_ok=True, parents=True)
    logging.info(f"Experiment directories created at {exp_dir}")
    return base_dir, exp_dir, indices_dir, checkpoints_dir, fig_dir, log_dir


def save_label_encoder(enc, exp_dir: str | Path) -> None:
    """Persist label encoder used for multiclass tasks.

    Parameters
    ----------
    enc : sklearn.preprocessing.LabelEncoder
        Fitted encoder mapping string labels to integers.
    exp_dir : str or Path
        Experiment directory where the pickle is written.

    Returns
    -------
    None
    """
    enc_path = Path(exp_dir) / "label_encoder_multiclass.pkl"
    joblib.dump(enc, enc_path, compress=3)
    logging.info(f"Saved LabelEncoder to {enc_path.resolve()}")


def plot_and_save_confusion_matrix(
    cm, labels, fig_dir: Path, fname_prefix: str, experiment
) -> None:
    """Plot, save, and log a confusion matrix.

    Parameters
    ----------
    cm : np.ndarray of shape (C, C)
        Confusion matrix counts.
    labels : list[str]
        Class display names in order of indices.
    fig_dir : Path
        Output directory for figures.
    fname_prefix : str
        File prefix (PNG/PDF will be produced).
    experiment : comet_ml.CometExperiment
        Comet experiment for figure logging.

    Returns
    -------
    None
    """
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
    """Copy training and engine configuration files into the experiment folder.

    Parameters
    ----------
    train_cfg_path : str or Path
        Path to the training configuration YAML file.
    engine_cfg_path : str or Path
        Path to the engine configuration YAML file.
    exp_dir : str or Path
        Path to the experiment directory where configs will be saved.

    Raises
    ------
    FileNotFoundError
        If either source config file does not exist.

    Returns
    -------
    None
    """
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
    """Train the model, evaluate on test set, and save artifacts.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train (logits from ``forward``; embeddings from ``forward_features``).
    datasets : dict
        Output of ``create_balanced_datasets`` including ``train``, ``val``,
        ``test``, and metadata such as ``label_encoder`` (multiclass) and
        ``test_idx``.
    seg_meta_df : pandas.DataFrame
        Segment-level metadata indexed by ``measurement_id``.
    val_loader : DataLoader
        Validation loader.
    test_loader : DataLoader
        Test loader.
    device : torch.device
        Target device.
    train_params : dict
        Training configuration dictionary.
    checkpoints_dir : Path
        Directory for model checkpoints.
    exp_dir : Path
        Experiment directory for artifacts.
    fig_dir : Path
        Directory for figures.
    experiment : comet_ml.CometExperiment
        Comet experiment handle.
    task : {"binary", "multiclass"}
        Task type controlling epochs and label handling.

    Returns
    -------
    None

    Side Effects
    ------------
    - Saves predictions CSV with per-class probabilities.
    - Saves compressed NPZ with ``probs/logits/embeddings/labels/preds/test_index``.
    - Logs figures and models to Comet.
    """
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


def main() -> None:
    """CLI entry-point: load configs, prepare data, train/eval, and log artifacts.

    Reads the training config (``--cfg``) to determine dataset/model/task
    settings; prepares folders; initializes Comet; builds datasets with optional
    spectral-prior attention mask; trains the configured model; evaluates on
    the test set; and saves/logs all artifacts.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If neither the config nor ``--task`` provides a valid task.
    """
    parser = argparse.ArgumentParser(
        description="Train and evaluate model. Reads 'engineLabel' and 'task' from training config."
    )
    parser.add_argument(
        "--task",
        choices=["binary", "multiclass"],
        help="[optional] Override: Task to train: binary or multiclass",
    )
    parser.add_argument(
        "--cfg", type=str, default=None, help="Training config YAML filename"
    )
    args = parser.parse_args()

    # Determine config file
    train_cfg_path = f"../training_configs/{args.cfg}"

    # Load training configuration
    train_params = load_yaml(train_cfg_path)
    engine = train_params.get("engineLabel", None)
    task_from_cfg = train_params.get("task", None)
    task = task_from_cfg if task_from_cfg else args.task
    if not task:
        raise RuntimeError(
            "Task must be specified in training configuration (task: binary|multiclass) or via --task CLI argument."
        )

    # Prepare directories & logger
    base_dir, exp_dir, indices_dir, checkpoints_dir, fig_dir, log_dir = (
        prepare_directories(engine, args.cfg)
    )
    save_configs(train_cfg_path, base_dir / "engine.yml", exp_dir)
    setup_logger(str(log_dir))
    logging.info(
        f"Starting Training/Evaluation Script for engine '{engine}', task '{task}'"
    )

    # Start Comet and set seeds/device
    config = load_configurations()
    experiment, run_time = start_experiment(
        config, train_params.get("comet_online", True)
    )

    set_all_seeds(train_params.get("seed", 42))
    device_str = (
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

    # Load engine config & metadata
    engine_config = load_yaml(base_dir / "engine.yml")
    experiment.log_parameters(engine_config)
    metadata_df = pd.read_csv(base_dir / "metadata.csv").set_index("measurement_id")
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
    fault_freqs = {ft: ANOMALY_FREQS[ft](**MCSA_cfg[ft]) for ft in fault_types_to_use}
    experiment.log_parameters(fault_freqs)
    logging.info(f"Fault frequencies calculated: {fault_freqs}")

    # Preprocessing (segmentation + spectrum)
    inj_cfg = train_params["processing_parameters"]
    test_run = train_params.get("test_run", False)
    if test_run:
        indices = np.random.choice(
            metadata_df.shape[0], size=int(metadata_df.shape[0] * 0.05), replace=False
        )
    else:
        indices = np.arange(metadata_df.shape[0])

    segments, seg_meta_df, freqs = preprocessing(
        metadata_df=metadata_df.iloc[indices],
        out_dir=base_dir,
        segment_length=inj_cfg["segment_length"],
        step=inj_cfg["shift"],
        f_sampling=inj_cfg["f_sampling"],
        cutoff_freq=inj_cfg["cutoff_freq"],
        apply_window=False,
        db=inj_cfg["db"],
    )

    # Data injectors
    freq_resolution = freqs[1] - freqs[0]
    peak_segment_bins = int(np.ceil(inj_cfg["peak_segment"] / freq_resolution))
    logging.info(
        f"Converted peak segment from Hz to bins: {inj_cfg['peak_segment']} Hz -> {peak_segment_bins}"
    )

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
    train_dataset_kwargs = train_params["dataset_parameters"]["train_dataset_kwargs"]
    batch_size = train_params["data_parameters"]["batch_size"]

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
    datasets = create_balanced_datasets(
        segments=segments[indices],
        seg_meta_df=seg_meta_df.iloc[indices],
        freqs=freqs,
        fault_freqs=fault_freqs,
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
    )
    datasets["normalizer"].save(exp_dir / f"normalizer_{task}.json")
    val_loader = DataLoader(datasets["val"], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False)

    # Classes
    num_classes = 2 if task == "binary" else len(datasets["label_encoder"].classes_)

    # Optional spectral prior attention
    if train_params["model_parameters"]["attention_module"]:
        mask = make_importance_mask(
            freqs, fault_freqs, delta_hz=inj_cfg["peak_segment"]
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
