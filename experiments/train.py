# Code for training models
import argparse
import datetime
import logging
import os
import shutil
import time
from pathlib import Path

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


def setup_logger(log_dir, log_file="training.log"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    # Configure root logger
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Remove any existing handlers to avoid duplicates
    for h in list(root.handlers):
        root.removeHandler(h)

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    root.addHandler(ch)

    root.info(f"Logging to file: {log_path}")


def load_configurations():
    """Load general experiment configuration from YAML file."""
    with open("../cfg.yaml", "r") as f:
        config = yaml.safe_load(f)
    logging.info("Loaded main configuration from ../cfg.yaml.")
    return config


def start_experiment(config, online=True):
    """Start Comet ML experiment for logging metrics, code, and parameters."""
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


def prepare_directories(engine, config_name, res_dir="../res"):
    """
    Prepare and create output directories for experiment results, checkpoints, figures, and logs.
    Returns: base_dir, exp_dir, indices_dir, checkpoints_dir, fig_dir, log_dir
    """
    base_dir = Path(f"../dataset/{engine.replace('-', '_')}")
    res_dir = Path(res_dir)

    if ".yml" or ".yaml" in config_name:
        config_name = config_name.split(".")[0]
    exp_dir = create_experiment_folder(res_dir, config_name)
    indices_dir = exp_dir / "indices"
    checkpoints_dir = exp_dir / "checkpoints"
    fig_dir = exp_dir / "figs"
    log_dir = exp_dir / "logs"
    for d in [indices_dir, checkpoints_dir, fig_dir, log_dir]:
        d.mkdir(exist_ok=True, parents=True)
    logging.info(f"Experiment directories created at {exp_dir}")
    return base_dir, exp_dir, indices_dir, checkpoints_dir, fig_dir, log_dir


def save_label_encoder(enc, exp_dir):
    """Save label encoder for later inference/analysis."""
    enc_path = Path(exp_dir) / "label_encoder_multiclass.pkl"
    joblib.dump(enc, enc_path, compress=3)
    logging.info(f"Saved LabelEncoder to {enc_path.resolve()}")


def plot_and_save_confusion_matrix(cm, labels, fig_dir, fname_prefix, experiment):
    """Plot, save, and log confusion matrix."""
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
    ax.set_title(f"Confusion Matrix")
    fpath_png = fig_dir / f"{fname_prefix}.png"
    fpath_pdf = fig_dir / f"{fname_prefix}.pdf"
    plt.savefig(fpath_png, dpi=300)
    plt.savefig(fpath_pdf, dpi=300)
    experiment.log_figure(figure=fig, figure_name=f"{fname_prefix}.png")
    plt.close(fig)
    logging.info(f"Confusion matrix plotted and saved: {fpath_png}")


def save_configs(train_cfg_path, engine_cfg_path, exp_dir):
    """
    Copy training and engine configuration files into the experiment directory.

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
    """
    # Convert to Path objects
    train_cfg_path = Path(train_cfg_path)
    engine_cfg_path = Path(engine_cfg_path)
    exp_dir = Path(exp_dir)

    # Verify source files exist
    if not train_cfg_path.is_file():
        raise FileNotFoundError(f"Training config not found: {train_cfg_path}")
    if not engine_cfg_path.is_file():
        raise FileNotFoundError(f"Engine config not found:   {engine_cfg_path}")

    # Ensure experiment directory exists
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Define destination paths
    dest_train = exp_dir / "training_config.yaml"
    dest_engine = exp_dir / "engine_config.yaml"

    # Copy files
    shutil.copy2(train_cfg_path, dest_train)
    logging.info(f"Copied training config to {dest_train}")

    shutil.copy2(engine_cfg_path, dest_engine)
    logging.info(f"Copied engine   config to {dest_engine}")


def train_and_eval(
    model,
    datasets,
    seg_meta_df,
    val_loader,
    test_loader,
    device,
    train_params,
    checkpoints_dir,
    exp_dir,
    fig_dir,
    experiment,
    task,
):
    """
    Unified function for training, evaluating, and logging for both binary and multiclass tasks.
    """
    logging.info(f"Training model for {task} classification...")
    # Select number of epochs based on task
    num_epochs = (
        train_params["training_parameters"]["num_epochs_binary"]
        if task == "binary"
        else train_params["training_parameters"]["num_epochs_multi"]
    )
    # Time the training process
    train_start = time.perf_counter()
    # Train model
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
    train_end = time.perf_counter()
    train_time = train_end - train_start
    logging.info(f"Training complete. Time elapsed: {train_time:.2f} seconds.")
    experiment.log_metric(f"{task}_train_time_seconds", train_time)

    # Inference on test set
    logging.info("Starting inference on test set.")
    inference_start = time.perf_counter()
    true_labels, predictions = inference_model(model, test_loader, device=device)
    inference_end = time.perf_counter()
    inference_time = inference_end - inference_start
    logging.info(f"Inference complete. Time elapsed: {inference_time:.2f} seconds.")
    experiment.log_metric(f"{task}_inference_time_seconds", inference_time)

    # Prepare target names
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

    # Calculate and log metrics
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
        f"{task.capitalize()} Classification Report:\n{classification_report(true_labels, predictions, target_names=target_names)}"
    )
    logging.info(
        f"{task.capitalize()} Test Set Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}"
    )
    plot_and_save_confusion_matrix(
        cm, target_names, fig_dir, f"confusion_matrix_{task}", experiment
    )

    predictions = np.asarray(predictions, dtype=np.int64)
    test_idx = datasets["test_idx"]
    meta_test = seg_meta_df.loc[test_idx].copy()

    if task == "binary":
        # Save binary predictions and log model
        meta_test["binary_prediction"] = predictions
        meta_test["binary_prediction_state"] = np.where(
            predictions == 0, "normal", "anomalous"
        )
        out_path = exp_dir / "segments_metadata_test_binary_pred.csv"
        model_name = "bin_model"
        model_file = checkpoints_dir / "best_binary.pth"
    else:
        # Save multiclass label encoder and predictions, log model
        save_label_encoder(datasets["label_encoder"], exp_dir)
        meta_test["multiclass_prediction"] = predictions
        meta_test["multiclass_prediction_state"] = datasets[
            "label_encoder"
        ].inverse_transform(predictions)
        out_path = exp_dir / "segments_metadata_test_multiclass_pred.csv"
        model_name = "multi_model"
        model_file = checkpoints_dir / "best_multiclass.pth"

    meta_test.to_csv(out_path, index=True)
    logging.info(f"{task.capitalize()} test predictions saved to: {out_path}")
    experiment.log_model(name=model_name, file_or_folder=model_file)
    logging.info(f"{task.capitalize()} model logged to experiment: {model_file}")


def main():
    # Parse CLI only for an optional override --task and --cfg
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
    # Read engineLabel and task from training config
    engine = train_params.get("engineLabel", None)
    task_from_cfg = train_params.get("task", None)
    # CLI override for task (rare use; config takes precedence)
    task = task_from_cfg if task_from_cfg else args.task
    if not task:
        raise RuntimeError(
            "Task must be specified in training configuration (task: binary|multiclass) or via --task CLI argument."
        )

    # Prepare directories early, before logger setup
    base_dir, exp_dir, indices_dir, checkpoints_dir, fig_dir, log_dir = (
        prepare_directories(engine, args.cfg)
    )
    save_configs(train_cfg_path, base_dir / "engine.yml", exp_dir)
    setup_logger(str(log_dir))
    logging.info(
        f"Starting Training/Evaluation Script for engine '{engine}', task '{task}'"
    )

    # Load general experiment configuration and start experiment logging
    config = load_configurations()
    experiment, run_time = start_experiment(
        config, train_params.get("comet_online", True)
    )

    # Set all random seeds for reproducibility
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

    # Load engine-specific configuration and metadata
    engine_config = load_yaml(base_dir / "engine.yml")
    experiment.log_parameters(engine_config)
    metadata_df = pd.read_csv(base_dir / "metadata.csv").set_index("measurement_id")
    experiment.log_parameters(
        {"num_of_measurements": metadata_df["state"].value_counts().to_dict()}
    )
    experiment.log_parameters(train_params)

    # Prepare fault frequencies for data augmentation
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

    # Preprocessing: segmentation and spectral transformation
    inj_cfg = train_params["processing_parameters"]

    test_run = train_params.get("test_run", False)
    if test_run:
        # Use 5% of the data for testing
        indices = np.random.choice(
            metadata_df.shape[0], size=int(metadata_df.shape[0] * 0.05), replace=False
        )
    else:
        # Use all the data
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

    # Instantiate data injectors for data augmentation
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

    # === Apply filtering ===
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
        f"After filtering → segments: {segments.shape}, "
        f"Metadata: {seg_meta_df.shape}"
    )

    # === Shrink segments and metadata for testing if needed ===
    if test_run:
        # Use 5% of the data for testing
        indices = np.random.choice(
            seg_meta_df.shape[0], size=int(seg_meta_df.shape[0] * 0.05), replace=False
        )
    else:
        # Use all the data
        indices = np.arange(seg_meta_df.shape[0])

    # Instantiate normalizer according to config
    normalizer = Normalizer(
        method=train_params["dataset_parameters"]["normalization_method"],
        mode=train_params["dataset_parameters"]["normalization_mode"],
    )
    logging.info(
        f"Normalizer initialized: method={normalizer.method}, mode={normalizer.mode}"
    )

    # Select dataset class as specified in config
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

    if task == "binary":
        num_classes = 2
    else:
        num_classes = len(datasets["label_encoder"].classes_)

    if train_params["model_parameters"]["attention_module"]:
        # Compute and log spectral importance mask (prior for attention)
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

    model_type = train_params["model"]

    if model_type == "ResNet":
        model = ResNet(
            ResidualBlock,
            [2, 2, 2, 2],
            num_classes=num_classes,
            dropout_rate=train_params["model_parameters"]["dropout"],
            prior_kwargs=None,
        ).to(device)
    elif model_type == "CNN":
        model = CNN(
            num_classes, dropout_rate=train_params["model_parameters"]["dropout"]
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {model_type}")

    logging.info(f"{task.capitalize()} {model_type} model instantiated.")

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

    # Finalize and close experiment
    experiment.end()
    logging.info("Training and evaluation complete. Experiment ended.")


if __name__ == "__main__":
    main()
