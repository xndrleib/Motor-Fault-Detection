# utils.py
import datetime
import random
import shutil

import numpy as np
import yaml


def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def create_experiment_folder(base_dir="../res", exp_name="engine_1"):
    # e.g., res/2024-05-21_15-03-25_engine_1/
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{timestamp}_{exp_name}"
    full_path = Path(base_dir) / folder_name
    full_path.mkdir(parents=True, exist_ok=False)
    print(f"[✓] Experiment results folder: {full_path.resolve()}")
    return full_path


import json
import os
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Literal

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler  # type: ignore

logger = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    """Compute SHA256 for a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_best(
    model: torch.nn.Module,
    epoch: int,
    metric: float,
    mode: Literal["binary", "multiclass"],
    out_dir: str | Path,
    *,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LRScheduler] = None,
    metric_name: str = "val_accuracy",
    filename: Optional[str] = None,
    keep_snapshot: bool = True,
    write_meta_json: bool = True,
    extra: Optional[dict] = None,
) -> Path:
    """Persist the best-performing model checkpoint (atomic & metadata-rich).

    Parameters
    ----------
    model : torch.nn.Module
        Trained model whose parameters will be saved via ``state_dict()``.
    epoch : int
        Epoch number at which the best metric was observed.
    metric : float
        Higher-is-better score (e.g., validation accuracy).
    mode : {"binary", "multiclass"}
        Short tag used in filenames (e.g., ``best_binary.pth``).
    out_dir : str or Path
        Output directory where the checkpoint will be written.
    optimizer : torch.optim.Optimizer, optional
        If provided, the optimizer state dict is stored under key ``"optimizer"``.
    scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        If provided, the scheduler state dict is stored under key ``"scheduler"``.
    metric_name : str, default="val_accuracy"
        Name of the metric used for model selection (stored in metadata).
    filename : str or None, optional
        Explicit filename for the canonical checkpoint. If ``None``, defaults to
        ``f"best_{mode}.pth"``.
    keep_snapshot : bool, default=True
        If ``True``, also write a versioned snapshot with epoch/metric in the name.
    write_meta_json : bool, default=True
        If ``True``, write a JSON sidecar (same stem) containing metadata and SHA256.
    extra : dict or None, optional
        Additional user-defined metadata to embed into the checkpoint package.

    Returns
    -------
    pathlib.Path
        Path to the canonical checkpoint file (e.g., ``.../best_binary.pth``).

    Notes
    -----
    - Save is **atomic**: writes to a temporary file in the same directory, then renames.
    - The returned checkpoint contains keys:
      ``{"epoch","metric","metric_name","model_class","state_dict",["optimizer"],["scheduler"],["num_classes"],["embedding_dim"],["extra"]}``.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Canonical filename (stable path used by the rest of the pipeline)
    canon_name = filename or f"best_{mode}.pth"
    canon_path = out_path / canon_name

    # Optional versioned snapshot filename (for history/audits)
    snap_name = f"{canon_path.stem}_e{epoch:04d}_{metric_name}-{metric:.4f}.pth"
    snap_path = out_path / snap_name

    # Build checkpoint package
    pkg: dict = {
        "epoch": int(epoch),
        "metric": float(metric),
        "metric_name": str(metric_name),
        "model_class": model.__class__.__name__,
        "state_dict": model.state_dict(),
        "saved_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    # Nice to have: model attributes if present
    if hasattr(model, "num_classes"):
        try:
            pkg["num_classes"] = int(getattr(model, "num_classes"))
        except Exception:
            pass
    if hasattr(model, "embedding_dim"):
        try:
            pkg["embedding_dim"] = int(getattr(model, "embedding_dim"))
        except Exception:
            pass

    if optimizer is not None:
        pkg["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        try:
            pkg["scheduler"] = scheduler.state_dict()
        except Exception:
            # Some schedulers don't implement state_dict; ignore gracefully
            logger.debug("Scheduler has no state_dict; skipping.")

    if extra:
        pkg["extra"] = dict(extra)

    # Atomic write: save to a temp file in the same directory, then replace
    tmp_path = out_path / f".{canon_path.name}.tmp.{os.getpid()}"
    torch.save(pkg, tmp_path)
    tmp_path.replace(canon_path)  # atomic on same filesystem

    # Optional snapshot
    if keep_snapshot:
        try:
            torch.save(pkg, snap_path)
        except Exception as e:
            logger.warning(f"Could not write snapshot {snap_path}: {e}")

    # JSON sidecar with checksum & metadata
    if write_meta_json:
        try:
            sha = _sha256(canon_path)
            meta = {
                "file": str(canon_path),
                "epoch": pkg["epoch"],
                "metric": pkg["metric"],
                "metric_name": pkg["metric_name"],
                "model_class": pkg["model_class"],
                "num_classes": pkg.get("num_classes"),
                "embedding_dim": pkg.get("embedding_dim"),
                "saved_at": pkg["saved_at"],
                "sha256": sha,
                "bytes": canon_path.stat().st_size,
            }
            if extra:
                meta["extra"] = extra
            meta_path = canon_path.with_suffix(".json")
            with meta_path.open("w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not write JSON sidecar for {canon_path}: {e}")

    logger.info(
        f"[✓] New best ({mode}) → {canon_path}  "
        f"({metric_name}={metric:.4f}, epoch={epoch})"
    )
    return canon_path


def dataloader_to_numpy(dataloader):
    """
    Converts a DataLoader to numpy arrays.

    Parameters:
    - dataloader: DataLoader containing data and labels.

    Returns:
    - all_data: Numpy array of data.
    - all_labels: Numpy array of labels.
    """
    all_data = []
    all_labels = []
    for inputs, labels in dataloader:
        batch_inputs = inputs.detach().cpu().numpy().reshape(inputs.size(0), -1)
        batch_labels = labels.detach().cpu().numpy()
        all_data.append(batch_inputs)
        all_labels.append(batch_labels)
    return np.concatenate(all_data), np.concatenate(all_labels)


def zip_folder(folder_path, zip_path):
    shutil.make_archive(zip_path, "zip", folder_path)


def move_random_csv_files(input_dir, output_dir, n_files):
    """
    Randomly selects a specified number of .csv files from the input directory
    and moves them to the output directory. If the output directory does not exist,
    it will be created automatically.

    Parameters:
    input_dir (str): The path to the directory containing the .csv files to be moved.
    output_dir (str): The path to the destination directory where the selected files will be moved.
    n_files (int): The number of .csv files to randomly select and move.

    Returns:
    list: A list of the moved file names.
    """

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List all files in the input directory that end with .csv
    csv_files = [file for file in os.listdir(input_dir) if file.endswith(".csv")]

    # Check if there are enough files to select from
    if n_files > len(csv_files):
        raise ValueError(
            f"Requested {n_files} files, but only found {len(csv_files)} .csv files in {input_dir}."
        )

    # Randomly select n_files from the list of .csv files
    selected_files = random.sample(csv_files, n_files)

    # Move each selected file to the output directory
    moved_files = []
    for file_name in selected_files:
        src_path = os.path.join(input_dir, file_name)
        dest_path = os.path.join(output_dir, file_name)
        shutil.move(src_path, dest_path)
        moved_files.append(file_name)

    return moved_files


def rename_folder(current_folder_path, new_folder_path):
    """
    Renames a folder by moving it from the current path to a new path with a different name.

    Parameters:
    current_folder_path (str): The path to the existing folder.
    new_folder_path (str): The path with the new folder name.
    """
    try:
        os.rename(current_folder_path, new_folder_path)
        return f"Folder renamed from '{current_folder_path}' to '{new_folder_path}'."
    except FileNotFoundError:
        return f"Error: The folder '{current_folder_path}' does not exist."
    except Exception as e:
        return f"An error occurred: {e}"


def generate_induction_motor_signal(
    signal_length=1000,
    sampling_rate=1000,
    fundamental_freq=50,
    noise_scale=0.2,
    add_harmonics=True,
    add_spikes=True,
    add_noise=True,
    add_modulation=True,
):
    """
    Generate a synthetic signal simulating induction motor current with selectable problems.

    Parameters:
        signal_length (int): Number of samples in the signal.
        sampling_rate (int): Sampling rate in Hz.
        fundamental_freq (float): Fundamental frequency in Hz.
        noise_scale (float): Standard deviation of Gaussian noise.
        add_harmonics (bool): If True, includes harmonic components.
        add_spikes (bool): If True, introduces transient spikes.
        add_noise (bool): If True, adds Gaussian noise.
        add_modulation (bool): If True, applies amplitude modulation.

    Returns:
        tuple: (clean_signal, noisy_signal)
            - clean_signal: The base signal without any disturbances.
            - noisy_signal: The signal with selected disturbances added.
    """
    t = np.linspace(0, signal_length / sampling_rate, signal_length)

    # Fundamental signal
    fundamental = np.sin(2 * np.pi * fundamental_freq * t)

    # Harmonics
    harmonics = (
        (
            0.1 * np.sin(2 * np.pi * 2 * fundamental_freq * t)
            + 0.1 * np.sin(2 * np.pi * 3 * fundamental_freq * t)
            + 0.1 * np.sin(2 * np.pi * 4 * fundamental_freq * t)
        )
        if add_harmonics
        else 0
    )

    # Amplitude modulation
    modulation = (1 + 0.1 * np.sin(2 * np.pi * 0.5 * t)) if add_modulation else 1

    # Combine base signal
    clean_signal = modulation * (fundamental + harmonics)

    # Initialize noisy signal
    noisy_signal = clean_signal.copy()

    # Gaussian noise
    if add_noise:
        noise = np.random.normal(scale=noise_scale, size=signal_length)
        noisy_signal += noise

    # Transient spikes
    if add_spikes:
        for _ in range(5):  # Introduce 5 random spikes
            spike_index = np.random.randint(0, signal_length)
            noisy_signal[spike_index : spike_index + 10] += np.random.normal(
                scale=3.0, size=10
            )

    return clean_signal, noisy_signal


def load_yaml(path):
    """Load a YAML file and return the configuration dictionary."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    print(f"Loaded configuration from {path}")
    return data


def get_run(run_path: Path):
    run_cfg = load_yaml(run_path / "training_config.yaml")
    print(run_cfg)


if __name__ == "__main__":
    path = Path("res/runs/2025-06-03_02-14-40_train_full-data-removeES-42-16")
    get_run(path)
