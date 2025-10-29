#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embed FID windows with a run's penultimate-layer features.

This script loads a trained model and its artifacts (checkpoint, normalizer,
label encoder, and training_config.yaml), replaces the final classifier with
an identity layer, and produces embeddings for all FID input windows found
under --fid-dir (files matching: real_*.npy, synth_*.npy). Outputs are saved
under --out-root/<run_id>/ as *_embs.npy.

Examples
--------
Command-line usage:

    python experiments/build_fid_embeddings.py \
        --run-dir res/runs/2025-06-03_02-14-40_train_full-data-removeES-42-16 \
        --fid-dir res/fid_inputs \
        --out-root res/fid_embs \
        --batch-size 2048 \
        --device auto

Outputs
-------
Embeddings are written as:

    res/fid_embs/<run_id>/synth_100_RBD_embs.npy
    res/fid_embs/<run_id>/real_40_RBD_embs.npy

Notes
-----
- We avoid forward hooks and instead **swap** the final classifier module
  (`fc` / `classifier`) with `nn.Identity()` inside a context manager,
  so the forward returns penultimate features. This is safer than global hooks.
- If the dataset uses global normalization, a `normalizer_*.json` must
  exist in the run directory (loaded via `Normalizer.load`).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import yaml

# Repo-local imports
from src.models import CNN, ResNet, ResidualBlock, MLP
from src.normalization import Normalizer


# =============================================================================
# Run discovery / artifact loading
# =============================================================================

def _safe_load_yaml(p: Path) -> Dict[str, Any]:
    """
    Load a YAML file safely.

    Parameters
    ----------
    p : Path
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed YAML contents.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    yaml.YAMLError
        If the YAML is malformed.
    """
    with open(p, "r") as f:
        return yaml.safe_load(f)


def _discover_checkpoint(run_dir: Path, task: str) -> Path:
    """
    Find a checkpoint file inside <run_dir>/checkpoints.

    Parameters
    ----------
    run_dir : Path
        Training run directory.
    task : {"binary", "multiclass"}
        Task name; used to prefer task-specific checkpoint names.

    Returns
    -------
    Path
        Path to a `.pth`/`.pt` checkpoint.

    Raises
    ------
    FileNotFoundError
        If no checkpoints directory or file is found.
    """
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Missing checkpoints dir: {ckpt_dir}")

    # Prefer explicit task-specific names; fall back to the first .pth/.pt file.
    preferred = [
        f"best_{'multiclass' if task == 'multiclass' else 'binary'}.pth",
        f"best_{'multiclass' if task == 'multiclass' else 'binary'}.pt",
    ]
    for name in preferred:
        p = ckpt_dir / name
        if p.exists():
            return p

    # Fallback: first *.pth / *.pt
    candidates: List[Path] = []
    for ext in ("*.pth", "*.pt"):
        candidates.extend(sorted(ckpt_dir.glob(ext)))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    return candidates[0]


def _discover_normalizer(run_dir: Path, task: str) -> Optional[Path]:
    """
    Discover a normalizer JSON file inside a run directory.

    Parameters
    ----------
    run_dir : Path
        Training run directory.
    task : {"binary", "multiclass"}
        Task name; used to prefer task-specific filenames.

    Returns
    -------
    Path or None
        Found normalizer path or None if not present.
    """
    # Prefer task-specific filenames; fall back to a generic name.
    names = [
        f"normalizer_{'multiclass' if task == 'multiclass' else 'binary'}.json",
        "normalizer.json",
    ]
    for n in names:
        p = run_dir / n
        if p.exists():
            return p
    return None


def _discover_label_encoder(run_dir: Path, task: str) -> Optional[Path]:
    """
    Discover a label encoder file for inferring `num_classes`.

    Parameters
    ----------
    run_dir : Path
        Training run directory.
    task : {"binary", "multiclass"}
        Task name; used to prefer task-specific filenames.

    Returns
    -------
    Path or None
        Found label encoder path or None if not present.
    """
    names = [
        f"label_encoder_{'multiclass' if task == 'multiclass' else 'binary'}.pkl",
        "label_encoder.pkl",
    ]
    for n in names:
        p = run_dir / n
        if p.exists():
            return p
    return None


def _infer_num_classes(task: str, le_path: Optional[Path]) -> int:
    """
    Infer number of classes from the label encoder.

    Parameters
    ----------
    task : {"binary", "multiclass"}
        Task type. If not "multiclass", returns 2.
    le_path : Path or None
        Path to a joblib-compressed label encoder. Must expose `.classes_`.

    Returns
    -------
    int
        Number of classes.

    Raises
    ------
    FileNotFoundError
        If `task == "multiclass"` and no label encoder is found.
    AttributeError
        If the loaded encoder does not have a `classes_` attribute.
    """
    if task != "multiclass":
        return 2
    if not le_path or not le_path.exists():
        raise FileNotFoundError(
            "Label encoder not found. Expected a joblib-compressed file like "
            "'label_encoder_multiclass.pkl' or '.joblib' in the run directory."
        )
    le = joblib.load(le_path)
    if not hasattr(le, "classes_"):
        raise AttributeError(f"Loaded label encoder at {le_path} has no 'classes_' attribute.")
    return int(len(le.classes_))


def _arch_from_config(cfg_model: str) -> str:
    """
    Normalize model name from the training config to an internal key.

    Parameters
    ----------
    cfg_model : str
        Value from training_config.yaml under key `model` (e.g., "ResNet").

    Returns
    -------
    str
        One of {"resnet18", "cnn", "mlp"}.

    Raises
    ------
    ValueError
        If the model is unsupported.
    """
    m = (cfg_model or "").strip().lower()
    if m in {"resnet", "resnet18"}:
        return "resnet18"
    if m == "cnn":
        return "cnn"
    if m == "mlp":
        return "mlp"
    raise ValueError(f"Unsupported model in training_config.yaml: {cfg_model!r}")


# =============================================================================
# Model construction & checkpoint loading
# =============================================================================

def build_model(
    arch: str,
    input_dim: int,
    num_classes: int,
    use_prior: bool,
    dropout: Optional[float],
) -> nn.Module:
    """
    Instantiate a model matching the training-time architecture.

    Parameters
    ----------
    arch : {"resnet18", "cnn", "mlp"}
        Model architecture.
    input_dim : int
        Input length (L) of the 1D signal segment.
    num_classes : int
        Number of output classes (defines the discarded classifier head).
    use_prior : bool
        If True, pass a neutral "prior" structure to the model (if supported).
    dropout : float or None
        Optional dropout rate to set on supported models.

    Returns
    -------
    torch.nn.Module
        Constructed model instance (classifier layer intact).

    Notes
    -----
    We set a **neutral** prior when requested so that weight loading remains
    robust (i.e., does not skew features on load).
    """
    prior_kwargs = None
    if use_prior:
        # Neutral prior to keep weight loading robust (multiplicative identity).
        mask = torch.zeros(1, 1, input_dim)
        prior_kwargs = {
            "mask": mask,
            "w_in_init": 1.0,
            "w_out_init": 1.0,
            "learnable_in": True,
            "learnable_out": True,
        }

    if arch == "cnn":
        m = CNN(num_classes=num_classes, prior_kwargs=prior_kwargs)
        if dropout is not None:
            m.dropout_rate = float(dropout)
        return m

    if arch == "resnet18":
        m = ResNet(
            block=ResidualBlock,
            layers=[2, 2, 2, 2],
            num_classes=num_classes,
            prior_kwargs=prior_kwargs,
        )
        if dropout is not None:
            m.dropout_rate = float(dropout)
        return m

    if arch == "mlp":
        return MLP(
            input_dim=input_dim,
            hidden_dims=[512, 256, 128],
            num_classes=num_classes,
            dropout_rate=float(dropout or 0.5),
        )

    raise ValueError(arch)


def load_checkpoint_into(model: nn.Module, ckpt_path: Path, device: torch.device) -> nn.Module:
    """
    Load checkpoint weights into a model, handling several common formats.

    Parameters
    ----------
    model : torch.nn.Module
        Preconstructed model with matching architecture.
    ckpt_path : Path
        Path to a checkpoint file (.pth/.pt).
    device : torch.device
        Device map for loading tensors.

    Returns
    -------
    torch.nn.Module
        Model with weights loaded (or the loaded module if a full module was saved).

    Raises
    ------
    RuntimeError
        If the checkpoint format is unrecognized.
    """
    obj = torch.load(str(ckpt_path), map_location=device)

    # Case 1: entire module was saved
    if isinstance(obj, nn.Module):
        try:
            model.load_state_dict(obj.state_dict(), strict=False)
            return model
        except Exception:
            # If state dicts are incompatible, just return the loaded module.
            return obj

    # Case 2: a dict container (commonly with "state_dict" key)
    if isinstance(obj, dict):
        for k in ("state_dict", "model_state_dict"):
            if k in obj and isinstance(obj[k], dict):
                model.load_state_dict(obj[k], strict=False)
                return model
        # Maybe it's already a state_dict
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            model.load_state_dict(obj, strict=False)
            return model
        # Last resort: any dict value that *looks* like a state_dict
        for v in obj.values():
            if isinstance(v, dict) and all(isinstance(x, torch.Tensor) for x in v.values()):
                model.load_state_dict(v, strict=False)
                return model

    # Fallthrough: direct state_dict-like dict
    if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        model.load_state_dict(obj, strict=False)
        return model

    raise RuntimeError(f"Unrecognized checkpoint format at {ckpt_path}")


# =============================================================================
# Normalizer
# =============================================================================

def make_normalizer(method: Optional[str], mode: Optional[str], path: Optional[Path]) -> Optional[Normalizer]:
    """
    Construct and optionally load a `Normalizer`.

    Parameters
    ----------
    method : str or None
        Normalization method name (e.g., "min-max"). If None, returns None.
    mode : str or None
        Normalization mode (e.g., "global"). If None, returns None.
    path : Path or None
        Path to a JSON file with global normalization statistics.

    Returns
    -------
    Normalizer or None
        Configured normalizer, or None if not requested.

    Raises
    ------
    FileNotFoundError
        If `mode == "global"` but the `path` does not exist.
    """
    if method is None or mode is None:
        return None
    norm = Normalizer(method=method, mode=mode)
    if mode == "global":
        if not path or not path.exists():
            raise FileNotFoundError(
                f"Global normalization selected but stats file not found at: {path}"
            )
        norm.load(path)
    return norm


# =============================================================================
# Safe penultimate extractor (no forward hooks)
# =============================================================================

class _SwapClassifier:
    """
    Temporarily replace the final classifier with identity to read features.

    This context manager searches for a terminal `nn.Linear` classifier
    (attributes named `fc` or `classifier` are tried first) and replaces it
    with `nn.Identity()` so that `model(x)` returns penultimate features.
    The original module is restored on exit.

    Notes
    -----
    - This avoids global forward hooks, which can introduce confusing global
      state and are discouraged outside of debugging.
    - If no top-level `fc`/`classifier` is found, the **last** `nn.Linear`
      module discovered via `named_modules()` is swapped.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.attr_name: Optional[str] = None
        self.orig: Optional[nn.Module] = None

    def __enter__(self) -> "_SwapClassifier":
        # Preferred attributes commonly used by CNN/ResNet wrappers.
        for name in ("fc", "classifier"):
            layer = getattr(self.model, name, None)
            if isinstance(layer, (nn.Linear, nn.Identity)):
                self.attr_name = name
                self.orig = layer
                setattr(self.model, name, nn.Identity())
                return self

        # Fallback: swap the last encountered nn.Linear anywhere in the module tree.
        last_linear_name: Optional[str] = None
        for n, m in self.model.named_modules():
            if isinstance(m, nn.Linear):
                last_linear_name = n
        if not last_linear_name:
            raise RuntimeError("Could not find a classifier Linear layer to swap.")

        parent, leaf = self._split_parent(last_linear_name)
        self.attr_name = last_linear_name
        self.orig = getattr(parent, leaf)
        setattr(parent, leaf, nn.Identity())
        return self

    def _split_parent(self, dotted: str) -> Tuple[nn.Module, str]:
        """Resolve 'a.b.c' to (obj.a.b, 'c')."""
        parts = dotted.split(".")
        obj: nn.Module = self.model
        for p in parts[:-1]:
            obj = getattr(obj, p)
        return obj, parts[-1]

    def __exit__(self, exc_type, exc, tb) -> bool:
        # Restore the original classifier layer.
        if self.attr_name is None:
            return False
        if "." in self.attr_name:
            parent, leaf = self._split_parent(self.attr_name)
            setattr(parent, leaf, self.orig)
        else:
            setattr(self.model, self.attr_name, self.orig)
        return False


# =============================================================================
# FID windows I/O
# =============================================================================

def discover_windows(fid_dir: Path) -> List[Path]:
    """
    Discover all input window files to embed.

    Parameters
    ----------
    fid_dir : Path
        Directory containing FID windows with names like `real_*.npy`
        and `synth_*.npy`.

    Returns
    -------
    list of Path
        Sorted list of matching window files.
    """
    hits: List[Path] = []
    for pat in ("real_*.npy", "synth_*.npy"):
        hits.extend(sorted(fid_dir.glob(pat)))
    return hits


# =============================================================================
# Embedding
# =============================================================================

def compute_embeddings(
    model: nn.Module,
    windows: np.ndarray,
    device: torch.device,
    batch_size: int,
    normalizer: Optional[Normalizer],
) -> np.ndarray:
    """
    Compute penultimate-layer embeddings for a batch of windows.

    Parameters
    ----------
    model : torch.nn.Module
        Model with a classifier head that will be swapped to identity.
    windows : ndarray of shape (N, L)
        Real or synthetic signal segments, one per row.
    device : torch.device
        Target device for inference.
    batch_size : int
        Number of windows per forward pass.
    normalizer : Normalizer or None
        Optional normalizer. If provided, `transform` is applied per batch.

    Returns
    -------
    ndarray of shape (N, D)
        Embedding matrix (float32). If `windows` is empty or invalid,
        returns an empty array with shape (0, 0).

    Notes
    -----
    - Uses `torch.inference_mode()` to disable autograd and enable
      additional runtime optimizations.
    - The classifier is swapped out only for the duration of the forward passes.
    """
    # Early-out for empty or malformed inputs.
    if windows is None or windows.size == 0 or windows.ndim != 2 or windows.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float32)

    model.eval()
    N, _ = windows.shape
    feats: List[np.ndarray] = []

    # Temporarily replace classifier with identity to return features.
    with _SwapClassifier(model):
        with torch.inference_mode():
            for s in range(0, N, batch_size):
                e = min(s + batch_size, N)
                batch = windows[s:e]

                # Apply normalization if configured (e.g., min-max/global).
                if normalizer is not None:
                    batch = normalizer.transform(batch)

                # NN expects (N, C, L); here C=1 for a single 1D channel.
                x = torch.as_tensor(batch, dtype=torch.float32, device=device).unsqueeze(1)

                # Forward now returns penultimate features thanks to the swap.
                f = model(x)

                # Move to CPU numpy (float32) and stash.
                feats.append(f.detach().cpu().numpy().astype(np.float32))

    if not feats:
        # Should not happen if N > 0, but stay defensive.
        return np.empty((0, 0), dtype=np.float32)

    emb = np.concatenate(feats, axis=0)
    if emb.shape[0] != N:
        raise RuntimeError(f"Mismatch: got {emb.shape[0]} embeddings for {N} windows.")
    return emb


# =============================================================================
# CLI / main
# =============================================================================

def main() -> None:
    """
    Entrypoint: embed all FID windows using the specified run.

    Command-line arguments
    ----------------------
    --run-dir : Path (required)
        Specific training run directory under `res/runs/...`.
    --fid-dir : Path (default: res/fid_inputs)
        Directory containing FID windows (`real_*.npy`, `synth_*.npy`).
    --out-root : Path (default: res/fid_embs)
        Root folder under which `<run_id>` subfolder will be created.
    --batch-size : int or None (default: None)
        If provided, overrides the batch size from training config.
        Otherwise uses `data_parameters.batch_size` or 1024.
    --device : {"auto","cpu","cuda"} (default: "auto")
        Device selection policy. "auto" uses CUDA if available, else CPU.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="A specific training run directory under res/runs/…",
    )
    ap.add_argument(
        "--fid-dir",
        type=Path,
        default=Path("res/fid_inputs"),
        help="Where real_*.npy / synth_*.npy live.",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("res/fid_embs"),
        help="Root folder to place <run_id> subfolder with embeddings.",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size. If omitted, uses training config's batch_size or 1024.",
    )
    ap.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to run inference on.",
    )
    args = ap.parse_args()

    # ---- Load training configuration & artifacts ----------------------
    run_dir: Path = args.run_dir
    run_id = run_dir.name
    cfg_path = run_dir / "training_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing training_config.yaml at {cfg_path}")

    cfg = _safe_load_yaml(cfg_path)
    task = str(cfg.get("task", "binary")).lower()
    arch = _arch_from_config(cfg.get("model", "ResNet"))
    use_prior = bool(cfg.get("model_parameters", {}).get("attention_module", False))
    dropout = cfg.get("model_parameters", {}).get("dropout", None)

    # Batch size selection: CLI > config > fallback
    bs_cfg = int(cfg.get("data_parameters", {}).get("batch_size", 1024))
    batch_size = int(args.batch_size or bs_cfg or 1024)

    # Normalizer (optional; required if mode == "global")
    method = cfg.get("dataset_parameters", {}).get("normalization_method", "min-max")
    mode = cfg.get("dataset_parameters", {}).get("normalization_mode", "global")
    norm_path = _discover_normalizer(run_dir, task)
    normalizer = make_normalizer(method, mode, norm_path)

    # Model weights
    ckpt_path = _discover_checkpoint(run_dir, task)

    # ---- Discover input windows --------------------------------------
    fid_dir = args.fid_dir
    files = discover_windows(fid_dir)
    if not files:
        raise FileNotFoundError(
            f"No FID windows found in {fid_dir} (looked for real_*.npy / synth_*.npy)"
        )

    # Probe input length from the first file to configure the model.
    probe = np.load(files[0], allow_pickle=False)
    if probe.ndim != 2:
        raise ValueError(f"Expected 2D windows in {files[0].name}, got {probe.shape}")
    _, L = probe.shape

    # Number of classes (for constructing the model head we later swap)
    le_path = _discover_label_encoder(run_dir, task)
    num_classes = _infer_num_classes(task, le_path)

    # Device resolution
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # ---- Build model, load weights, move to device --------------------
    model = build_model(
        arch=arch,
        input_dim=L,
        num_classes=num_classes,
        use_prior=use_prior,
        dropout=dropout,
    )
    model.to(device)
    model = load_checkpoint_into(model, ckpt_path, device)
    model.to(device).eval()

    # ---- Output directory per run ------------------------------------
    out_dir = args.out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Embed each window file --------------------------------------
    for f in files:
        X = np.load(f, allow_pickle=False).astype(np.float32, copy=False)

        # Defensive shape checks with informative messages (skip invalid files).
        if X is None or X.size == 0:
            print(f"[skip] {f.name} is empty (0 windows).")
            continue
        if X.ndim == 1 and X.size > 0:
            print(f"[skip] {f.name} has unexpected shape {X.shape}; expected (N, {L}).")
            continue
        if X.ndim != 2 or X.shape[1] != L:
            print(f"[skip] {f.name} has shape {X.shape}; expected (*, {L}).")
            continue
        if X.shape[0] == 0:
            print(f"[skip] {f.name} is empty (0 windows).")
            continue

        embs = compute_embeddings(
            model=model,
            windows=X,
            device=device,
            batch_size=batch_size,
            normalizer=normalizer,
        )

        out_path = out_dir / f"{f.stem}_embs.npy"
        # Save as float32 (compact and sufficient for downstream FID).
        np.save(out_path, embs.astype(np.float32, copy=False), allow_pickle=False)
        print(f"[✓] {f.name:>30s} → {embs.shape} saved at {out_path}")

    print(f"\nDone. Embeddings stored in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
