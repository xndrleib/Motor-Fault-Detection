#!/usr/bin/env python3
"""Aggregate SGDA peak-ablation runs across seeds and compare MCSA vs random.

This script scans run folders produced by ``experiments/train.py`` under a
peak-ablation experiment directory (e.g., ``res/runs/sgda_peak_ablation``),
computes evaluation metrics for each run, aggregates across seeds for each
peak mode, and saves comparison plots with mean ± (std_mult * std).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm.auto import tqdm

BIN_CSV = "segments_metadata_test_binary_pred.csv"
MULTI_CSV = "segments_metadata_test_multiclass_pred.csv"
ENCODER_PKL = "label_encoder_multiclass.pkl"
LOSS_CSV = "loss_history.csv"

SHORT_NAMES = {
    "bearing defect": "BD",
    "inter-turn short circuits": "ITSC",
    "rotor bar defect": "RBD",
    "normal": "Normal",
}

def setup_logger(log_dir: Path, log_file: str = "peak_ablation_results.log") -> logging.Logger:
    """Configure a console + file logger.

    Parameters
    ----------
    log_dir : Path
        Directory to write the log file.
    log_file : str, optional
        Log filename.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_file

    logger = logging.getLogger("peak_ablation_results")
    logger.setLevel(logging.INFO)

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info("Logging to file: %s", log_path)
    return logger


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _is_run_dir(path: Path) -> bool:
    return path.is_dir() and (path / "training_config.yaml").exists()


def _discover_run_dirs(runs_dir: Path) -> List[Path]:
    return sorted([p for p in runs_dir.iterdir() if _is_run_dir(p)])


def _infer_mode_seed(cfg: Optional[dict], run_name: str) -> Tuple[Optional[str], Optional[int]]:
    mode = None
    seed = None
    if cfg:
        seed = cfg.get("seed", None)
        proc = cfg.get("processing_parameters", {}) or {}
        mode = proc.get("peak_mode", None)

    name = run_name.lower()
    if mode is None:
        if "mcsa" in name:
            mode = "mcsa"
        elif "random" in name:
            mode = "random"

    if seed is None:
        match = re.search(r"seed(\d+)", name)
        if match:
            seed = int(match.group(1))

    return mode, seed


def _infer_task(cfg: Optional[dict], run_dir: Path) -> Optional[str]:
    if cfg and cfg.get("task"):
        return str(cfg["task"])
    if (run_dir / BIN_CSV).exists():
        return "binary"
    if (run_dir / MULTI_CSV).exists():
        return "multiclass"
    return None


def _load_binary_arrays(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    df = pd.read_csv(csv_path, index_col=0)
    if "binary_label" in df.columns:
        y_true = df["binary_label"].astype(int).to_numpy()
    elif "state" in df.columns:
        y_true = (df["state"].astype(str).str.strip() != "normal").astype(int).to_numpy()
    else:
        raise ValueError(f"Missing binary labels in {csv_path}")

    if "binary_prediction" in df.columns:
        y_pred = df["binary_prediction"].astype(int).to_numpy()
    elif "binary_prediction_state" in df.columns:
        y_pred = (df["binary_prediction_state"].astype(str).str.strip() != "normal").astype(int).to_numpy()
    else:
        raise ValueError(f"Missing binary predictions in {csv_path}")

    y_score = None
    if "score_anomalous" in df.columns:
        y_score = df["score_anomalous"].astype(float).to_numpy()

    return y_true, y_pred, y_score


def _resolve_prob_cols(df: pd.DataFrame, classes: Iterable[str]) -> List[str]:
    prob_cols = []
    for cname in classes:
        col = f"score_{SHORT_NAMES.get(cname, cname)}"
        if col not in df.columns:
            alt = f"score_{cname}"
            if alt in df.columns:
                col = alt
            else:
                return []
        prob_cols.append(col)
    return prob_cols


def _load_multiclass_arrays(run_dir: Path, csv_path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    enc_path = run_dir / ENCODER_PKL
    if not enc_path.exists():
        raise FileNotFoundError(f"Missing label encoder at {enc_path}")
    enc = joblib.load(enc_path)

    df = pd.read_csv(csv_path, index_col=0)
    if "state" not in df.columns:
        raise ValueError(f"Missing 'state' column in {csv_path}")
    y_true = enc.transform(df["state"].astype(str).str.strip().to_numpy())

    if "multiclass_prediction" in df.columns:
        y_pred = df["multiclass_prediction"].astype(int).to_numpy()
    elif "multiclass_prediction_state" in df.columns:
        y_pred = enc.transform(
            df["multiclass_prediction_state"].astype(str).str.strip().to_numpy()
        )
    else:
        raise ValueError(f"Missing multiclass predictions in {csv_path}")

    prob_cols = _resolve_prob_cols(df, enc.classes_)
    probs = df[prob_cols].to_numpy(dtype=float) if prob_cols else None
    return y_true, y_pred, probs


def _safe_roc_auc_binary(y_true: np.ndarray, y_score: Optional[np.ndarray]) -> float:
    if y_score is None or len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def _safe_roc_auc_multiclass(y_true: np.ndarray, probs: Optional[np.ndarray]) -> float:
    if probs is None:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")


def _compute_metrics(task: str, y_true: np.ndarray, y_pred: np.ndarray,
                     scores: Optional[np.ndarray]) -> Dict[str, float]:
    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    if task == "binary":
        roc_auc = _safe_roc_auc_binary(y_true, scores)
    else:
        roc_auc = _safe_roc_auc_multiclass(y_true, scores)

    return {
        "accuracy": acc,
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "macro_f1": float(f1),
        "roc_auc": roc_auc,
        "n_samples": int(len(y_true)),
    }


def _load_loss_history(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "epoch" not in df.columns:
        return None
    cols = ["epoch"]
    if "train_loss" in df.columns:
        cols.append("train_loss")
    if "val_loss" in df.columns:
        cols.append("val_loss")
    return df[cols].copy()


def _summarize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "roc_auc"]
    rows = []
    for (task, mode), g in df.groupby(["task", "mode"]):
        row = {"task": task, "mode": mode, "n_runs": int(len(g))}
        for metric in metrics:
            vals = g[metric].astype(float).to_numpy()
            valid = vals[~np.isnan(vals)]
            row[f"{metric}_mean"] = float(np.mean(valid)) if valid.size else float("nan")
            row[f"{metric}_std"] = float(np.std(valid, ddof=1)) if valid.size > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_loss(loss_by_mode: Dict[str, List[pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for mode, dfs in loss_by_mode.items():
        if not dfs:
            continue
        merged = pd.concat(dfs, ignore_index=True)
        grouped = merged.groupby("epoch")
        mean = grouped.mean(numeric_only=True)
        std = grouped.std(ddof=1, numeric_only=True).fillna(0.0)
        count = grouped.count()

        agg = pd.concat(
            [
                mean.add_suffix("_mean"),
                std.add_suffix("_std"),
                count.add_suffix("_n"),
            ],
            axis=1,
        ).reset_index()
        agg["mode"] = mode
        rows.append(agg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _plot_metric_bars(summary: pd.DataFrame, out_dir: Path, std_mult: float) -> None:
    fig_dir = out_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics = [c.replace("_mean", "") for c in summary.columns if c.endswith("_mean")]
    tasks = sorted(summary["task"].unique().tolist())
    mode_order = ["mcsa", "random"]
    colors = {"mcsa": "#1f77b4", "random": "#ff7f0e"}

    for task in tasks:
        df_task = summary[summary["task"] == task].copy()
        df_task["mode"] = pd.Categorical(df_task["mode"], categories=mode_order, ordered=True)
        df_task = df_task.sort_values("mode")
        modes = df_task["mode"].tolist()

        for metric in metrics:
            means = df_task[f"{metric}_mean"].to_numpy(dtype=float)
            stds = df_task[f"{metric}_std"].to_numpy(dtype=float) * std_mult

            x = np.arange(len(modes))
            plt.figure()
            plt.bar(
                x,
                means,
                yerr=stds,
                capsize=6,
                color=[colors.get(m, "#888888") for m in modes],
                alpha=0.9,
            )
            plt.xticks(x, modes)
            plt.ylabel(metric.replace("_", " ").upper())
            plt.title(f"{metric.replace('_', ' ').title()} ({task}) mean ± {std_mult} std")
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(fig_dir / f"metric_{metric}_{task}.png", dpi=300)
            plt.close()


def _plot_loss_curves(loss_summary: pd.DataFrame, out_dir: Path, std_mult: float) -> None:
    if loss_summary.empty:
        return
    fig_dir = out_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for loss_key in ("train_loss", "val_loss"):
        if f"{loss_key}_mean" not in loss_summary.columns:
            continue
        plt.figure()
        for mode, df_mode in loss_summary.groupby("mode"):
            df_mode = df_mode.sort_values("epoch")
            epochs = df_mode["epoch"].to_numpy(dtype=float)
            mean = df_mode[f"{loss_key}_mean"].to_numpy(dtype=float)
            std = df_mode.get(f"{loss_key}_std", pd.Series(0, index=df_mode.index)).to_numpy(dtype=float)
            band = std_mult * std
            plt.plot(epochs, mean, label=mode)
            plt.fill_between(epochs, mean - band, mean + band, alpha=0.2)

        plt.xlabel("Epoch")
        plt.ylabel(loss_key.replace("_", " ").title())
        plt.title(f"{loss_key.replace('_', ' ').title()} mean ± {std_mult} std")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / f"loss_{loss_key}.png", dpi=300)
        plt.close()


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Aggregate SGDA peak-ablation results across seeds."
    )
    ap.add_argument(
        "--runs-dir",
        type=str,
        default="res/runs/sgda_peak_ablation",
        help="Directory containing run subfolders.",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for aggregated tables/plots (default: <runs_dir>/peak_ablation_results).",
    )
    ap.add_argument(
        "--std-mult",
        type=float,
        default=3.0,
        help="Std multiplier for plot bands/error bars (default: 3).",
    )
    ap.add_argument(
        "--plot-loss",
        action="store_true",
        help="Include loss history aggregation and plots if loss_history.csv is present.",
    )
    return ap.parse_args()


def main() -> None:
    """Run peak-ablation aggregation and plotting.

    Returns
    -------
    None
        This function writes summary files and plots to disk.
    """
    args = _parse_args()
    runs_dir = Path(args.runs_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else runs_dir / "peak_ablation_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(out_dir)
    logger.info("Runs dir: %s", runs_dir)
    logger.info("Output dir: %s", out_dir)

    run_dirs = _discover_run_dirs(runs_dir)
    if not run_dirs:
        logger.error("No run folders found in %s", runs_dir)
        return

    per_run_records: List[Dict[str, object]] = []
    loss_by_mode: Dict[str, List[pd.DataFrame]] = {}

    for run_dir in tqdm(run_dirs, desc="Scanning runs"):
        cfg_path = run_dir / "training_config.yaml"
        cfg = _load_yaml(cfg_path) if cfg_path.exists() else None
        mode, seed = _infer_mode_seed(cfg, run_dir.name)
        if mode not in {"mcsa", "random"}:
            logger.info("Skipping run without peak_mode mcsa/random: %s", run_dir.name)
            continue

        task = _infer_task(cfg, run_dir)
        if task is None:
            logger.warning("Skipping run without task/predictions: %s", run_dir.name)
            continue

        try:
            if task == "binary":
                y_true, y_pred, scores = _load_binary_arrays(run_dir / BIN_CSV)
            else:
                y_true, y_pred, scores = _load_multiclass_arrays(run_dir, run_dir / MULTI_CSV)
        except Exception as exc:
            logger.warning("Failed to load predictions for %s: %s", run_dir.name, exc)
            continue

        metrics = _compute_metrics(task, y_true, y_pred, scores)
        record = {
            "run_name": run_dir.name,
            "mode": mode,
            "seed": seed,
            "task": task,
            "path": str(run_dir),
            **metrics,
        }
        per_run_records.append(record)

        if args.plot_loss:
            loss_df = _load_loss_history(run_dir / LOSS_CSV)
            if loss_df is not None and not loss_df.empty:
                loss_by_mode.setdefault(mode, []).append(loss_df)

    if not per_run_records:
        logger.error("No valid runs found to aggregate.")
        return

    per_run_df = pd.DataFrame(per_run_records)
    if "seed" in per_run_df.columns:
        dup = (
            per_run_df.dropna(subset=["seed"])
            .groupby(["mode", "seed"])
            .size()
        )
        if (dup > 1).any():
            logger.warning(
                "Detected multiple runs for the same (mode, seed). "
                "Aggregation will include all runs unless you filter manually."
            )
    per_run_csv = out_dir / "metrics_per_run.csv"
    per_run_df.to_csv(per_run_csv, index=False)
    logger.info("Saved per-run metrics: %s", per_run_csv)

    summary_df = _summarize_metrics(per_run_df)
    summary_csv = out_dir / "metrics_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    logger.info("Saved summary metrics: %s", summary_csv)

    summary_json = out_dir / "metrics_summary.json"
    summary_payload = {
        "runs_dir": str(runs_dir),
        "out_dir": str(out_dir),
        "std_mult": float(args.std_mult),
        "summary": summary_df.to_dict(orient="records"),
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    _plot_metric_bars(summary_df, out_dir, std_mult=args.std_mult)
    logger.info("Saved metric comparison plots under: %s", out_dir / "figs")

    if args.plot_loss and loss_by_mode:
        loss_summary = _aggregate_loss(loss_by_mode)
        if not loss_summary.empty:
            loss_csv = out_dir / "loss_summary.csv"
            loss_summary.to_csv(loss_csv, index=False)
            logger.info("Saved loss summary: %s", loss_csv)
            _plot_loss_curves(loss_summary, out_dir, std_mult=args.std_mult)
            logger.info("Saved loss plots under: %s", out_dir / "figs")
        else:
            logger.info("Loss histories were missing or empty; skipping loss plots.")


if __name__ == "__main__":
    main()
