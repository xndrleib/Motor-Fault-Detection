#!/usr/bin/env python3
"""AWGN robustness evaluation for time-domain signals (before FFT)."""
import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from src.data_pipeline import (
    filter_segments,
    load_measurement,
    perform_fft_on_segments,
    read_oscilloscope_data,
    segment_signal,
    make_importance_mask,
)
from src.electrical_signature_frequencies import ANOMALY_FREQS
from src.inference import inference_model
from src.models import CNN, ResNet, ResidualBlock
from src.normalization import Normalizer
from src.utils import set_all_seeds


@dataclass(frozen=True)
class RunInfo:
    name: str
    path: Path
    task: str
    config: dict


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate model robustness with time-domain AWGN before FFT."
    )
    parser.add_argument(
        "--runs_dir",
        type=str,
        default="res/runs/exp9",
        help="Directory containing run subfolders (default: res/runs/exp9).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for results (default: <runs_dir>/awgn_robustness).",
    )
    parser.add_argument(
        "--snr_list",
        type=float,
        nargs="+",
        default=[30, 20, 10, 5, 0],
        help="List of SNR values in dB (e.g., --snr_list 30 20 10 5 0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for deterministic noise generation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cpu/cuda/mps). Default: auto.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="dataset",
        help="Root directory for dataset (default: dataset).",
    )
    return parser.parse_args()


def _detect_device(override: Optional[str] = None) -> torch.device:
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _find_runs(runs_dir: Path) -> List[RunInfo]:
    runs = []
    for p in sorted(runs_dir.iterdir()):
        if not p.is_dir():
            continue
        cfg_path = p / "training_config.yaml"
        if not cfg_path.exists():
            continue
        cfg = yaml.safe_load(cfg_path.read_text())
        task = cfg.get("task")
        if task not in ("binary", "multiclass"):
            continue
        runs.append(RunInfo(name=p.name, path=p, task=task, config=cfg))
    return runs


def _group_runs_by_task(runs: Iterable[RunInfo]) -> Dict[str, List[RunInfo]]:
    grouped: Dict[str, List[RunInfo]] = {"binary": [], "multiclass": []}
    for run in runs:
        grouped[run.task].append(run)
    return {k: v for k, v in grouped.items() if v}


def _config_signature(cfg: dict) -> str:
    payload = {
        "engineLabel": cfg.get("engineLabel"),
        "processing_parameters": cfg.get("processing_parameters", {}),
        "dataset_parameters": cfg.get("dataset_parameters", {}),
        "data_parameters": cfg.get("data_parameters", {}),
        "model": cfg.get("model"),
        "model_parameters": cfg.get("model_parameters", {}),
    }
    return json.dumps(payload, sort_keys=True)


def _awgn(segments: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    if snr_db is None or np.isinf(snr_db):
        return segments.copy()
    segs = segments.astype(np.float32, copy=True)
    power = np.mean(segs ** 2, axis=1, keepdims=True)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = np.where(power > 0, power / snr_linear, 0.0)
    noise_std = np.sqrt(noise_power)
    noise = rng.normal(0.0, 1.0, size=segs.shape).astype(np.float32) * noise_std
    return segs + noise


def _build_segments_with_awgn(
    metadata_df: pd.DataFrame,
    segment_length: int,
    step: int,
    f_sampling: int,
    cutoff_freq: int,
    db: bool,
    snr_db: float,
    seed: int,
) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    seg_rows: List[dict] = []
    seg_arrays: List[np.ndarray] = []
    freqs: Optional[np.ndarray] = None
    rng = np.random.default_rng(seed)

    for m_id, meta in metadata_df.iterrows():
        if str(meta["file_path"]).endswith(".csv"):
            df = load_measurement(m_id, metadata_df)
            current = df["Current"].dropna().values
        else:
            df = read_oscilloscope_data(meta["file_path"])
            current = df["Data"].dropna().values

        win_arr = segment_signal(
            current,
            segment_length=segment_length,
            step=step,
            apply_window=False,
        )
        win_arr = _awgn(win_arr, snr_db=snr_db, rng=rng)

        fft_arr, freqs = perform_fft_on_segments(
            win_arr,
            f_sampling=f_sampling,
            cutoff_freq=cutoff_freq,
            db=db,
        )

        n_segments = fft_arr.shape[0]
        for s_idx in range(n_segments):
            start_t = df.index[s_idx * step]
            end_t = df.index[s_idx * step + segment_length - 1]
            seg_rows.append(
                {
                    "measurement_id": m_id,
                    "base_id": meta["base_id"],
                    "segment_idx": s_idx,
                    "start_time": float(start_t),
                    "end_time": float(end_t),
                    "state": meta["state"],
                    "binary_label": 0 if meta["state"] == "normal" else 1,
                    "multiclass_label": meta["state"],
                    "phase": int(meta["phase"]),
                    "load_condition": meta["load_condition"],
                    "experiment": meta["experiment"],
                }
            )

        seg_arrays.append(fft_arr.astype(np.float32))

    segments = np.vstack(seg_arrays)
    seg_meta_df = pd.DataFrame(seg_rows).set_index(["measurement_id", "segment_idx"])
    if freqs is None:
        raise RuntimeError("Failed to compute FFT frequencies.")
    return segments, seg_meta_df, freqs


def _load_normalizer(run_dir: Path, cfg: dict, task: str) -> Optional[Normalizer]:
    norm_cfg = cfg.get("dataset_parameters", {})
    normalizer = Normalizer(
        method=norm_cfg.get("normalization_method", "min-max"),
        mode=norm_cfg.get("normalization_mode", "global"),
    )
    norm_path = run_dir / f"normalizer_{task}.json"
    if norm_path.exists():
        normalizer.load(norm_path)
        return normalizer
    if normalizer.mode == "per":
        return normalizer
    return None


def _load_label_encoder(run_dir: Path) -> Optional[object]:
    enc_path = run_dir / "label_encoder_multiclass.pkl"
    if enc_path.exists():
        return joblib.load(enc_path)
    return None


def _build_model(
    cfg: dict,
    num_classes: int,
    freqs: np.ndarray,
    engine_dir: Path,
) -> torch.nn.Module:
    model_type = cfg.get("model", "ResNet")
    model_params = cfg.get("model_parameters", {})
    attention = model_params.get("attention_module", False)

    prior_kwargs = None
    if attention:
        engine_cfg = yaml.safe_load((engine_dir / "engine.yml").read_text())
        fault_types = cfg["processing_parameters"]["fault_types_to_use"]
        MCSA_cfg = {
            "rotor bar defect": {"engine_config": engine_cfg["mcsa"], "n_range": range(1, 4)},
            "inter-turn short circuits": {
                "engine_config": engine_cfg["mcsa"],
                "k_range": range(1, 4, 2),
                "m_range": range(0, 2),
            },
            "bearing defect": {"engine_config": engine_cfg["mcsa"]},
        }
        fault_freqs = {
            ft: ANOMALY_FREQS[ft](**MCSA_cfg[ft]) for ft in fault_types if ft in ANOMALY_FREQS
        }
        mask = make_importance_mask(
            freqs,
            fault_freqs,
            delta_hz=cfg["processing_parameters"]["peak_segment"],
        )
        mask_torch = torch.tensor(mask, dtype=torch.float32)[None, None, :]
        prior_kwargs = dict(
            mask=mask_torch,
            w_in_init=1.0,
            w_out_init=0.0,
            learnable_in=True,
            learnable_out=False,
        )

    if model_type == "ResNet":
        model = ResNet(
            ResidualBlock,
            [2, 2, 2, 2],
            num_classes=num_classes,
            dropout_rate=model_params.get("dropout", 0.0),
            prior_kwargs=prior_kwargs,
        )
    elif model_type == "CNN":
        model = CNN(
            num_classes=num_classes,
            dropout_rate=model_params.get("dropout", 0.0),
            prior_kwargs=prior_kwargs,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    task: str,
) -> Dict[str, float]:
    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    roc_auc = float("nan")
    try:
        if task == "binary":
            if len(np.unique(y_true)) >= 2:
                roc_auc = float(roc_auc_score(y_true, probs[:, 1]))
        else:
            roc_auc = float(
                roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
            )
    except Exception:
        roc_auc = float("nan")

    return {
        "accuracy": acc,
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "macro_f1": float(f1),
        "roc_auc": roc_auc,
    }


def _aggregate_metrics(per_run: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    agg: Dict[str, Dict[str, float]] = {}
    if not per_run:
        return agg
    keys = per_run[0].keys()
    for key in keys:
        vals = np.array([r[key] for r in per_run], dtype=float)
        valid = vals[~np.isnan(vals)]
        mean = float(np.mean(valid)) if valid.size else float("nan")
        std = float(np.std(valid, ddof=1)) if valid.size > 1 else 0.0
        agg[key] = {"mean": mean, "std": std, "n": int(valid.size)}
    return agg


def _plot_degradation(
    out_dir: Path,
    snr_list: List[float],
    agg_by_task: Dict[str, Dict[float, Dict[str, Dict[str, float]]]],
    metric: str,
    title: str,
) -> None:
    plt.figure()
    for task, per_snr in agg_by_task.items():
        means = [per_snr[snr][metric]["mean"] for snr in snr_list]
        stds = [per_snr[snr][metric]["std"] for snr in snr_list]
        plt.plot(snr_list, means, marker="o", label=task)
        plt.fill_between(
            snr_list,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            alpha=0.2,
        )
    plt.xlabel("SNR (dB)")
    plt.ylabel(metric.replace("_", " ").upper() if metric != "roc_auc" else "ROC AUC")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"degradation_{metric}.png", dpi=300)
    plt.close()


def main() -> None:
    args = _parse_args()
    set_all_seeds(args.seed)

    runs_dir = Path(args.runs_dir).resolve()
    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else runs_dir / "awgn_robustness"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _detect_device(args.device)
    runs = _find_runs(runs_dir)
    if not runs:
        raise FileNotFoundError(f"No runs found in {runs_dir}")

    grouped = _group_runs_by_task(runs)
    summary: Dict[str, object] = {
        "runs_dir": str(runs_dir),
        "out_dir": str(out_dir),
        "snr_list_db": args.snr_list,
        "seed": args.seed,
        "device": str(device),
        "tasks": {},
    }

    rows = []
    agg_for_plots: Dict[str, Dict[float, Dict[str, Dict[str, float]]]] = {}

    for task, task_runs in grouped.items():
        sigs = {_config_signature(r.config) for r in task_runs}
        if len(sigs) != 1:
            raise ValueError(
                f"Found multiple configs for task={task}. "
                "Ensure runs share the same preprocessing/model settings."
            )
        cfg = task_runs[0].config
        proc = cfg["processing_parameters"]
        data_root = Path(args.data_root)
        engine_dir = data_root / cfg.get("engineLabel", "engine_2")

        metadata_df = pd.read_csv(engine_dir / "metadata.csv").set_index("measurement_id")

        training_classes = proc["fault_types_to_use"] + ["normal"]
        loads_to_use = proc["loads_to_use"]
        phases_to_use = proc["phases_to_use"]

        agg_for_plots[task] = {}
        task_payload = {
            "runs": [r.name for r in task_runs],
            "metrics": {},
        }

        for snr_db in args.snr_list:
            snr_seed = int(args.seed + snr_db * 100)
            segments, seg_meta_df, freqs = _build_segments_with_awgn(
                metadata_df=metadata_df,
                segment_length=proc["segment_length"],
                step=proc["shift"],
                f_sampling=proc["f_sampling"],
                cutoff_freq=proc["cutoff_freq"],
                db=proc["db"],
                snr_db=snr_db,
                seed=snr_seed,
            )

            seg_meta_df, segments = filter_segments(
                seg_meta_df,
                segments,
                training_classes,
                loads_to_use,
                phases_to_use,
            )

            per_run_metrics: List[Dict[str, float]] = []
            per_run_samples: List[int] = []
            per_run_detail: Dict[str, Dict[str, float]] = {}

            for run in task_runs:
                test_idx_path = run.path / "indices" / f"test_idx_{task}.npy"
                if not test_idx_path.exists():
                    raise FileNotFoundError(f"Missing test indices: {test_idx_path}")
                test_idx = np.load(test_idx_path)

                X_test = segments[test_idx]
                meta_test = seg_meta_df.iloc[test_idx].copy()

                normalizer = _load_normalizer(run.path, cfg, task)
                if normalizer is not None:
                    X_test = normalizer.transform(X_test)

                if task == "binary":
                    y_true = meta_test["binary_label"].astype(int).to_numpy()
                    num_classes = 2
                else:
                    enc = _load_label_encoder(run.path)
                    if enc is None:
                        raise FileNotFoundError(
                            f"Missing label encoder for multiclass in {run.path}"
                        )
                    y_true = enc.transform(meta_test["state"].astype(str).str.strip())
                    num_classes = len(enc.classes_)

                X_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
                y_tensor = torch.tensor(y_true, dtype=torch.long)
                loader = DataLoader(
                    TensorDataset(X_tensor, y_tensor),
                    batch_size=cfg["data_parameters"]["batch_size"],
                    shuffle=False,
                )

                model = _build_model(cfg, num_classes=num_classes, freqs=freqs, engine_dir=engine_dir)
                ckpt = run.path / "checkpoints" / f"best_{task}.pth"
                if not ckpt.exists():
                    raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
                pkg = torch.load(ckpt, map_location=device)
                model.load_state_dict(pkg['state_dict'])
                model.to(device)

                res = inference_model(model, loader, device=device, return_extra=True)
                metrics = _compute_metrics(res["labels"], res["preds"], res["probs"], task)
                per_run_metrics.append(metrics)
                per_run_samples.append(len(res["labels"]))
                per_run_detail[run.name] = metrics

            agg_metrics = _aggregate_metrics(per_run_metrics)
            agg_for_plots[task][snr_db] = agg_metrics
            n_samples = int(np.mean(per_run_samples)) if per_run_samples else 0

            rows.append(
                {
                    "snr_db": snr_db,
                    "task_type": task,
                    "metrics": json.dumps(agg_metrics),
                    "n_samples": n_samples,
                }
            )

            task_payload["metrics"][snr_db] = {
                "aggregate": agg_metrics,
                "per_run": per_run_detail,
                "n_samples": n_samples,
            }

        summary["tasks"][task] = task_payload

    results_csv = out_dir / "results.csv"
    pd.DataFrame(rows).to_csv(results_csv, index=False)

    results_json = out_dir / "results.json"
    results_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _plot_degradation(
        out_dir,
        snr_list=args.snr_list,
        agg_by_task=agg_for_plots,
        metric="macro_f1",
        title="Macro F1 vs SNR",
    )
    _plot_degradation(
        out_dir,
        snr_list=args.snr_list,
        agg_by_task=agg_for_plots,
        metric="accuracy",
        title="Accuracy vs SNR",
    )
    _plot_degradation(
        out_dir,
        snr_list=args.snr_list,
        agg_by_task=agg_for_plots,
        metric="roc_auc",
        title="ROC AUC vs SNR",
    )

    print(f"[✓] Saved results to {out_dir}")


if __name__ == "__main__":
    main()
