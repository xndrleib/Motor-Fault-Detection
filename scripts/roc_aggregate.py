#!/usr/bin/env python3
# tools/roc_aggregate.py
"""
Aggregate ROC curves across multiple runs (different seeds) for the same
(task, model, config-without-seed), compute mean ± std ROC and AUC, and plot.

Discovery:
- Provide one or more paths (run dirs or experiment dirs). The script will:
  * find run folders containing training_config.yaml and a results CSV,
  * read training_config.yaml, drop the 'seed' field, and hash the normalized config
    to group runs that differ only by seed,
  * group by (task, model, normalized_config_hash).

Outputs per group:
- Plots:
  * Binary: figs/roc_agg_binary.(png|pdf) in the group's output dir.
  * Multiclass: figs/roc_agg_overlay.(png|pdf) and figs/roc_agg_micro.(png|pdf).
- JSON with per-class AUC mean/std and per-FPR-grid mean/std TPR.
- CSV table with AUC statistics.

Usage
-----
python tools/roc_aggregate.py res/runs/exp1 res/runs/exp2 \
  --task binary \
  --out-dir res/analysis \
  --fpr-points 201
"""
import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import joblib

SHORT_NAMES = {
    "bearing defect": "BD",
    "inter-turn short circuits": "ITSC",
    "rotor bar defect": "RBD",
    "normal": "Normal",
}

BIN_CSV = "segments_metadata_test_binary_pred.csv"
MULTI_CSV = "segments_metadata_test_multiclass_pred.csv"
ENCODER_PKL = "label_encoder_multiclass.pkl"


def _is_run_dir(d: Path) -> bool:
    return (d / "training_config.yaml").exists() and ((d / BIN_CSV).exists() or (d / MULTI_CSV).exists())


def _discover_run_dirs(paths: List[Path]) -> List[Path]:
    runs = []
    for p in paths:
        p = p.resolve()
        if _is_run_dir(p):
            runs.append(p)
            continue
        # Walk one or two levels deep to catch runs under exp folders
        for sub in p.rglob("*"):
            if sub.is_dir() and _is_run_dir(sub):
                runs.append(sub)
    # de-duplicate
    uniq = sorted(set(runs))
    return uniq


def _normalize_config(cfg: dict) -> dict:
    """Return a copy of cfg with seed removed so that only-seed differences collapse."""
    def _strip(d):
        if isinstance(d, dict):
            return {k: _strip(v) for k, v in d.items() if k != "seed"}
        if isinstance(d, list):
            return [_strip(x) for x in d]
        return d
    return _strip(cfg)


def _cfg_fingerprint(cfg: dict) -> str:
    """Stable hash of a normalized config (sorted keys YAML)."""
    dumped = yaml.safe_dump(cfg, sort_keys=True)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()[:16]


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_binary(run_dir: Path):
    df = pd.read_csv(run_dir / BIN_CSV, index_col=0)
    y_true = (df["state"].astype(str) != "normal").astype(int).to_numpy()
    y_score = df["score_anomalous"].astype(float).to_numpy()
    return y_true, y_score


def _load_multiclass(run_dir: Path):
    df = pd.read_csv(run_dir / MULTI_CSV, index_col=0)
    enc = joblib.load(run_dir / ENCODER_PKL)
    classes: List[str] = list(enc.classes_)
    y_idx = enc.transform(df["state"].astype(str).to_numpy())

    prob_cols = []
    for cname in classes:
        col = f"score_{SHORT_NAMES.get(cname, cname)}"
        if col not in df.columns:
            alt = f"score_{cname}"
            if alt in df.columns:
                col = alt
            else:
                raise ValueError(f"Missing probability column for class {cname}")
        prob_cols.append(col)
    P = df[prob_cols].to_numpy(float)
    return y_idx, P, classes


def _normalize_roc(fpr: np.ndarray, tpr: np.ndarray, force_endpoints: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return monotone, de-duplicated ROC with explicit endpoints, without creating
    duplicate FPR=0 or FPR=1 entries that can confuse interpolation.

    Steps:
    - sort by FPR
    - enforce non-decreasing TPR
    - collapse duplicate FPRs (keep max TPR per FPR)
    - if first FPR == 0 → force TPR[0] = 0; elif first FPR > 0 → prepend (0,0)
    - if last  FPR == 1 → force TPR[-1] = 1; elif last  FPR < 1 → append (1,1)
    """
    fpr = np.asarray(fpr, float)
    tpr = np.asarray(tpr, float)
    if fpr.size == 0 or tpr.size == 0:
        return fpr, tpr

    # sort by FPR
    order = np.argsort(fpr)
    fpr, tpr = fpr[order], tpr[order]

    # enforce non-decreasing TPR
    tpr = np.maximum.accumulate(tpr)

    # collapse duplicate FPRs: keep max TPR per FPR
    uniq = np.unique(fpr)
    tpr_max = np.array([tpr[fpr == u].max() for u in uniq], dtype=float)
    fpr, tpr = uniq, tpr_max

    if force_endpoints:
        # left endpoint
        if np.isclose(fpr[0], 0.0):
            tpr[0] = 0.0
        elif fpr[0] > 0.0:
            fpr = np.insert(fpr, 0, 0.0)
            tpr = np.insert(tpr, 0, 0.0)
        # right endpoint
        if np.isclose(fpr[-1], 1.0):
            tpr[-1] = 1.0
        elif fpr[-1] < 1.0:
            fpr = np.append(fpr, 1.0)
            tpr = np.append(tpr, 1.0)

    return fpr, tpr


def _interp_on_grid(fpr, tpr, grid):
    # Assumes endpoints present; left/right protect anyway.
    return np.interp(grid, fpr, tpr, left=0.0, right=1.0)


def _ensure_outdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "figs").mkdir(parents=True, exist_ok=True)
    return path


def _plot_binary_agg(out_dir: Path, grid: np.ndarray, tpr_stack: np.ndarray, aucs: List[float], title_extra: str):
    mean_tpr = np.nanmean(tpr_stack, axis=0)
    std_tpr = np.nanstd(tpr_stack, axis=0)
    mu_auc = float(np.nanmean(aucs)) if len(aucs) else float("nan")
    sd_auc = float(np.nanstd(aucs)) if len(aucs) else float("nan")

    plt.figure()
    plt.plot(grid, mean_tpr, label=f"Binary (AUC {mu_auc:.3f} ± {sd_auc:.3f})")
    plt.fill_between(grid, np.maximum(0, mean_tpr - std_tpr), np.minimum(1, mean_tpr + std_tpr), alpha=0.2)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Aggregated ROC – Binary{title_extra}")
    plt.legend(loc="lower right")
    for ext in (".png", ".pdf"):
        plt.savefig(out_dir / "figs" / f"roc_agg_binary{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    return mu_auc, sd_auc


def _plot_multiclass_agg(out_dir: Path, grid: np.ndarray, class_to_tpr_stack: Dict[str, np.ndarray],
                         class_to_aucs: Dict[str, List[float]], classes: List[str], title_extra: str):
    # Overlay of per-class means
    plt.figure()
    stats = {}
    for cls in classes:
        stack = class_to_tpr_stack.get(cls)
        aucs = class_to_aucs.get(cls, [])
        if stack is None or stack.size == 0:
            continue
        mean_tpr = np.nanmean(stack, axis=0)
        std_tpr = np.nanstd(stack, axis=0)
        mu_auc = float(np.nanmean(aucs)) if len(aucs) else float("nan")
        sd_auc = float(np.nanstd(aucs)) if len(aucs) else float("nan")
        disp = SHORT_NAMES.get(cls, cls)
        plt.plot(grid, mean_tpr, label=f"{disp} (AUC {mu_auc:.3f} ± {sd_auc:.3f})")
        plt.fill_between(grid, np.maximum(0, mean_tpr - std_tpr), np.minimum(1, mean_tpr + std_tpr), alpha=0.2)
        stats[cls] = {"auc_mean": mu_auc, "auc_std": sd_auc}

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Aggregated ROC – Multiclass (OVR){title_extra}")
    plt.legend(loc="lower right")
    for ext in (".png", ".pdf"):
        plt.savefig(out_dir / "figs" / f"roc_agg_overlay{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    return stats


def _plot_micro_agg(out_dir: Path, grid: np.ndarray, micro_tpr_stack: np.ndarray, micro_aucs: List[float], title_extra: str):
    mean_tpr = np.nanmean(micro_tpr_stack, axis=0)
    std_tpr = np.nanstd(micro_tpr_stack, axis=0)
    mu_auc = float(np.nanmean(micro_aucs)) if len(micro_aucs) else float("nan")
    sd_auc = float(np.nanstd(micro_aucs)) if len(micro_aucs) else float("nan")

    plt.figure()
    plt.plot(grid, mean_tpr, label=f"Micro (AUC {mu_auc:.3f} ± {sd_auc:.3f})")
    plt.fill_between(grid, np.maximum(0, mean_tpr - std_tpr), np.minimum(1, mean_tpr + std_tpr), alpha=0.2)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Aggregated ROC – Micro-average{title_extra}")
    plt.legend(loc="lower right")
    for ext in (".png", ".pdf"):
        plt.savefig(out_dir / "figs" / f"roc_agg_micro{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    return mu_auc, sd_auc


def main():
    ap = argparse.ArgumentParser(description="Aggregate ROC curves across seeds.")
    ap.add_argument("paths", nargs="+", help="Run or experiment directories to search for runs.")
    ap.add_argument("--task", choices=["binary", "multiclass"], required=True, help="Task to aggregate.")
    ap.add_argument("--out-dir", type=str, default="res/analysis", help="Where to save aggregated outputs.")
    ap.add_argument("--fpr-points", type=int, default=201, help="Number of points in the common FPR grid.")
    ap.add_argument("--model", type=str, default=None, help="Optional model filter (e.g., ResNet or CNN).")
    args = ap.parse_args()

    roots = [Path(p) for p in args.paths]
    run_dirs = _discover_run_dirs(roots)
    if not run_dirs:
        print("No runs found.")
        return

    # Build groups (task, model, cfg_hash)
    groups: Dict[Tuple[str, str, str], List[Path]] = {}
    cfg_meta: Dict[Tuple[str, str, str], dict] = {}

    for rd in run_dirs:
        cfg_path = rd / "training_config.yaml"
        try:
            cfg = _load_yaml(cfg_path)
        except Exception:
            continue

        task = cfg.get("task")
        model = cfg.get("model")
        if task != args.task:
            continue
        if args.model and model != args.model:
            continue

        norm = _normalize_config(cfg)
        h = _cfg_fingerprint(norm)
        key = (task, model, h)
        groups.setdefault(key, []).append(rd)
        cfg_meta.setdefault(key, {"normalized_config": norm, "hash": h})

    if not groups:
        print("No matching groups to aggregate.")
        return

    grid = np.linspace(0.0, 1.0, args.fpr_points, dtype=float)
    out_root = Path(args.out_dir).resolve()

    for (task, model, h), runs in groups.items():
        # Derive output dir name
        group_dir = _ensure_outdir(out_root / f"{task}_{model}_{h}")

        extra = False
        if extra:
            title_extra = f"\nmodel={model} | cfg={h} | seeds={len(runs)}"
        else:
            title_extra = ''

        if task == "binary":
            tpr_list = []
            aucs = []
            used_runs = []
            for rd in runs:
                try:
                    # Prefer saved JSON if present
                    roc_json = rd / "roc_data.json"
                    if roc_json.exists():
                        payload = json.loads(roc_json.read_text(encoding="utf-8"))
                        if payload.get("task") == "binary":
                            fpr = np.asarray(payload["fpr"], float)
                            tpr = np.asarray(payload["tpr"], float)
                            auc = float(payload["auc"])
                        else:
                            raise RuntimeError("roc_data.json is multiclass here.")
                    else:
                        y_true, y_score = _load_binary(rd)
                        fpr, tpr, _ = roc_curve(y_true, y_score)
                        auc = float(roc_auc_score(y_true, y_score))

                    # Normalize then interpolate
                    fpr, tpr = _normalize_roc(fpr, tpr)
                    tpr_list.append(_interp_on_grid(fpr, tpr, grid))
                    aucs.append(auc)
                    used_runs.append(str(rd))
                except Exception as e:
                    print(f"Skip run {rd}: {e}")

            if not tpr_list:
                print(f"[{model}/{h}] No valid binary runs.")
                continue

            tpr_stack = np.vstack(tpr_list)
            mu_auc, sd_auc = _plot_binary_agg(group_dir, grid, tpr_stack, aucs, title_extra)

            # Save summary
            summary = {
                "task": task,
                "model": model,
                "config_hash": h,
                "n_runs": len(aucs),
                "fpr_grid": grid.tolist(),
                "tpr_mean": np.nanmean(tpr_stack, axis=0).tolist(),
                "tpr_std": np.nanstd(tpr_stack, axis=0).tolist(),
                "auc_mean": mu_auc,
                "auc_std": sd_auc,
                "runs": used_runs,
            }
            (group_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            pd.DataFrame({"fpr": grid, "tpr_mean": summary["tpr_mean"], "tpr_std": summary["tpr_std"]}) \
              .to_csv(group_dir / "tpr_stats.csv", index=False)

        else:  # multiclass
            # Collect per-class stacks
            classes_ref: Optional[List[str]] = None
            class_to_tpr_stack: Dict[str, List[np.ndarray]] = {}
            class_to_aucs: Dict[str, List[float]] = {}
            micro_tpr_list: List[np.ndarray] = []
            micro_aucs: List[float] = []
            used_runs = []

            for rd in runs:
                try:
                    roc_json = rd / "roc_data.json"
                    if roc_json.exists():
                        payload = json.loads(roc_json.read_text(encoding="utf-8"))
                        if payload.get("task") != "multiclass":
                            raise RuntimeError("roc_data.json is binary here.")
                        classes = payload["classes"]
                        if classes_ref is None:
                            classes_ref = classes
                        elif classes != classes_ref:
                            raise RuntimeError("Class order mismatch across runs.")
                        per = payload["per_class"]
                        for cls in classes:
                            if cls in per and "fpr" in per[cls]:
                                fpr = np.asarray(per[cls]["fpr"], float)
                                tpr = np.asarray(per[cls]["tpr"], float)
                                auc = float(per[cls]["auc"])
                                fpr, tpr = _normalize_roc(fpr, tpr)
                                class_to_tpr_stack.setdefault(cls, []).append(_interp_on_grid(fpr, tpr, grid))
                                class_to_aucs.setdefault(cls, []).append(auc)
                        if "micro" in payload:
                            fpr_m = np.asarray(payload["micro"]["fpr"], float)
                            tpr_m = np.asarray(payload["micro"]["tpr"], float)
                            auc_m = float(payload["micro"]["auc"])
                            fpr_m, tpr_m = _normalize_roc(fpr_m, tpr_m)
                            micro_tpr_list.append(_interp_on_grid(fpr_m, tpr_m, grid))
                            micro_aucs.append(auc_m)
                    else:
                        # recompute from CSV
                        y_idx, P, classes = _load_multiclass(rd)
                        if classes_ref is None:
                            classes_ref = classes
                        elif classes != classes_ref:
                            raise RuntimeError("Class order mismatch across runs.")

                        present = np.unique(y_idx)
                        Y = label_binarize(y_idx, classes=list(range(len(classes))))
                        # per class
                        for c, cls in enumerate(classes):
                            if c not in present:
                                continue
                            y_true_c = Y[:, c]
                            y_score_c = P[:, c]
                            fpr_c, tpr_c, _ = roc_curve(y_true_c, y_score_c)
                            auc_c = float(roc_auc_score(y_true_c, y_score_c))
                            fpr_c, tpr_c = _normalize_roc(fpr_c, tpr_c)
                            class_to_tpr_stack.setdefault(cls, []).append(_interp_on_grid(fpr_c, tpr_c, grid))
                            class_to_aucs.setdefault(cls, []).append(auc_c)
                        # micro
                        fpr_m, tpr_m, _ = roc_curve(Y.ravel(), P.ravel())
                        auc_m = float(roc_auc_score(Y.ravel(), P.ravel()))
                        fpr_m, tpr_m = _normalize_roc(fpr_m, tpr_m)
                        micro_tpr_list.append(_interp_on_grid(fpr_m, tpr_m, grid))
                        micro_aucs.append(auc_m)

                    used_runs.append(str(rd))
                except Exception as e:
                    print(f"Skip run {rd}: {e}")

            if classes_ref is None:
                print(f"[{model}/{h}] No valid multiclass runs.")
                continue

            # Convert lists to stacks
            class_to_tpr_stack_arr = {cls: np.vstack(stk) if len(stk) else np.empty((0, grid.size))
                                      for cls, stk in class_to_tpr_stack.items()}

            _ = _plot_multiclass_agg(group_dir, grid, class_to_tpr_stack_arr, class_to_aucs, classes_ref, title_extra)
            mu_micro, sd_micro = (np.nan, np.nan)
            if len(micro_tpr_list):
                mu_micro, sd_micro = _plot_micro_agg(group_dir, grid, np.vstack(micro_tpr_list), micro_aucs, title_extra)

            # Save summary JSON and CSV
            summary = {
                "task": task,
                "model": model,
                "config_hash": h,
                "n_runs": len(used_runs),
                "fpr_grid": grid.tolist(),
                "per_class": {
                    cls: {
                        "tpr_mean": (np.nanmean(class_to_tpr_stack_arr[cls], axis=0).tolist()
                                     if class_to_tpr_stack_arr.get(cls) is not None and class_to_tpr_stack_arr[cls].size
                                     else []),
                        "tpr_std": (np.nanstd(class_to_tpr_stack_arr[cls], axis=0).tolist()
                                    if class_to_tpr_stack_arr.get(cls) is not None and class_to_tpr_stack_arr[cls].size
                                    else []),
                        "auc_mean": float(np.nanmean(class_to_aucs.get(cls, []))) if class_to_aucs.get(cls) else float("nan"),
                        "auc_std": float(np.nanstd(class_to_aucs.get(cls, []))) if class_to_aucs.get(cls) else float("nan"),
                    }
                    for cls in classes_ref
                },
                "micro": {
                    "auc_mean": mu_micro,
                    "auc_std": sd_micro,
                },
                "runs": used_runs,
            }
            (group_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

            # Flat CSV of AUC stats
            rows = []
            for cls in classes_ref:
                rows.append({"class": cls,
                             "auc_mean": summary["per_class"][cls]["auc_mean"],
                             "auc_std": summary["per_class"][cls]["auc_std"]})
            rows.append({"class": "micro", "auc_mean": mu_micro, "auc_std": sd_micro})
            pd.DataFrame(rows).to_csv(group_dir / "auc_stats.csv", index=False)

    print(f"[✓] Aggregation complete. See outputs under: {out_root}")


if __name__ == "__main__":
    main()
