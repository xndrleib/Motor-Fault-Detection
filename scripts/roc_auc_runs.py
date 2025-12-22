#!/usr/bin/env python3
# tools/roc_auc_runs.py
"""
Compute ROC AUC for one or more runs produced by experiments/train.py and
save ROC curve figures and (optionally) raw ROC arrays (roc_data.json)
into each run's folder.

Binary:
  - expects: segments_metadata_test_binary_pred.csv with `state`, `score_anomalous`
  - saves: figs/roc_binary.(png|pdf), and optionally roc_data.json

Multiclass:
  - expects: segments_metadata_test_multiclass_pred.csv with per-class scores
             and label_encoder_multiclass.pkl
  - saves: figs/roc_ovr_<class>.(png|pdf), figs/roc_ovr_overlay.(png|pdf),
           figs/roc_micro.(png|pdf), and optionally roc_data.json

Usage
-----
python tools/roc_auc_runs.py res/runs/exp1/runA res/runs/exp1/runB \
    --save-csv res/roc_auc_summary.csv --save-json res/roc_auc_summary.json \
    --save-roc-data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

SHORT_NAMES = {
    "bearing defect": "BD",
    "inter-turn short circuits": "ITSC",
    "rotor bar defect": "RBD",
    "normal": "Normal",
}

# ----------------------------- IO helpers ------------------------------------


def _detect_files(run_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    bin_csv = run_dir / "segments_metadata_test_binary_pred.csv"
    multi_csv = run_dir / "segments_metadata_test_multiclass_pred.csv"
    return (bin_csv if bin_csv.exists() else None,
            multi_csv if multi_csv.exists() else None)


def _figs_dir(run_dir: Path) -> Path:
    d = run_dir / "figs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_name(s: str) -> str:
    return (
        s.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(",", "_")
        .lower()
    )


# --------------------------- Loaders -----------------------------------------


def _load_binary(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path, index_col=0)
    if "state" not in df.columns:
        raise ValueError(f"'state' column not found in {csv_path}")
    if "score_anomalous" not in df.columns:
        raise ValueError(f"'score_anomalous' column not found in {csv_path}")

    y_true = (df["state"].astype(str) != "normal").astype(int).to_numpy()
    y_score = df["score_anomalous"].astype(float).to_numpy()
    return y_true, y_score


def _load_multiclass(run_dir: Path, csv_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    enc_path = run_dir / "label_encoder_multiclass.pkl"
    if not enc_path.exists():
        raise FileNotFoundError(f"Missing label encoder at {enc_path}")

    enc = joblib.load(enc_path)
    classes: List[str] = list(enc.classes_)  # encoder order
    display = [SHORT_NAMES.get(c, c) for c in classes]

    df = pd.read_csv(csv_path, index_col=0)
    if "state" not in df.columns:
        raise ValueError(f"'state' column not found in {csv_path}")

    y_true_idx = enc.transform(df["state"].astype(str).to_numpy())

    prob_cols = []
    for cname in classes:
        col = f"score_{SHORT_NAMES.get(cname, cname)}"
        if col not in df.columns:
            alt = f"score_{cname}"
            if alt in df.columns:
                col = alt
            else:
                raise ValueError(
                    f"Expected probability column '{col}' (or '{alt}') not found in {csv_path}"
                )
        prob_cols.append(col)

    P = df[prob_cols].to_numpy(dtype=float)  # (N, C)
    return y_true_idx, P, classes, display


# --------------------------- Plotting ----------------------------------------


def _plot_and_save(figpath_stem: Path, fpr: np.ndarray, tpr: np.ndarray, auc_val: float, title: str) -> None:
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_val:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    png = figpath_stem.with_suffix(".png")
    pdf = figpath_stem.with_suffix(".pdf")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_overlay(figpath_stem: Path, curves: List[Tuple[str, np.ndarray, np.ndarray, float]], title: str) -> None:
    plt.figure()
    for label, fpr, tpr, auc_val in curves:
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc_val:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    png = figpath_stem.with_suffix(".png")
    pdf = figpath_stem.with_suffix(".pdf")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, dpi=300, bbox_inches="tight")
    plt.close()


# ------------------------- AUC computation -----------------------------------


def _safe_binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _multiclass_aucs_and_curves(
    y_idx: np.ndarray, P: np.ndarray, classes: List[str], display_names: List[str]
):
    present = np.unique(y_idx)
    C = len(classes)
    Y = label_binarize(y_idx, classes=list(range(C)))
    aucs: Dict[str, float] = {}
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    per_class_vals = []
    for c in range(C):
        name = classes[c]
        if c not in present:
            aucs[name] = float("nan")
            continue
        y_true = Y[:, c]
        y_score = P[:, c]
        try:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_val = float(roc_auc_score(y_true, y_score))
            curves[name] = (fpr, tpr)
            aucs[name] = auc_val
            per_class_vals.append(auc_val)
        except Exception:
            aucs[name] = float("nan")

    aucs["macro_ovr"] = float(np.mean(per_class_vals)) if per_class_vals else float("nan")

    # Micro-average
    micro = None
    try:
        fpr_m, tpr_m, _ = roc_curve(Y.ravel(), P.ravel())
        auc_m = float(roc_auc_score(Y.ravel(), P.ravel()))
        micro = (fpr_m, tpr_m, auc_m)
    except Exception:
        micro = None

    return aucs, curves, micro


# ------------------------------ Runner ---------------------------------------


def eval_run(run_dir: Path, save_figs: bool = True, save_roc_data: bool = False) -> Dict[str, object]:
    bin_csv, multi_csv = _detect_files(run_dir)
    if bin_csv and multi_csv:
        raise ValueError(f"Both binary and multiclass CSVs present in {run_dir}; ambiguous.")
    if not bin_csv and not multi_csv:
        raise FileNotFoundError(
            f"No result CSV found in {run_dir}. Expected segments_metadata_test_*_pred.csv"
        )

    out: Dict[str, object] = {"run": str(run_dir)}
    figs_dir = _figs_dir(run_dir) if save_figs else None

    if bin_csv:
        y_true, y_score = _load_binary(bin_csv)
        auc_val = _safe_binary_auc(y_true, y_score)
        out["task"] = "binary"
        out["auc"] = auc_val
        out["n_samples"] = int(len(y_true))
        out["positive_rate"] = float(y_true.mean()) if len(y_true) else 0.0

        if auc_val is not None:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            if save_figs:
                stem = figs_dir / "roc_binary"
                _plot_and_save(stem, fpr, tpr, auc_val, title="ROC (Binary)")
                out["roc_figs"] = [str(stem.with_suffix(".png")), str(stem.with_suffix(".pdf"))]
            if save_roc_data:
                (run_dir / "roc_data.json").write_text(
                    json.dumps(
                        {"task": "binary", "fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": auc_val},
                        indent=2
                    ),
                    encoding="utf-8",
                )
        return out

    # Multiclass
    y_idx, P, classes, display = _load_multiclass(run_dir, multi_csv)
    aucs, curves, micro = _multiclass_aucs_and_curves(y_idx, P, classes, display)
    out["task"] = "multiclass"
    out["n_samples"] = int(len(y_idx))
    out["classes"] = classes
    out["per_class_auc"] = {cls: aucs.get(cls, float("nan")) for cls in classes}
    out["macro_ovr"] = aucs["macro_ovr"]

    if save_figs or save_roc_data:
        saved = []
        roc_payload = {"task": "multiclass", "classes": classes, "per_class": {}, "macro_ovr": aucs["macro_ovr"]}
        # Per-class plots/data
        for cls in classes:
            if cls in curves:
                fpr, tpr = curves[cls]
                disp = SHORT_NAMES.get(cls, cls)
                if save_figs:
                    stem = figs_dir / f"roc_ovr_{_safe_name(disp)}"
                    _plot_and_save(stem, fpr, tpr, aucs[cls], title=f"ROC OVR – {disp}")
                    saved.extend([str(stem.with_suffix(".png")), str(stem.with_suffix(".pdf"))])
                roc_payload["per_class"][cls] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(aucs[cls])}

        # Overlay
        if save_figs:
            overlay_curves = []
            for cls in classes:
                if cls in curves:
                    disp = SHORT_NAMES.get(cls, cls)
                    fpr, tpr = curves[cls]
                    overlay_curves.append((disp, fpr, tpr, aucs[cls]))
            if overlay_curves:
                stem = figs_dir / "roc_ovr_overlay"
                _plot_overlay(stem, overlay_curves, title="ROC OVR – All Classes")
                saved.extend([str(stem.with_suffix(".png")), str(stem.with_suffix(".pdf"))])

        # Micro-average
        if micro is not None:
            fpr_m, tpr_m, auc_m = micro
            if save_figs:
                stem = figs_dir / "roc_micro"
                _plot_and_save(stem, fpr_m, tpr_m, auc_m, title="ROC – Micro-average")
                saved.extend([str(stem.with_suffix(".png")), str(stem.with_suffix(".pdf"))])
            roc_payload["micro"] = {"fpr": fpr_m.tolist(), "tpr": tpr_m.tolist(), "auc": float(auc_m)}

        if save_figs:
            out["roc_figs"] = saved
        if save_roc_data:
            (run_dir / "roc_data.json").write_text(json.dumps(roc_payload, indent=2), encoding="utf-8")

    return out


def main():
    p = argparse.ArgumentParser(description="Compute ROC AUC and save ROC curves for specified run directories.")
    p.add_argument("runs", nargs="+", help="Paths to run directories (each containing results CSVs).")
    p.add_argument("--save-csv", type=str, default=None, help="Path to save a flat CSV summary.")
    p.add_argument("--save-json", type=str, default=None, help="Path to save a detailed JSON summary.")
    p.add_argument("--no-figs", dest="no_figs", action="store_true", help="Disable saving ROC curve figures.")
    p.add_argument("--save-roc-data", dest="save_roc_data", action="store_true", help="Also write <run>/roc_data.json.")
    args = p.parse_args()

    rows = []
    details = []
    for run in args.runs:
        run_dir = Path(run).resolve()
        res = eval_run(run_dir, save_figs=not args.no_figs, save_roc_data=args.save_roc_data)
        details.append(res)

        if res["task"] == "binary":
            rows.append({
                "run": str(run_dir),
                "task": "binary",
                "auc": res["auc"],
                "n_samples": res["n_samples"],
                "positive_rate": res["positive_rate"],
            })
        else:
            for cls, auc in res["per_class_auc"].items():
                rows.append({
                    "run": str(run_dir),
                    "task": "multiclass",
                    "class": cls,
                    "auc": auc,
                    "n_samples": res["n_samples"],
                })
            rows.append({
                "run": str(run_dir),
                "task": "multiclass",
                "class": "macro_ovr",
                "auc": res["macro_ovr"],
                "n_samples": res["n_samples"],
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        with pd.option_context("display.max_colwidth", None, "display.width", 120):
            print(df.to_string(index=False))
    else:
        print("No results computed.")

    if args.save_csv:
        out_csv = Path(args.save_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"\n[✓] Saved CSV summary → {out_csv}")

    if args.save_json:
        out_json = Path(args.save_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(details, f, indent=2)
        print(f"[✓] Saved JSON details → {out_json}")


if __name__ == "__main__":
    main()
