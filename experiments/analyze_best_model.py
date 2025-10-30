import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)

from src.evaluation import slice_metrics
from src.visualization import plot_heatmap

# === CLI ARGUMENTS ===
parser = argparse.ArgumentParser(description="Evaluate best model.")
parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Filename of best config (e.g. train-74.yml)",
)
parser.add_argument(
    "--res-dir", type=str, default="../res", help="Path to result directory"
)
args = parser.parse_args()

RES_DIR = Path(args.res_dir)
CFG_FILE = args.config


# === Detect Task from YAML ===
def detect_task_from_config(cfg_file):
    search_path = list(RES_DIR.glob(f"*{cfg_file}*"))
    if not search_path:
        raise FileNotFoundError(f"No result folder found for {cfg_file}")
    config_path = search_path[0] / "training_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Cannot find training config for {cfg_file} in {search_path[0]}"
        )
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["task"], search_path[0]


task, exp_dir = detect_task_from_config(CFG_FILE)
print(f"Evaluating {task.upper()} model from: {CFG_FILE}")
print(f"Located experiment folder: {exp_dir}")

# === Load prediction CSV ===
if task == "binary":
    pred_path = exp_dir / "segments_metadata_test_binary_pred.csv"
else:
    pred_path = exp_dir / "segments_metadata_test_multiclass_pred.csv"

if not pred_path.exists():
    raise FileNotFoundError(f"Prediction file missing: {pred_path}")

df = pd.read_csv(pred_path)

# === Segment-Level Evaluation ===
if task == "binary":
    df["binary_label"] = df["binary_label"].astype(int)
    df["binary_prediction"] = df["binary_prediction"].astype(int)

    cm = confusion_matrix(df["binary_label"], df["binary_prediction"])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["normal", "anomalous"],
        yticklabels=["normal", "anomalous"],
    )
    plt.title("Binary Confusion Matrix – Segment Level")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(RES_DIR / "binary_confusion_matrix_segment.png", dpi=300)
    plt.savefig(RES_DIR / "binary_confusion_matrix_segment.pdf", dpi=300)
    plt.close()

    # Slice metrics — Segment level
    seg_slice = slice_metrics(
        df, y_true="binary_label", y_pred="binary_prediction", average="binary"
    )
    plot_heatmap(
        seg_slice.pivot(index="load", columns="phase", values="f1"),
        "Binary (Segment-Level) – F₁ by Load & Phase",
        save=True,
        save_path=RES_DIR,
    )

    # Measurement-level majority vote
    agg = df.groupby("base_id").agg(
        n_segments=("binary_prediction", "size"),
        n_pred_anomalous=("binary_prediction", "sum"),
        true_label=("binary_label", "first"),
        phase=("phase", "first"),
        load_condition=("load_condition", "first"),
    )
    agg["pred_majority"] = (agg["n_pred_anomalous"] / agg["n_segments"] >= 0.5).astype(
        int
    )

    precision, recall, f1_score, support = precision_recall_fscore_support(
        agg["true_label"], agg["pred_majority"], average="macro", zero_division=0
    )
    print(
        f"Binary (Measurement) → Acc={accuracy_score(agg['true_label'], agg['pred_majority']):.3f}, "
        f"Precision={precision:.3f}, Recall={recall:.3f}, F1-score={f1_score:.3f}"
    )

    heat = slice_metrics(
        agg, y_true="true_label", y_pred="pred_majority", average="binary"
    )
    plot_heatmap(
        heat.pivot(index="load", columns="phase", values="f1"),
        "Binary (Majority Vote) – F₁ by Load & Phase",
        save=True,
        save_path=RES_DIR,
    )
    agg.to_csv(RES_DIR / "measurement_metrics_binary.csv", index=True)

elif task == "multiclass":
    true_labels = df["multiclass_label"].astype(str).str.strip()
    pred_labels = df["multiclass_prediction"].astype(str).str.strip()

    classes = sorted(true_labels.unique())
    short_names = {
        "bearing defect": "BD",
        "inter-turn short circuits": "ITSC",
        "rotor bar defect": "RBD",
        "normal": "Normal",
    }

    # Slice metrics — Segment level
    df["multiclass_label"] = df["multiclass_label"].astype(str).str.strip()
    df["multiclass_prediction"] = df["multiclass_prediction"].astype(str).str.strip()

    seg_slice = slice_metrics(
        df, y_true="multiclass_label", y_pred="multiclass_prediction", average="macro"
    )
    plot_heatmap(
        seg_slice.pivot(index="load", columns="phase", values="f1"),
        "Multiclass (Segment-Level) – F₁ by Load & Phase",
        save=True,
        save_path=RES_DIR,
    )

    cm = confusion_matrix(true_labels, pred_labels, labels=classes)
    classes = [short_names[class_name] for class_name in classes]
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.title("Multiclass Confusion Matrix – Segment Level")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45)
    plt.savefig(RES_DIR / "multiclass_confusion_matrix_segment.png", dpi=300)
    plt.savefig(RES_DIR / "multiclass_confusion_matrix_segment.pdf", dpi=300)
    plt.close()

    # Measurement-level vote
    label_counts = (
        df.groupby("base_id")["multiclass_prediction"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    label_counts["pred_majority"] = label_counts.idxmax(axis=1)
    label_counts["true_label"] = df.groupby("base_id")["multiclass_label"].first()
    label_counts["phase"] = df.groupby("base_id")["phase"].first()
    label_counts["load_condition"] = df.groupby("base_id")["load_condition"].first()

    accuracy = accuracy_score(label_counts["true_label"], label_counts["pred_majority"])
    precision, recall, f1_score, support = precision_recall_fscore_support(
        label_counts["true_label"],
        label_counts["pred_majority"],
        average="macro",
        zero_division=0,
    )
    print(
        f"Multiclass (Measurement) → Accuracy={accuracy:.3f}, "
        f"Precision={precision:.3f}, Recall={recall:.3f}, F1-score={f1_score:.3f}"
    )

    heat = slice_metrics(
        label_counts, y_true="true_label", y_pred="pred_majority", average="macro"
    )
    plot_heatmap(
        heat.pivot(index="load", columns="phase", values="f1"),
        "Multiclass (Majority Vote) – F₁ by Load & Phase",
        save=True,
        save_path=RES_DIR,
    )
    label_counts.to_csv("measurement_metrics_multiclass.csv", index=True)

    # === Multiclass → Binary Remapping ===
    print("\n▶ Multiclass → Binary evaluation:")
    to_binary = lambda x: 0 if x.strip() == "normal" else 1

    df["binary_label_from_multi"] = df["multiclass_label"].map(to_binary)
    df["binary_pred_from_multi"] = df["multiclass_prediction"].map(to_binary)

    cm_bin = confusion_matrix(
        df["binary_label_from_multi"], df["binary_pred_from_multi"]
    )
    sns.heatmap(
        cm_bin,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["normal", "fault"],
        yticklabels=["normal", "fault"],
    )
    plt.title("Multiclass → Binary – Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(RES_DIR / "multiclass_to_binary_confusion_matrix.png", dpi=300)
    plt.savefig(RES_DIR / "multiclass_to_binary_confusion_matrix.pdf", dpi=300)
    plt.close()

    seg_metrics = slice_metrics(
        df,
        y_true="binary_label_from_multi",
        y_pred="binary_pred_from_multi",
        average="binary",
    )
    plot_heatmap(
        seg_metrics.pivot(index="load", columns="phase", values="f1"),
        "Multiclass→Binary – F₁ by Load & Phase (segment)",
        save=True,
        save_path=RES_DIR,
    )

    # Measurement majority vote
    label_counts["binary_true"] = label_counts["true_label"].map(to_binary)
    label_counts["binary_pred"] = label_counts["pred_majority"].map(to_binary)

    mv_metrics = slice_metrics(
        label_counts, y_true="binary_true", y_pred="binary_pred", average="binary"
    )
    plot_heatmap(
        mv_metrics.pivot(index="load", columns="phase", values="f1"),
        "Multiclass→Binary – F₁ by Load & Phase (majority vote)",
        save=True,
        save_path=RES_DIR,
    )

    label_counts.to_csv(
        RES_DIR / "measurement_metrics_multiclass_binary.csv", index=True
    )
