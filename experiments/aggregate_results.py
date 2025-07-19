import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report
from tqdm.auto import tqdm

# === Paths ===
RES_DIR = Path("../res")
RUNS_DIR = RES_DIR / "runs"
EXP_NAME = 'exp8'
CFG_TEMPLATE = r"(train_engine-ResNet-1-phase-40-load-removeES-\d+-\d+)"
CONFIG_CSV = Path(f"../training_configs/{EXP_NAME}/config_index.csv")

# === Load configuration mapping ===
config_df = pd.read_csv(CONFIG_CSV)

# === Metric-safe helper ===
def safe_classification_report(y_true, y_pred):
    y_true = y_true.astype(str).str.strip()
    y_pred = y_pred.astype(str).str.strip()
    all_labels = sorted(set(y_true.unique()).union(set(y_pred.unique())))
    report = classification_report(y_true, y_pred, labels=all_labels, output_dict=True, zero_division=0)
    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
    }

# === Collect metrics for each result folder ===
results = []

for exp_dir in tqdm(RUNS_DIR.iterdir()):
    if not exp_dir.is_dir():
        continue

    match = re.search(CFG_TEMPLATE, exp_dir.name)
    if not match:
        continue

    cfg_file = match.group(1)
    cfg_id = cfg_file

    row = config_df[config_df["filename"] == cfg_file+'.yml']
    if row.empty:
        print(f"Skipping unmatched config: {cfg_file}")
        continue
    params = row.iloc[0].to_dict()

    task = params["task"]
    if task == "binary":
        pred_path = exp_dir / "segments_metadata_test_binary_pred.csv"
        label_col = "binary_label"
        pred_col = "binary_prediction"
    else:
        pred_path = exp_dir / "segments_metadata_test_multiclass_pred.csv"
        label_col = "multiclass_label"
        pred_col = "multiclass_prediction_state"

    if not pred_path.exists():
        print(f"Missing prediction file for: {cfg_file}")
        continue

    df = pd.read_csv(pred_path)
    true_labels = df[label_col]
    pred_labels = df[pred_col]

    # Safe metrics
    metrics = safe_classification_report(true_labels, pred_labels)
    result = {**params, **metrics, 'path_to_run': exp_dir}
    results.append(result)

# === Save results to dataframe ===
metrics_df = pd.DataFrame(results)
save_path = RES_DIR / EXP_NAME / "aggregate_results.csv"
metrics_df.to_csv(save_path, index=False)
print(f"Saved aggregated metrics to {save_path}")
