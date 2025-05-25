import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Load aggregated results
RESULTS_FOLDER = Path('../res/')
df = pd.read_csv(RESULTS_FOLDER / "aggregate_results.csv")


# Number of top configs to report
TOP_N = 5

# Columns to analyze
param_cols = ["batch_size", "dropout", "attention_module", "normalization_mode"]

# Analyze each task separately
for task in ["binary", "multiclass"]:
    df_task = df[df["task"] == task].copy()

    print(f"\n=== {task.upper()} CLASSIFICATION ===")
    print(f"Total configurations: {len(df_task)}")

    # === Plotting
    for param in param_cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=param, y="macro_f1", data=df_task)
        plt.title(f"Effect of {param} on Macro F1 - {task.capitalize()}")
        plt.xlabel(param.replace("_", " ").capitalize())
        plt.ylabel("Macro F1 Score")
        plt.grid(True)
        plt.tight_layout()
        fname = f"analysis_{param}_{task}"
        plt.savefig(RESULTS_FOLDER / f"{fname}.png", dpi=300)
        plt.savefig(RESULTS_FOLDER / f"{fname}.pdf", dpi=300)
        plt.close()
        print(f"Saved: {fname}")

    # === Top Configurations by F1 ===
    top_df = df_task.sort_values("macro_f1", ascending=False).head(TOP_N)
    print(f"\nTop {TOP_N} Configurations by Macro F1 ({task}):\n")
    print(top_df[["filename", "macro_f1", "batch_size", "dropout", "attention_module", "normalization_mode"]])

    # Save top configs to CSV
    top_df.to_csv(RESULTS_FOLDER / f"top_{task}_configs.csv", index=False)
    print(f"Saved top configs CSV: top_{task}_configs.csv")
