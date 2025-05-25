import itertools
from pathlib import Path
import yaml
import pandas as pd

# 1. Paths
BASE_CFG_PATH = Path("train_engine-2.yml")
OUT_DIR = Path(".")

# 2. Parameter grid
attention_vals = [True, False]
dropout_vals = [0.2, 0.3, 0.4]
batch_size_vals = [256, 512, 1024, 2048, 4096]
norm_mode_vals = ["global", "per"]
tasks = ['multiclass', 'binary']

# 3. Load base config
with BASE_CFG_PATH.open() as f:
    base_cfg = yaml.safe_load(f)

# 4. Store all config metadata
records = []

# 5. Iterate over all combinations
i = 1
for attention, dropout, batch_size, norm_mode, task in itertools.product(
    attention_vals, dropout_vals, batch_size_vals, norm_mode_vals, tasks
):
    # a) Clone the base
    cfg = base_cfg.copy()

    # b) Update nested entries
    cfg["model_parameters"]["attention_module"] = attention
    cfg["model_parameters"]["dropout"] = float(dropout)
    cfg["data_parameters"]["batch_size"] = int(batch_size)
    cfg["dataset_parameters"]["normalization_mode"] = norm_mode
    cfg["task"] = task
    cfg['comet_online'] = False
    cfg['test_run'] = False
    cfg['training_parameters']['num_epochs_binary'] = 25
    cfg['training_parameters']['num_epochs_multi'] = 25

    # c) Name the config for clarity
    fname = f"train-{i}.yml"
    out_path = OUT_DIR / fname
    with out_path.open("w") as outf:
        yaml.safe_dump(cfg, outf)

    print(f"Written config: {out_path}")

    # d) Save metadata for tracking
    records.append({
        "filename": fname,
        "attention_module": attention,
        "dropout": dropout,
        "batch_size": batch_size,
        "normalization_mode": norm_mode,
        "task": task
    })

    i += 1

# 6. Save all records to CSV
df = pd.DataFrame(records)
df.to_csv(OUT_DIR / "config_index.csv", index=False)
print("Saved config index to config_index.csv")
