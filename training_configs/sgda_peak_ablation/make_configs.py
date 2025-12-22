#!/usr/bin/env python
"""Generate SGDA peak-ablation configs for parallel runs."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

from src.sgda_peak_selection import (
    DEFAULT_PEAK_SEED_OFFSET,
    resolve_peak_location_seed,
)


def _parse_int_list(values: Iterable[str]) -> list[int]:
    return [int(v) for v in values]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate config files for SGDA peak-mode ablation."
    )
    ap.add_argument(
        "--base-cfg",
        type=str,
        required=True,
        help="Path to the base training config YAML.",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=".",
        help="Directory to write generated configs and index CSV.",
    )
    ap.add_argument(
        "--seeds",
        type=str,
        nargs="+",
        default=["1", "2", "3", "4", "5"],
        help="Seeds to use (default: 1 2 3 4 5).",
    )
    ap.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=["mcsa", "random"],
        help="Peak modes to generate (default: mcsa random).",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="ResNet",
        help="Model name to set in configs (default: ResNet).",
    )
    ap.add_argument(
        "--task",
        type=str,
        default=None,
        choices=["binary", "multiclass"],
        help="Optional task override.",
    )
    ap.add_argument(
        "--test-run",
        action="store_true",
        help="Enable test_run in generated configs.",
    )
    ap.add_argument(
        "--disable-early-stopping",
        action="store_true",
        help="Set patience >= num_epochs to avoid early stopping.",
    )
    args = ap.parse_args()

    base_path = Path(args.base_cfg).resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_path}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with base_path.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    seeds = _parse_int_list(args.seeds)
    modes = [m.lower().strip() for m in args.modes]
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    records = []
    idx = 1
    for mode in modes:
        for seed in seeds:
            cfg = copy.deepcopy(base_cfg)
            cfg["model"] = args.model
            if args.task:
                cfg["task"] = args.task
            cfg["seed"] = int(seed)
            cfg["test_run"] = bool(args.test_run)

            cfg.setdefault("model_parameters", {})
            cfg["model_parameters"]["attention_module"] = False

            proc_cfg = cfg.setdefault("processing_parameters", {})
            proc_cfg["peak_mode"] = mode
            proc_cfg["peak_location_seed"] = resolve_peak_location_seed(
                cfg.get("seed"), proc_cfg.get("peak_location_seed", None)
            )

            if args.disable_early_stopping:
                task = cfg.get("task", "binary")
                train_cfg = cfg.setdefault("training_parameters", {})
                num_epochs_key = (
                    "num_epochs_binary" if task == "binary" else "num_epochs_multi"
                )
                num_epochs = int(train_cfg.get(num_epochs_key, 0))
                if num_epochs <= 0:
                    raise ValueError(f"Invalid {num_epochs_key} in training config.")
                train_cfg["patience"] = max(
                    int(train_cfg.get("patience", 0)), num_epochs + 1
                )

            fname = f"train_peakablation_{mode}_seed{seed}_{idx}.yml"
            out_path = out_dir / fname
            with out_path.open("w", encoding="utf-8") as outf:
                yaml.safe_dump(cfg, outf, sort_keys=False)

            records.append(
                {
                    "filename": fname,
                    "mode": mode,
                    "seed": int(seed),
                    "peak_location_seed": proc_cfg["peak_location_seed"],
                    "model": cfg.get("model"),
                    "task": cfg.get("task"),
                    "base_cfg": str(base_path),
                    "seed_offset": DEFAULT_PEAK_SEED_OFFSET,
                    "timestamp": timestamp,
                }
            )
            idx += 1

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "config_index.csv", index=False)
    print(f"Generated {len(records)} configs in {out_dir}")


if __name__ == "__main__":
    main()
