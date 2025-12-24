#!/usr/bin/env python
"""Generate configs for noise-robust training with time-domain AWGN."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import yaml


def _parse_int_list(values: Iterable[str]) -> list[int]:
    return [int(v) for v in values]


def _parse_schedule(values: Iterable[str]) -> List[dict]:
    schedule = []
    for item in values:
        parts = item.split(":")
        if len(parts) != 4:
            raise ValueError(
                "Schedule items must be start_pct:end_pct:snr_min:snr_max."
            )
        schedule.append(
            {
                "start_pct": float(parts[0]),
                "end_pct": float(parts[1]),
                "snr_min": float(parts[2]),
                "snr_max": float(parts[3]),
            }
        )
    return schedule


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate config files for noise-robust training."
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
        "--p-clean",
        type=float,
        default=0.3,
        help="Probability of keeping a sample clean (default: 0.3).",
    )
    ap.add_argument(
        "--snr-min",
        type=float,
        default=20.0,
        help="Base minimum SNR for sampling (default: 20).",
    )
    ap.add_argument(
        "--snr-max",
        type=float,
        default=40.0,
        help="Base maximum SNR for sampling (default: 40).",
    )
    ap.add_argument(
        "--distribution",
        type=str,
        default="uniform",
        choices=["uniform", "mixture", "bin_weighted"],
        help="SNR sampling distribution (default: uniform).",
    )
    ap.add_argument(
        "--knee-range",
        type=float,
        nargs=2,
        default=[18.0, 22.0],
        help="Knee range for mixture distribution (default: 18 22).",
    )
    ap.add_argument(
        "--knee-weight",
        type=float,
        default=0.5,
        help="Probability mass on knee range for mixture (default: 0.5).",
    )
    ap.add_argument(
        "--disable-curriculum",
        action="store_true",
        help="Disable the default curriculum schedule.",
    )
    ap.add_argument(
        "--schedule",
        type=str,
        nargs="*",
        default=[
            "0.0:0.2:30:40",
            "0.2:0.6:20:35",
            "0.6:1.0:10:30",
        ],
        help="Curriculum schedule items: start:end:min:max (default uses 3 stages).",
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
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        choices=["binary", "multiclass"],
        help="Generate configs for multiple tasks (overrides --task if set).",
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
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    schedule = _parse_schedule(args.schedule)

    records = []
    idx = 1
    tasks = args.tasks if args.tasks else ([args.task] if args.task else [base_cfg.get("task", "binary")])
    for task in tasks:
        for seed in seeds:
            cfg = copy.deepcopy(base_cfg)
            cfg["model"] = args.model
            cfg["task"] = task
            cfg["seed"] = int(seed)
            cfg["test_run"] = bool(args.test_run)

            cfg.setdefault("model_parameters", {})
            cfg["model_parameters"]["attention_module"] = False

            proc_cfg = cfg.setdefault("processing_parameters", {})
            proc_cfg["train_noise_policy"] = {
                "enabled": True,
                "p_clean": float(args.p_clean),
                "snr_db": {
                    "distribution": args.distribution,
                    "snr_min": float(args.snr_min),
                    "snr_max": float(args.snr_max),
                    "knee_range": [
                        float(args.knee_range[0]),
                        float(args.knee_range[1]),
                    ],
                    "knee_weight": float(args.knee_weight),
                },
                "curriculum": {
                    "enabled": (not args.disable_curriculum),
                    "schedule": schedule,
                },
                "epsilon": 1e-12,
            }

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

            fname = f"train_noiserobust_{task}_seed{seed}_{idx}.yml"
            out_path = out_dir / fname
            with out_path.open("w", encoding="utf-8") as outf:
                yaml.safe_dump(cfg, outf, sort_keys=False)

            records.append(
                {
                    "filename": fname,
                    "seed": int(seed),
                    "p_clean": float(args.p_clean),
                    "snr_min": float(args.snr_min),
                    "snr_max": float(args.snr_max),
                    "distribution": args.distribution,
                    "knee_range": list(args.knee_range),
                    "knee_weight": float(args.knee_weight),
                    "curriculum_enabled": not args.disable_curriculum,
                    "schedule": schedule,
                    "model": cfg.get("model"),
                    "task": cfg.get("task"),
                    "base_cfg": str(base_path),
                    "timestamp": timestamp,
                }
            )
            idx += 1

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "config_index.csv", index=False)
    print(f"Generated {len(records)} configs in {out_dir}")


if __name__ == "__main__":
    main()
