#!/usr/bin/env python
"""Run SGDA peak-mode ablation (MCSA vs random) across multiple seeds."""

from __future__ import annotations

import argparse
import datetime as dt
import copy
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import yaml

from src.sgda_peak_selection import resolve_peak_location_seed


def setup_logger(log_dir: Path, log_file: str = "peak_ablation.log") -> logging.Logger:
    """Create a logger that writes to console and a file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_file

    logger = logging.getLogger("peak_ablation")
    logger.setLevel(logging.INFO)

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info("Logging to file: %s", log_path)
    return logger


def _load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dump_cfg(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _parse_seeds(values: Iterable[str]) -> list[int]:
    return [int(v) for v in values]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run SGDA peak-mode ablation (mcsa vs random) across seeds."
    )
    ap.add_argument("--cfg", type=str, required=True, help="Base training config YAML.")
    ap.add_argument(
        "--exp-name",
        type=str,
        default="sgda_peak_ablation",
        help="Experiment group name under res/runs.",
    )
    ap.add_argument(
        "--runs-dir",
        type=str,
        default=None,
        help="Override runs root directory (defaults to res/runs).",
    )
    ap.add_argument(
        "--seeds",
        type=str,
        nargs="+",
        default=["1", "2", "3", "4", "5"],
        help="List of seeds to run (default: 1 2 3 4 5).",
    )
    ap.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=["mcsa", "random"],
        help="Peak modes to run (default: mcsa random).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs without executing.",
    )
    args = ap.parse_args()

    cfg_path = Path(args.cfg).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Training config not found: {cfg_path}")

    repo_root = Path(__file__).resolve().parents[1]
    runs_root = Path(args.runs_dir) if args.runs_dir else (repo_root / "res" / "runs")
    exp_root = runs_root / args.exp_name
    logger = setup_logger(exp_root)

    base_cfg = _load_cfg(cfg_path)
    task = base_cfg.get("task")
    if task is None:
        raise ValueError("Training config must define 'task' (binary/multiclass).")

    seeds = _parse_seeds(args.seeds)
    modes = [m.lower().strip() for m in args.modes]
    cfg_stem = cfg_path.stem
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    manifest: list[dict] = []

    for mode in modes:
        for seed in seeds:
            cfg = copy.deepcopy(base_cfg)
            cfg["seed"] = int(seed)
            cfg["model"] = "ResNet"
            cfg.setdefault("model_parameters", {})
            cfg["model_parameters"]["attention_module"] = False

            proc_cfg = cfg.setdefault("processing_parameters", {})
            proc_cfg["peak_mode"] = mode
            proc_cfg["peak_location_seed"] = resolve_peak_location_seed(
                cfg.get("seed"), proc_cfg.get("peak_location_seed", None)
            )

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

            run_name = f"{cfg_stem}_{mode}_seed{seed}_{timestamp}"
            cfg_out = exp_root / "configs" / f"{run_name}.yml"
            _dump_cfg(cfg, cfg_out)

            cmd = [
                sys.executable,
                str(repo_root / "experiments" / "train.py"),
                "--cfg",
                str(cfg_out),
                "--exp-name",
                args.exp_name,
                "--run-name",
                run_name,
            ]
            if args.runs_dir:
                cmd += ["--runs-dir", str(runs_root)]

            manifest.append(
                {
                    "run_name": run_name,
                    "mode": mode,
                    "seed": int(seed),
                    "config": str(cfg_out),
                    "command": cmd,
                }
            )

            logger.info("Planned run: %s | mode=%s | seed=%s", run_name, mode, seed)
            if args.dry_run:
                continue

            logger.info("Executing: %s", " ".join(cmd))
            subprocess.run(cmd, check=True)

    manifest_path = exp_root / "ablation_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
