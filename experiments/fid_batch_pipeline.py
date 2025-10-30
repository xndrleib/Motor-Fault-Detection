#!/usr/bin/env python3
"""
Batch FID/nFID pipeline:
- For each specified training run, ensure embeddings and detailed FID/nFID table exist.
- If missing, compute them by calling the existing per-run scripts.
- Aggregate FID/nFID across runs (mean and std per (fault_code, load)).
- Save aggregated table in CSV and Parquet, plus a YAML metafile with provenance.

This script expects you already have:
  - experiments/build_fid_embeddings.py
  - experiments/compute_fid_table.py

Typical layout:
  res/
    fid_inputs/               # windows for embedding (real_*.npy, synth_*.npy)
    fid_embs/<RUN_ID>/        # per-run embeddings + fid_table.{csv,parquet}
    fid_aggregates/<TAG>/     # where this script will write aggregated outputs

Example
-------
python experiments/fid_batch_pipeline.py \
  --run-dirs res/runs/2025-06-03_02-14-40_train_full-data-removeES-42-16 \
             res/runs/2025-06-10_12-00-00_train_full-data-42-16 \
  --fid-dir res/fid_inputs \
  --emb-root res/fid_embs \
  --agg-root res/fid_aggregates \
  --agg-name demo_itsc_rbd \
  --rr-repeats 10 --seed 123 \
  --neg-baselines real-other-fault real-synth-other-fault synth-other-fault \
  --neg-percentile 0.5

Notes
-----
- Prefer Parquet if available; fall back to CSV.
- Skips a run if fid_table.{parquet,csv} already exists unless --force is used.
- Aggregation covers all metric-like columns starting with "fid" or "nfid".
- YAML metafile documents runs used, modes detected, parameters, timestamps, etc.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


# =============================================================================
# Files / existence checks
# =============================================================================


def emb_dir_for_run(run_dir: Path, emb_root: Path) -> Path:
    """
    Get the embeddings directory for a given run.

    Parameters
    ----------
    run_dir : Path
        Path to the specific training run directory under res/runs/…
    emb_root : Path
        Root directory containing per-run embeddings subfolders.

    Returns
    -------
    Path
        emb_root / run_dir.name
    """
    return emb_root / run_dir.name


def fid_table_paths(
    emb_dir: Path, csv_name: str, parquet_name: str
) -> Tuple[Path, Path]:
    """
    Resolve expected FID table filenames inside a run's embeddings directory.

    Parameters
    ----------
    emb_dir : Path
        Embeddings directory for this run.
    csv_name : str
        CSV filename (e.g., "fid_table.csv").
    parquet_name : str
        Parquet filename (e.g., "fid_table.parquet").

    Returns
    -------
    (Path, Path)
        (csv_path, parquet_path)
    """
    return (emb_dir / csv_name, emb_dir / parquet_name)


def fid_table_exists(emb_dir: Path, csv_name: str, parquet_name: str) -> bool:
    """
    Check if a run already has a detailed FID table (parquet or csv).

    Parameters
    ----------
    emb_dir : Path
        Per-run embeddings directory.
    csv_name : str
        CSV filename.
    parquet_name : str
        Parquet filename.

    Returns
    -------
    bool
        True if either Parquet or CSV exists.
    """
    csv_path, pq_path = fid_table_paths(emb_dir, csv_name, parquet_name)
    return pq_path.exists() or csv_path.exists()


def read_run_fid_table(emb_dir: Path, csv_name: str, parquet_name: str) -> pd.DataFrame:
    """
    Load the detailed FID/nFID table for a run, preferring Parquet.

    Parameters
    ----------
    emb_dir : Path
        Per-run embeddings directory.
    csv_name : str
        CSV filename.
    parquet_name : str
        Parquet filename.

    Returns
    -------
    pandas.DataFrame
        Detailed (wide) per-run table.
    """
    csv_path, pq_path = fid_table_paths(emb_dir, csv_name, parquet_name)
    if pq_path.exists():
        return pd.read_parquet(pq_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"No per-run FID table found at {pq_path} or {csv_path}")


# =============================================================================
# Subprocess helpers to call the existing scripts
# =============================================================================


def run_embed_script(
    embed_script: Path,
    run_dir: Path,
    fid_dir: Path,
    emb_root: Path,
    batch_size: Optional[int],
    device: str,
) -> None:
    """
    Call the embedding script for a single run.

    Parameters
    ----------
    embed_script : Path
        Path to experiments/build_fid_embeddings.py
    run_dir : Path
        Training run directory under res/runs/…
    fid_dir : Path
        Directory where real_*.npy / synth_*.npy live.
    emb_root : Path
        Root folder to place <run_id> subfolder with embeddings.
    batch_size : int or None
        Optional override for batch size; if None, let script use defaults.
    device : {'auto','cpu','cuda'}
        Device selection.
    """
    cmd = [
        sys.executable,
        str(embed_script),
        "--run-dir",
        str(run_dir),
        "--fid-dir",
        str(fid_dir),
        "--out-root",
        str(emb_root),
        "--device",
        device,
    ]
    if batch_size is not None:
        cmd += ["--batch-size", str(batch_size)]
    subprocess.run(cmd, check=True)


def run_compute_script(
    compute_script: Path,
    run_dir: Path,
    emb_root: Path,
    out_csv_name: str,
    out_parquet_name: str,
    rr_repeats: int,
    seed: int,
    eps: float,
    neg_baselines: List[str],
    neg_percentile: float,
) -> None:
    """
    Call the compute_fid_table script for a single run.

    Parameters
    ----------
    compute_script : Path
        Path to experiments/compute_fid_table.py (single-table version).
    run_dir : Path
        Training run directory under res/runs/…
    emb_root : Path
        Root embeddings folder containing per-run subfolders.
    out_csv_name : str
        Filename for CSV output (inside the run's embeddings dir).
    out_parquet_name : str
        Filename for Parquet output (inside the run's embeddings dir).
    rr_repeats : int
        Split-half repeats for FID_RR.
    seed : int
        RNG seed.
    eps : float
        Covariance jitter.
    neg_baselines : list of str
        Baselines to compute.
    neg_percentile : float
        Percentile across multiple negative candidates.
    """
    cmd = [
        sys.executable,
        str(compute_script),
        "--run-dir",
        str(run_dir),
        "--emb-root",
        str(emb_root),
        "--out-csv",
        out_csv_name,
        "--out-parquet",
        out_parquet_name,
        "--rr-repeats",
        str(rr_repeats),
        "--seed",
        str(seed),
        "--eps",
        str(eps),
        "--neg-percentile",
        str(neg_percentile),
    ]
    if neg_baselines:
        cmd += ["--neg-baselines"] + list(neg_baselines)
    subprocess.run(cmd, check=True)


# =============================================================================
# Aggregation
# =============================================================================


def detect_metric_columns(df: pd.DataFrame) -> List[str]:
    """
    Determine which columns are metric-like and should be aggregated.

    Strategy
    --------
    - Include columns that start with "fid" or "nfid".
    - Exclude 'fid_rw__*' if you only want normalized metrics? We include both,
      because the user asked for FID aggregation as well. Keeping all 'fid*'
      and 'nfid*' covers 'fid', 'fid_rr', 'fid_rw__*', and 'nfid__*'.

    Parameters
    ----------
    df : pandas.DataFrame
        Per-run detailed table.

    Returns
    -------
    list of str
        Column names to aggregate.
    """
    metrics = [c for c in df.columns if c.startswith("fid") or c.startswith("nfid")]
    # Make sure grouping keys are not in metrics:
    for k in ("run_id", "fault_code", "load"):
        if k in metrics:
            metrics.remove(k)
    return metrics


def aggregate_runs(
    per_run_tables: List[pd.DataFrame],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Aggregate mean and std across runs for each (fault_code, load).

    Parameters
    ----------
    per_run_tables : list of pandas.DataFrame
        Detailed tables (one per run), possibly with slightly different columns.

    Returns
    -------
    (pandas.DataFrame, list of str)
        Aggregated table and the list of metric columns that were aggregated.
    """
    if not per_run_tables:
        return pd.DataFrame(), []

    # Outer-join compatible columns by reindexing missing columns to NaN
    all_cols = set().union(*(df.columns for df in per_run_tables))
    normalized = []
    for df in per_run_tables:
        add = {c: np.nan for c in all_cols if c not in df.columns}
        normalized.append(df.assign(**add)[sorted(all_cols)])

    cat = pd.concat(normalized, ignore_index=True)

    # Metrics to aggregate
    metrics = detect_metric_columns(cat)

    # Group and aggregate; ignore NaNs when computing mean/std
    def _nanstd(x: pd.Series) -> float:
        # Sample std with ddof=1; return NaN if fewer than 2 non-NaN values
        arr = x.to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 2:
            return float("nan")
        return float(np.nanstd(arr, ddof=1))

    grp = cat.groupby(["fault_code", "load"], dropna=False)
    agg_mean = grp[metrics].mean(numeric_only=True)
    agg_std = grp[metrics].agg(_nanstd)

    # Flatten multiindex columns: metric_mean / metric_std
    agg_mean.columns = [f"{c}__mean" for c in agg_mean.columns]
    agg_std.columns = [f"{c}__std" for c in agg_std.columns]

    out = pd.concat([agg_mean, agg_std], axis=1).reset_index()

    # Add run_count per group
    run_counts = grp["run_id"].nunique().rename("run_count").reset_index()
    out = out.merge(run_counts, on=["fault_code", "load"], how="left")

    return out, metrics


# =============================================================================
# YAML metafile
# =============================================================================


def now_timestamps() -> Dict[str, str]:
    """
    Get both UTC and local timestamps in ISO 8601.
    """
    utc_now = datetime.now(timezone.utc).isoformat()
    local_now = datetime.now().isoformat(timespec="seconds")
    return {"created_utc": utc_now, "created_local": local_now}


def write_yaml(path: Path, data: Dict) -> None:
    """
    Write a YAML metafile with safe dumper.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


# =============================================================================
# Main driver
# =============================================================================


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Batch FID/nFID pipeline with aggregation and metadata."
    )
    # Inputs: runs
    ap.add_argument(
        "--run-dirs",
        nargs="*",
        type=Path,
        default=[],
        help="List of training run directories under res/runs/…",
    )
    ap.add_argument(
        "--runs-file",
        type=Path,
        default=None,
        help="Optional text file with one run directory per line.",
    )
    # Where inputs/outputs live
    ap.add_argument(
        "--fid-dir",
        type=Path,
        default=Path("res/fid_inputs"),
        help="Folder with real_*.npy / synth_*.npy for embedding.",
    )
    ap.add_argument(
        "--emb-root",
        type=Path,
        default=Path("res/fid_embs"),
        help="Root folder that holds per-run embeddings and FID tables.",
    )
    ap.add_argument(
        "--agg-root",
        type=Path,
        default=Path("res/fid_aggregates"),
        help="Root folder where aggregated outputs will be written.",
    )
    ap.add_argument(
        "--agg-name",
        type=str,
        default=None,
        help="Optional subfolder name under agg-root; if omitted, uses a timestamp.",
    )
    # Script paths (allow overriding layout)
    ap.add_argument(
        "--embed-script",
        type=Path,
        default=Path("experiments/build_fid_embeddings.py"),
        help="Path to the per-run embedding script.",
    )
    ap.add_argument(
        "--compute-script",
        type=Path,
        default=Path("experiments/compute_fid_table.py"),
        help="Path to the per-run FID/nFID script (single-table CSV+Parquet).",
    )
    # Per-run output filenames to check/write
    ap.add_argument(
        "--per-run-csv",
        type=str,
        default="fid_table.csv",
        help="Filename of the detailed table written inside each run's emb dir (CSV).",
    )
    ap.add_argument(
        "--per-run-parquet",
        type=str,
        default="fid_table.parquet",
        help="Filename of the detailed table written inside each run's emb dir (Parquet).",
    )
    # Recompute policy
    ap.add_argument(
        "--force",
        action="store_true",
        help="If set, recompute FID/nFID even if table exists.",
    )
    ap.add_argument(
        "--skip-embed",
        action="store_true",
        help="If set, do not run the embedding script (assumes embeddings exist).",
    )
    # Embedding options
    ap.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional override for embedding batch size.",
    )
    ap.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for embedding.",
    )
    # FID options
    ap.add_argument(
        "--rr-repeats", type=int, default=10, help="Split-half repeats for FID_RR."
    )
    ap.add_argument("--seed", type=int, default=123, help="RNG seed for split-halves.")
    ap.add_argument("--eps", type=float, default=1e-6, help="Covariance jitter.")
    ap.add_argument(
        "--neg-baselines",
        nargs="*",
        choices=["real-other-fault", "real-synth-other-fault", "synth-other-fault"],
        default=None,
        help="Which negative baselines to compute; if omitted, compute all.",
    )
    ap.add_argument(
        "--neg-percentile",
        type=float,
        default=0.5,
        help="Percentile across multiple 'other fault' candidates (0.5 = median).",
    )
    # Aggregated output filenames
    ap.add_argument(
        "--agg-csv",
        type=str,
        default="fid_aggregate.csv",
        help="Filename of aggregated table (CSV).",
    )
    ap.add_argument(
        "--agg-parquet",
        type=str,
        default="fid_aggregate.parquet",
        help="Filename of aggregated table (Parquet).",
    )
    ap.add_argument(
        "--agg-meta",
        type=str,
        default="fid_aggregate_meta.yaml",
        help="Filename of metadata YAML.",
    )

    args = ap.parse_args()

    # Collect run list
    run_dirs: List[Path] = list(args.run_dirs)
    if args.runs_file is not None and args.runs_file.exists():
        with open(args.runs_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    run_dirs.append(Path(line))
    if not run_dirs:
        raise SystemExit(
            "No run directories provided. Use --run-dirs ... or --runs-file file.txt"
        )

    # Choose aggregation output folder
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agg_sub = args.agg_name or f"agg_{stamp}"
    out_dir = args.agg_root / agg_sub
    out_dir.mkdir(parents=True, exist_ok=True)

    # Book-keeping
    per_run_status: Dict[str, Dict] = {}
    per_run_tables: List[pd.DataFrame] = []

    # Normalize neg-baselines set for compute script
    neg_baselines = args.neg_baselines or [
        "real-other-fault",
        "real-synth-other-fault",
        "synth-other-fault",
    ]

    # Process each run
    for run_dir in run_dirs:
        run_dir = run_dir.resolve()
        run_id = run_dir.name
        status = {
            "run_dir": str(run_dir),
            "computed": False,
            "skipped_preexisting": False,
            "errors": [],
        }

        emb_dir = emb_dir_for_run(run_dir, args.emb_root)
        try:
            need_compute = args.force or (
                not fid_table_exists(emb_dir, args.per_run_csv, args.per_run_parquet)
            )
            if need_compute:
                # Ensure embeddings exist (unless user asked to skip)
                if not args.skip_embed:
                    run_embed_script(
                        embed_script=args.embed_script,
                        run_dir=run_dir,
                        fid_dir=args.fid_dir,
                        emb_root=args.emb_root,
                        batch_size=args.batch_size,
                        device=args.device,
                    )
                # Compute FID/nFID
                run_compute_script(
                    compute_script=args.compute_script,
                    run_dir=run_dir,
                    emb_root=args.emb_root,
                    out_csv_name=args.per_run_csv,
                    out_parquet_name=args.per_run_parquet,
                    rr_repeats=args.rr_repeats,
                    seed=args.seed,
                    eps=args.eps,
                    neg_baselines=neg_baselines,
                    neg_percentile=args.neg_percentile,
                )
                status["computed"] = True
            else:
                status["skipped_preexisting"] = True

            # Read the run's detailed table
            df_run = read_run_fid_table(emb_dir, args.per_run_csv, args.per_run_parquet)
            # Sanity: ensure run_id column matches folder (if not present, add)
            if "run_id" not in df_run.columns:
                df_run["run_id"] = run_id
            per_run_tables.append(df_run)

        except subprocess.CalledProcessError as e:
            status["errors"].append(
                f"subprocess error (returncode={e.returncode}) during compute/embed"
            )
        except Exception as e:
            status["errors"].append(str(e))

        per_run_status[run_id] = status

    # Aggregate across runs if we have at least one table
    if not per_run_tables:
        # Save an empty aggregate and metadata with errors
        agg_df = pd.DataFrame()
        metrics = []
    else:
        agg_df, metrics = aggregate_runs(per_run_tables)

    # Write aggregated outputs
    out_csv = out_dir / args.agg_csv
    out_parquet = out_dir / args.agg_parquet
    agg_df.to_csv(out_csv, index=False)
    try:
        agg_df.to_parquet(out_parquet, index=False)
        parquet_ok = True
    except Exception as e:
        parquet_ok = False
        print(f"[warn] Failed to write Parquet at {out_parquet}: {e}")
        print("       Hint: install a Parquet engine, e.g. `pip install pyarrow`.")

    # Build and write metafile
    times = now_timestamps()
    meta = {
        "timestamps": times,
        "agg_output": {
            "folder": str(out_dir),
            "csv": str(out_csv),
            "parquet": str(out_parquet),
            "parquet_ok": parquet_ok,
        },
        "inputs": {
            "run_dirs": [str(rd.resolve()) for rd in run_dirs],
            "fid_dir": str(args.fid_dir.resolve()),
            "emb_root": str(args.emb_root.resolve()),
        },
        "per_run_status": per_run_status,
        "parameters": {
            "force": bool(args.force),
            "skip_embed": bool(args.skip_embed),
            "embed": {
                "script": str(args.embed_script.resolve()),
                "batch_size": args.batch_size,
                "device": args.device,
            },
            "compute": {
                "script": str(args.compute_script.resolve()),
                "per_run_csv": args.per_run_csv,
                "per_run_parquet": args.per_run_parquet,
                "rr_repeats": args.rr_repeats,
                "seed": args.seed,
                "eps": args.eps,
                "neg_baselines": list(neg_baselines),
                "neg_percentile": float(args.neg_percentile),
            },
            "aggregation": {
                "groupby": ["fault_code", "load"],
                "aggregated_metrics": metrics,
                "std_ddof": 1,
            },
        },
        "summary": {
            "total_runs": len(run_dirs),
            "runs_with_tables": len(per_run_tables),
            "runs_successful": sum(
                1 for s in per_run_status.values() if not s["errors"]
            ),
            "runs_failed": sum(1 for s in per_run_status.values() if s["errors"]),
        },
    }
    write_yaml(out_dir / args.agg_meta, meta)

    # Console summary (compact)
    print("\n=== Batch FID/nFID pipeline complete ===")
    print(f"Aggregated CSV    : {out_csv}")
    print(f"Aggregated Parquet: {out_parquet} ({'ok' if parquet_ok else 'failed'})")
    print(f"Metadata YAML     : {out_dir / args.agg_meta}")
    ok_runs = [rid for rid, st in per_run_status.items() if not st["errors"]]
    bad_runs = {rid: st["errors"] for rid, st in per_run_status.items() if st["errors"]}
    print(
        f"Successful runs   : {len(ok_runs)} → {', '.join(ok_runs) if ok_runs else '-'}"
    )
    if bad_runs:
        print("Failed runs:")
        for rid, errs in bad_runs.items():
            print(f"  - {rid}: {'; '.join(errs)}")


if __name__ == "__main__":
    main()
