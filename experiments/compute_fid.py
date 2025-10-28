"""Compute FID scores between real and synthetic embeddings for each load/fault pair.

This script expects the real and synthetic embeddings to be organised in a directory
hierarchy of the form::

    <root>/<load>/<fault>/*.npy

All embedding files found below each ``<fault>`` directory will be concatenated along
the first axis. Supported file formats are ``.npy``, ``.npz`` (first array) and
``.csv``. For every pair that exists in both real and synthetic roots, the script runs
bootstrap resampling to estimate the distribution of the Fréchet Inception Distance
and stores the resulting median and 95% confidence interval in
``res/fid_results/fid_matrix.csv`` (or the path provided via ``--output``).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple, Optional

import numpy as np
import pandas as pd

from src.evaluation import bootstrap_fid

ALLOWED_SUFFIXES = {".npy", ".npz", ".csv"}


def _load_embedding_file(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        data = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as archive:
            if len(archive.files) == 0:
                raise ValueError(f"No arrays stored in npz file: {path}")
            key = "arr_0" if "arr_0" in archive.files else archive.files[0]
            data = archive[key]
    elif suffix == ".csv":
        df = pd.read_csv(path)
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise ValueError(f"No numeric columns found in CSV: {path}")
        data = numeric_df.to_numpy()
    else:
        raise ValueError(f"Unsupported embedding file type: {path.suffix}")

    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2:
        raise ValueError(f"Embedding matrix must be 2D: {path}")
    return data


def _iter_embedding_files(root: Path) -> Iterable[Tuple[Tuple[str, str], Path]]:
    """Yield ``((load, fault), file_path)`` pairs for every embedding file under root."""

    for load_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for fault_dir in sorted(p for p in load_dir.iterdir() if p.is_dir()):
            for file in sorted(fault_dir.rglob("*")):
                if file.is_file() and file.suffix.lower() in ALLOWED_SUFFIXES:
                    yield (load_dir.name, fault_dir.name), file


def _load_embeddings(root: Path) -> Dict[Tuple[str, str], np.ndarray]:
    grouped: Dict[Tuple[str, str], list[np.ndarray]] = {}
    for key, file in _iter_embedding_files(root):
        grouped.setdefault(key, []).append(_load_embedding_file(file))

    embeddings: Dict[Tuple[str, str], np.ndarray] = {}
    for key, mats in grouped.items():
        feature_dims = {m.shape[1] for m in mats}
        if len(feature_dims) != 1:
            raise ValueError(
                f"Inconsistent feature dimensions for pair {key}: {sorted(feature_dims)}"
            )
        embeddings[key] = np.vstack(mats)

    return embeddings


def compute_fid_matrix(real_root: Path,
                       synthetic_root: Path,
                       n_bootstrap: int,
                       seed: Optional[int]) -> pd.DataFrame:
    real_embeddings = _load_embeddings(real_root)
    synthetic_embeddings = _load_embeddings(synthetic_root)

    common_pairs = sorted(set(real_embeddings) & set(synthetic_embeddings))
    if not common_pairs:
        raise ValueError(
            "No matching (load, fault) pairs were found between the provided directories."
        )

    records = []
    for load, fault in common_pairs:
        real = real_embeddings[(load, fault)]
        synth = synthetic_embeddings[(load, fault)]

        if real.shape[1] != synth.shape[1]:
            raise ValueError(
                f"Feature mismatch for pair (load={load}, fault={fault}):"
                f" {real.shape[1]} != {synth.shape[1]}"
            )

        scores = bootstrap_fid(real, synth, n_bootstrap=n_bootstrap, random_state=seed)
        median = float(np.median(scores))
        ci_low, ci_high = np.percentile(scores, [2.5, 97.5])

        records.append({
            "load": load,
            "fault": fault,
            "fid": median,
            "fid_ci_low": float(ci_low),
            "fid_ci_high": float(ci_high),
        })

    df = pd.DataFrame.from_records(records)
    return df.sort_values(["load", "fault"], ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-root", type=Path, required=True,
                        help="Directory with real embeddings organised by load/fault.")
    parser.add_argument("--synthetic-root", type=Path, required=True,
                        help="Directory with synthetic embeddings organised by load/fault.")
    parser.add_argument("--output", type=Path,
                        default=Path("res") / "fid_results" / "fid_matrix.csv",
                        help="Where to store the resulting CSV file.")
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="Number of bootstrap iterations (default: 1000).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed used for bootstrap resampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.bootstrap <= 0:
        raise ValueError("--bootstrap must be a positive integer")

    if not args.real_root.is_dir():
        raise FileNotFoundError(f"Real embeddings directory not found: {args.real_root}")
    if not args.synthetic_root.is_dir():
        raise FileNotFoundError(f"Synthetic embeddings directory not found: {args.synthetic_root}")

    df = compute_fid_matrix(
        real_root=args.real_root,
        synthetic_root=args.synthetic_root,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved FID matrix to {output_path.resolve()}")


if __name__ == "__main__":
    main()

