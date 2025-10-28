"""Utility script to prepare real FFT windows for FID computation.

This script (optionally) rebuilds measurement-level metadata, regenerates (or loads)
FFT segments, filters them to the desired subset, and persists grouped windows for
later reuse. It logs progress with prints and uses tqdm progress bars when available.

Examples
--------
Run with defaults (no metadata rebuild):

    python prepare_fid_windows.py

Force metadata recreation and disable tqdm:

    python prepare_fid_windows.py --recreate-metadata --no-tqdm

Limit to a subset of engines:

    python prepare_fid_windows.py --engines engine_2 engine_5
"""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from src.data_pipeline import create_metadata_df, preprocessing, filter_segments

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ENGINE_IDS: Tuple[str, ...] = ("engine_2",)
"""Engines to process. Limited to engine_2 as per current requirements."""

STATE_MAP: Dict[int, str] = {
    1: "rotor bar defect",
    2: "normal",
    3: "bearing defect",
    4: "inter-turn short circuits",
}
"""Mapping from ordinal folder prefixes to human-readable fault labels."""

FAULT_NAME_TO_CODE: Dict[str, str] = {
    "inter-turn short circuits": "ITSC",
    "rotor bar defect": "RBD",
}
"""Subset of fault labels to keep and their short codes."""

LOADS_TO_USE: Tuple[str, ...] = ("0", "20", "40", "60", "80", "100")
SEGMENT_LENGTH = 10_000
STEP = 20
F_SAMPLING = 4_098
CUTOFF_FREQ = 250
APPLY_WINDOW = False
USE_DB = True

REPO_ROOT = Path(__file__).resolve().parents[1]
RES_ROOT = REPO_ROOT / "dataset"
FID_DIR = RES_ROOT / "fid_inputs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _progress(iterable, desc: str | None = None, total: int | None = None, enable_tqdm: bool = True):
    """Wrap an iterable with a tqdm progress bar.

    Parameters
    ----------
    iterable : Iterable
        The iterable to wrap.
    desc : str, optional
        Description to show next to the progress bar.
    total : int, optional
        Total length for the bar when it cannot be inferred.
    enable_tqdm : bool, default True
        If False or tqdm is unavailable, returns the iterable unchanged.

    Returns
    -------
    Iterable
        Either a tqdm-wrapped iterable or the original iterable.
    """
    if enable_tqdm:
        return tqdm(iterable, desc=desc, total=total)
    if desc:
        print(f"[.] {desc}...", flush=True)
    return iterable


def ensure_directory(path: Path) -> None:
    """Ensure that *path* exists.

    Parameters
    ----------
    path : pathlib.Path
        Directory path to create if missing.
    """
    path.mkdir(parents=True, exist_ok=True)


def get_or_rebuild_metadata(engine_dir: Path, recreate: bool) -> pd.DataFrame:
    """Load or (re)build measurement-level metadata for an engine directory.

    Parameters
    ----------
    engine_dir : pathlib.Path
        Root directory of the engine dataset.
    recreate : bool
        If True, metadata is rebuilt from scratch. If False, tries to load a cached
        CSV (``metadata_rebuilt.csv``) and falls back to rebuilding if not present.

    Returns
    -------
    pandas.DataFrame
        Metadata dataframe with one row per measurement.
    """
    meta_path = engine_dir / "metadata_rebuilt.csv"
    if not recreate and meta_path.exists():
        print(f"[metadata] Loading cached metadata: {meta_path}", flush=True)
        return pd.read_csv(meta_path)

    print(f"[metadata] Rebuilding metadata for {engine_dir} (recreate={recreate})", flush=True)
    t0 = time.time()
    metadata_df = create_metadata_df(str(engine_dir), state2name=STATE_MAP)
    metadata_df.to_csv(meta_path, index=False)
    print(f"[metadata] Saved to {meta_path} in {time.time() - t0:.2f}s", flush=True)
    return metadata_df


def load_or_preprocess(metadata_df: pd.DataFrame, cache_dir: Path) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Load cached FFT windows if available, otherwise regenerate them.

    Parameters
    ----------
    metadata_df : pandas.DataFrame
        Measurement-level metadata.
    cache_dir : pathlib.Path
        Directory where segments/frequencies and segment metadata are cached.

    Returns
    -------
    segments : numpy.ndarray
        Array of shape (n_segments, segment_length) with real FFT windows.
    seg_meta_df : pandas.DataFrame
        Segment-level metadata indexed by (measurement_id, segment_idx).
    freqs : numpy.ndarray
        Frequency bins corresponding to the FFT windows.
    """
    segments_path = cache_dir / "segments.npy"
    freqs_path = cache_dir / "freqs.npy"
    meta_path = cache_dir / "segments_metadata.csv"

    if segments_path.exists() and freqs_path.exists() and meta_path.exists():
        print(f"[cache] Loading segments/freqs/meta from {cache_dir}", flush=True)
        t0 = time.time()
        segments = np.load(segments_path)
        freqs = np.load(freqs_path)
        seg_meta_df = pd.read_csv(meta_path)
        seg_meta_df.set_index(["measurement_id", "segment_idx"], inplace=True)
        print(f"[cache] Loaded {segments.shape[0]:,} segments in {time.time() - t0:.2f}s", flush=True)
    else:
        print(f"[preprocess] Cache miss. Regenerating in {cache_dir}", flush=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        segments, seg_meta_df, freqs = preprocessing(
            metadata_df,
            out_dir=str(cache_dir),
            segment_length=SEGMENT_LENGTH,
            step=STEP,
            f_sampling=F_SAMPLING,
            cutoff_freq=CUTOFF_FREQ,
            apply_window=APPLY_WINDOW,
            db=USE_DB,
        )
        print(f"[preprocess] Generated {segments.shape[0]:,} segments in {time.time() - t0:.2f}s", flush=True)
    return segments, seg_meta_df, freqs


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main(
    engines: Iterable[str] = ENGINE_IDS,
    recreate_metadata: bool = False,
    enable_tqdm: bool = True,
) -> None:
    """End-to-end preparation of real FFT windows for FID.

    Parameters
    ----------
    engines : Iterable[str], default ENGINE_IDS
        Engine identifiers to process.
    recreate_metadata : bool, default False
        If True, rebuild measurement-level metadata even if a cached CSV exists.
    enable_tqdm : bool, default True
        If True and ``tqdm`` is available, show progress bars.

    Notes
    -----
    The function writes filtered segment groups into ``RES_ROOT / 'fid_inputs'`` and
    records an ``index_map.json`` to map group keys to row indices in the filtered
    metadata.
    """
    ensure_directory(FID_DIR)
    index_cache: Dict[str, List[int]] = {}

    # Make engines indexable & sized for tqdm
    engines_list = list(engines)
    print(f"[start] Preparing FID inputs for engines: {engines_list}", flush=True)

    for engine_id in _progress(engines_list, desc="Engines", total=len(engines_list), enable_tqdm=enable_tqdm):
        engine_dir = REPO_ROOT / "dataset" / engine_id
        if not engine_dir.exists():
            raise FileNotFoundError(f"Engine directory not found: {engine_dir}")

        print(f"\n[engine] {engine_id} @ {engine_dir}", flush=True)
        metadata_df = get_or_rebuild_metadata(engine_dir, recreate=recreate_metadata)

        cache_dir = RES_ROOT / engine_id / "current_fft"
        segments, seg_meta_df, freqs = load_or_preprocess(metadata_df, cache_dir)

        # Harmonise column types before filtering.
        seg_meta_df = seg_meta_df.copy()
        seg_meta_df["phase"] = seg_meta_df["phase"].astype(str)
        seg_meta_df["load_condition"] = seg_meta_df["load_condition"].astype(str)

        phases_to_use = sorted(seg_meta_df["phase"].unique())
        target_faults = list(FAULT_NAME_TO_CODE.keys())

        print(
            f"[filter] phases={phases_to_use} | faults={target_faults} | loads={list(LOADS_TO_USE)}",
            flush=True,
        )

        t0 = time.time()
        filtered_meta, filtered_segments = filter_segments(
            seg_meta_df,
            segments,
            training_classes=target_faults,
            loads_to_use=list(LOADS_TO_USE),
            phases_to_use=phases_to_use,
        )
        print(
            f"[filter] kept {filtered_segments.shape[0]:,} / {segments.shape[0]:,} segments "
            f"in {time.time() - t0:.2f}s",
            flush=True,
        )

        # Persist filtered metadata for transparency.
        filtered_meta_path = FID_DIR / f"{engine_id}_filtered_segments_metadata.csv"
        filtered_meta.to_csv(filtered_meta_path, index=False)
        print(f"[save] filtered segment metadata -> {filtered_meta_path}", flush=True)

        # Prepare grouped outputs: per (load, fault_code)
        segment_length = (
            filtered_segments.shape[1] if filtered_segments.size else SEGMENT_LENGTH
        )
        combos = [(fn, fc, ld) for fn, fc in FAULT_NAME_TO_CODE.items() for ld in LOADS_TO_USE]

        saved_files = 0
        for fault_name, fault_code, load in _progress(
            combos,
            desc=f"{engine_id}: group & save",
            total=len(combos),
            enable_tqdm=enable_tqdm,
        ):
            fault_mask = filtered_meta["state"] == fault_name
            load_mask = filtered_meta["load_condition"] == load
            mask = fault_mask & load_mask
            idx = np.flatnonzero(mask.to_numpy())

            key = f"{load}_{fault_code}"
            index_cache[key] = idx.tolist()

            grouped_segments = (
                filtered_segments[idx]
                if idx.size
                else np.empty((0, segment_length), dtype=filtered_segments.dtype)
            )

            out_path = FID_DIR / f"real_{load}_{fault_code}.npy"
            np.save(out_path, grouped_segments)
            saved_files += 1

            print(
                f"[save] {out_path.name}: segments={grouped_segments.shape[0]:,} "
                f"(load={load}, fault={fault_code})",
                flush=True,
            )

        # Frequencies per engine (if they could differ by preprocessing params)
        freqs_path = FID_DIR / f"{engine_id}_freqs.npy"
        np.save(freqs_path, freqs)
        print(f"[save] freqs -> {freqs_path} | saved files: {saved_files}", flush=True)

    # Persist the global index map
    index_path = FID_DIR / "index_map.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index_cache, f, indent=2)
    print(f"\n[done] index map -> {index_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : Iterable[str], optional
        Arguments to parse (for testing). If None, uses ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Prepare real FFT windows for FID computation.")
    parser.add_argument(
        "--recreate-metadata",
        action="store_true",
        default=False,
        help="Rebuild measurement-level metadata even if a cached CSV exists (default: False).",
    )
    parser.add_argument(
        "--no-tqdm",
        action="store_true",
        default=False,
        help="Disable tqdm progress bars (prints remain).",
    )
    parser.add_argument(
        "--engines",
        nargs="*",
        default=list(ENGINE_IDS),
        help=f"Engines to process (default: {list(ENGINE_IDS)}).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


if __name__ == "__main__":
    args = _parse_args()
    main(
        engines=args.engines,
        recreate_metadata=args.recreate_metadata,
        enable_tqdm=not args.no_tqdm,
    )
