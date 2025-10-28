"""Utility script to prepare real FFT windows for FID computation.

This script rebuilds measurement-level metadata, regenerates (or loads)
FFT segments, filters them to the desired subset, and persists grouped
windows for later reuse.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from src.data_pipeline import create_metadata_df, preprocessing, filter_segments

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ENGINE_IDS: Tuple[str, ...] = ("engine_2",)
"""Engines to process.  Limited to engine_2 as per current requirements."""

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
RES_ROOT = REPO_ROOT / "res"
FID_DIR = RES_ROOT / "fid_inputs"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def rebuild_metadata(engine_dir: Path) -> pd.DataFrame:
    """Recreate the measurement-level metadata for *engine_dir*."""
    metadata_df = create_metadata_df(str(engine_dir), state2name=STATE_MAP)
    out_path = engine_dir / "metadata_rebuilt.csv"
    metadata_df.to_csv(out_path)
    return metadata_df


def load_or_preprocess(metadata_df: pd.DataFrame, cache_dir: Path) -> tuple:
    """Load cached FFT windows if available, otherwise regenerate them."""
    segments_path = cache_dir / "segments.npy"
    freqs_path = cache_dir / "freqs.npy"
    meta_path = cache_dir / "segments_metadata.csv"

    if segments_path.exists() and freqs_path.exists() and meta_path.exists():
        segments = np.load(segments_path)
        freqs = np.load(freqs_path)
        seg_meta_df = pd.read_csv(meta_path)
        seg_meta_df.set_index(["measurement_id", "segment_idx"], inplace=True)
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
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
    return segments, seg_meta_df, freqs


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main(engines: Iterable[str] = ENGINE_IDS) -> None:
    ensure_directory(FID_DIR)
    index_cache: Dict[str, Iterable[int]] = {}

    for engine_id in engines:
        engine_dir = REPO_ROOT / "dataset" / engine_id
        if not engine_dir.exists():
            raise FileNotFoundError(f"Engine directory not found: {engine_dir}")

        metadata_df = rebuild_metadata(engine_dir)
        cache_dir = RES_ROOT / engine_id / "current_fft"
        segments, seg_meta_df, freqs = load_or_preprocess(metadata_df, cache_dir)

        # Harmonise column types before filtering.
        seg_meta_df = seg_meta_df.copy()
        seg_meta_df["phase"] = seg_meta_df["phase"].astype(str)
        seg_meta_df["load_condition"] = seg_meta_df["load_condition"].astype(str)

        phases_to_use = sorted(seg_meta_df["phase"].unique())
        target_faults = list(FAULT_NAME_TO_CODE.keys())

        filtered_meta, filtered_segments = filter_segments(
            seg_meta_df,
            segments,
            training_classes=target_faults,
            loads_to_use=list(LOADS_TO_USE),
            phases_to_use=phases_to_use,
        )

        filtered_meta_path = FID_DIR / f"{engine_id}_filtered_segments_metadata.csv"
        filtered_meta.to_csv(filtered_meta_path, index=False)

        segment_length = filtered_segments.shape[1] if filtered_segments.size else SEGMENT_LENGTH

        for fault_name, fault_code in FAULT_NAME_TO_CODE.items():
            fault_mask = filtered_meta["state"] == fault_name
            for load in LOADS_TO_USE:
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

        freqs_path = FID_DIR / f"{engine_id}_freqs.npy"
        np.save(freqs_path, freqs)

    index_path = FID_DIR / "index_map.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index_cache, f, indent=2)


if __name__ == "__main__":
    main()
