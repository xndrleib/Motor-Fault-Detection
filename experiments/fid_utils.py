"""Utilities for FID window preparation."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from src.anomaly_injector import CompositeAnomalyInjector


def generate_sgda_windows(
    normal_segments: np.ndarray,
    normal_meta: pd.DataFrame,
    freqs: np.ndarray,
    fault_freqs: Mapping[str, Sequence[float]],
    counts: Mapping[Tuple[str, str], int],
    injector: CompositeAnomalyInjector,
    *,
    rng_seed: int | None = None,
    injector_keys: Sequence[str] | None = ("peak-anomaly",),
    load_column: str = "load_condition",
) -> Tuple[Dict[Tuple[str, str], np.ndarray], Dict[Tuple[str, str], np.ndarray]]:
    """Generate SGDA windows that mirror the real fault counts.

    Parameters
    ----------
    normal_segments
        Array of normal FFT segments. The ``i``\ th row must correspond to
        ``normal_meta.iloc[i]``.
    normal_meta
        Segment-level metadata for ``normal_segments``. Must include the
        ``load_column``.
    freqs
        Frequency bins for the FFT segments.
    fault_freqs
        Mapping from fault name to diagnostic frequency bands.
    counts
        Target counts per ``(load, fault)`` bucket. The helper draws this many
        normal windows for every bucket and applies SGDA.
    injector
        Preconfigured :class:`~src.anomaly_injector.CompositeAnomalyInjector`.
    rng_seed
        Optional seed for reproducible sampling.
    injector_keys
        Injector components to apply. Defaults to only the ``"peak-anomaly"``
        branch.
    load_column
        Name of the column in ``normal_meta`` that stores the load identifier.

    Returns
    -------
    synth_windows, selection_indices
        Two dictionaries keyed by ``(load, fault)``. ``synth_windows`` contains
        the generated synthetic segments. ``selection_indices`` stores the
        sampled row indices (relative to ``normal_meta``) used for each bucket.
    """

    if len(normal_meta) != normal_segments.shape[0]:
        raise ValueError(
            "normal_segments and normal_meta must reference the same number of rows"
        )

    rng = np.random.default_rng(rng_seed)
    injector_keys_list: list[str] | None
    if injector_keys is None:
        injector_keys_list = None
    else:
        injector_keys_list = list(injector_keys)

    synth_windows: Dict[Tuple[str, str], np.ndarray] = {}
    selection_indices: Dict[Tuple[str, str], np.ndarray] = {}

    load_series = normal_meta[load_column].astype(str)

    for key, target_count in counts.items():
        load, fault_name = key
        target_count = int(target_count)
        fault_key = str(fault_name)
        load_key = str(load)

        if fault_key not in fault_freqs:
            raise KeyError(f"Missing fault frequencies for '{fault_key}'")

        if target_count <= 0:
            synth_windows[key] = np.empty(
                (0, normal_segments.shape[1]), dtype=normal_segments.dtype
            )
            selection_indices[key] = np.empty((0,), dtype=int)
            continue

        load_mask = load_series == load_key
        candidate_indices = np.flatnonzero(load_mask.to_numpy())
        if candidate_indices.size == 0:
            raise ValueError(
                f"No normal segments available for load '{load_key}' to synthesise fault '{fault_key}'"
            )

        replace = candidate_indices.size < target_count
        chosen_indices = rng.choice(
            candidate_indices, size=target_count, replace=replace
        )

        base_segments = normal_segments[chosen_indices]
        freq_array = np.asarray(fault_freqs[fault_key], dtype=float)
        if freq_array.size == 0:
            synth_windows[key] = np.empty(
                (0, normal_segments.shape[1]), dtype=normal_segments.dtype
            )
            selection_indices[key] = chosen_indices.astype(int)
            continue

        augmented = [
            injector.inject(
                segment=segment,
                fft_freqs=freqs,
                fault_freqs=freq_array,
                injector_keys=injector_keys_list,
            )
            for segment in base_segments
        ]
        synth_windows[key] = np.asarray(augmented, dtype=normal_segments.dtype)
        selection_indices[key] = chosen_indices.astype(int)

    return synth_windows, selection_indices
