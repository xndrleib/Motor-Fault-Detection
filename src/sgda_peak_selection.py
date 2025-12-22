# sgda_peak_selection.py
"""Peak selection helpers for SGDA injections."""

from __future__ import annotations

import logging
from typing import Dict, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_PEAK_SEED_OFFSET = 12345


def resolve_peak_location_seed(
    train_seed: int | None,
    peak_location_seed: int | None,
    *,
    offset: int = DEFAULT_PEAK_SEED_OFFSET,
) -> int | None:
    """Resolve the RNG seed used for random peak-location selection.

    Parameters
    ----------
    train_seed
        Global training seed. Used as the base when ``peak_location_seed`` is
        not provided.
    peak_location_seed
        Explicit seed for peak selection. If provided, it takes precedence.
    offset
        Offset added to ``train_seed`` when deriving the peak seed.

    Returns
    -------
    int or None
        The resolved seed, or ``None`` if no seed can be determined.
    """
    if peak_location_seed is not None:
        return int(peak_location_seed)
    if train_seed is None:
        return None
    return int(train_seed) + int(offset)


def select_peak_frequencies(
    fault_freqs_physics: Mapping[str, Sequence[float]],
    fft_freqs: np.ndarray,
    peak_mode: str,
    *,
    rng_seed: int | None = None,
    margin_bins: int = 0,
) -> Dict[str, np.ndarray]:
    """Select per-fault peak centers for SGDA injection.

    Parameters
    ----------
    fault_freqs_physics
        Mapping from fault name to physics-guided frequency list (MCSA).
    fft_freqs
        1D array of FFT frequency bins.
    peak_mode
        Peak selection mode. Must be ``"mcsa"`` or ``"random"``.
    rng_seed
        Seed for random peak selection. Only used when ``peak_mode="random"``.
    margin_bins
        Minimum number of bins to keep away from FFT boundaries. This avoids
        truncated peak windows at the edges.

    Returns
    -------
    dict
        Mapping from fault name to selected peak frequencies (float arrays).

    Raises
    ------
    ValueError
        If ``peak_mode`` is not a supported option.
    """
    mode = str(peak_mode).lower().strip()
    freqs = np.asarray(fft_freqs, dtype=float)
    if freqs.ndim != 1:
        raise ValueError("fft_freqs must be a 1D array of frequency bins.")

    if mode == "mcsa":
        return {
            fault: np.asarray(freqs_list, dtype=float)
            for fault, freqs_list in fault_freqs_physics.items()
        }
    if mode != "random":
        raise ValueError(f"Unsupported peak_mode: {peak_mode!r}")

    if rng_seed is None:
        logger.warning(
            "Random peak mode requested without rng_seed; results will be non-deterministic."
        )
    rng = np.random.default_rng(rng_seed)

    n_bins = freqs.size
    margin = max(0, int(margin_bins))
    if n_bins == 0:
        raise ValueError("fft_freqs is empty; cannot select random peak locations.")

    if n_bins - 2 * margin <= 0:
        logger.warning(
            "margin_bins=%d leaves no interior bins; falling back to full range.",
            margin,
        )
        valid_idx = np.arange(n_bins)
    else:
        valid_idx = np.arange(margin, n_bins - margin)

    out: Dict[str, np.ndarray] = {}
    for fault, freqs_list in fault_freqs_physics.items():
        n_peaks = len(freqs_list)
        if n_peaks == 0:
            out[fault] = np.asarray([], dtype=float)
            continue

        replace = n_peaks > valid_idx.size
        if replace:
            logger.warning(
                "Requested %d random peaks for '%s' with only %d valid bins; "
                "sampling with replacement.",
                n_peaks,
                fault,
                valid_idx.size,
            )

        chosen_idx = rng.choice(valid_idx, size=n_peaks, replace=replace)
        chosen_freqs = freqs[chosen_idx]
        out[fault] = np.sort(chosen_freqs.astype(float))

    return out
