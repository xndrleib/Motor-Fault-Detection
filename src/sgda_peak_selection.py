# sgda_peak_selection.py
"""Peak selection helpers for SGDA injections."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_PEAK_SEED_OFFSET = 12345


def _resolve_valid_indices(
    freqs: np.ndarray, margin_bins: int, *, label: str | None = None
) -> np.ndarray:
    """Return valid FFT indices after applying a margin.

    Parameters
    ----------
    freqs
        1D array of FFT frequency bins.
    margin_bins
        Number of bins to exclude on either side.
    label
        Optional label to include in warning messages.

    Returns
    -------
    np.ndarray
        1D array of valid indices for sampling.
    """
    n_bins = freqs.size
    if n_bins == 0:
        raise ValueError("fft_freqs is empty; cannot select random peak locations.")

    margin = max(0, int(margin_bins))
    if n_bins - 2 * margin <= 0:
        if label:
            logger.warning(
                "margin_bins=%d leaves no interior bins for %s; falling back to full range.",
                margin,
                label,
            )
        else:
            logger.warning(
                "margin_bins=%d leaves no interior bins; falling back to full range.",
                margin,
            )
        return np.arange(n_bins)
    return np.arange(margin, n_bins - margin)


@dataclass
class RandomPeakSampler:
    """Callable sampler that draws random peak centers per injection.

    Parameters
    ----------
    peak_count
        Fixed number of peaks to draw per call. Ignored if ``count_range`` is set.
    count_range
        Optional (min, max) range for the number of peaks to draw per call.
    rng
        NumPy random generator to use for sampling.
    margin_bins
        Minimum number of bins to keep away from FFT boundaries.
    label
        Optional label used in warning messages.
    """

    peak_count: int | None
    count_range: tuple[int, int] | None
    rng: np.random.Generator
    margin_bins: int = 0
    label: str | None = None
    _cached_len: int | None = field(default=None, init=False, repr=False)
    _cached_valid_idx: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.peak_count is not None and self.peak_count < 0:
            raise ValueError("peak_count must be >= 0.")
        if self.count_range is not None:
            if len(self.count_range) != 2:
                raise ValueError("count_range must be a (min, max) pair when provided.")
            min_peaks, max_peaks = (int(self.count_range[0]), int(self.count_range[1]))
            if min_peaks < 1 or max_peaks < min_peaks:
                raise ValueError("count_range must be >= 1 and min <= max.")
            self.count_range = (min_peaks, max_peaks)

    @property
    def peak_count_range(self) -> tuple[int, int] | None:
        """Return the configured per-call peak count range, if any."""
        return self.count_range

    def __call__(self, fft_freqs: np.ndarray) -> np.ndarray:
        """Sample random peak centers for a single injection."""
        freqs = np.asarray(fft_freqs, dtype=float)
        if freqs.ndim != 1:
            raise ValueError("fft_freqs must be a 1D array of frequency bins.")

        valid_idx = self._get_valid_indices(freqs)
        if self.count_range is not None:
            n_peaks = int(
                self.rng.integers(self.count_range[0], self.count_range[1] + 1)
            )
        else:
            n_peaks = int(self.peak_count or 0)

        if n_peaks == 0:
            return np.asarray([], dtype=float)

        replace = n_peaks > valid_idx.size
        if replace:
            logger.warning(
                "Requested %d random peaks for '%s' with only %d valid bins; "
                "sampling with replacement.",
                n_peaks,
                self.label or "fault",
                valid_idx.size,
            )

        chosen_idx = self.rng.choice(valid_idx, size=n_peaks, replace=replace)
        chosen_freqs = freqs[chosen_idx]
        return np.sort(chosen_freqs.astype(float))

    def _get_valid_indices(self, freqs: np.ndarray) -> np.ndarray:
        n_bins = freqs.size
        if self._cached_len == n_bins and self._cached_valid_idx is not None:
            return self._cached_valid_idx
        valid_idx = _resolve_valid_indices(freqs, self.margin_bins, label=self.label)
        self._cached_len = n_bins
        self._cached_valid_idx = valid_idx
        return valid_idx


def build_random_peak_samplers(
    fault_freqs_physics: Mapping[str, Sequence[float]],
    *,
    rng_seed: int | None,
    margin_bins: int = 0,
    random_peak_count_range: Sequence[int] | None = None,
) -> Dict[str, RandomPeakSampler]:
    """Create per-fault random peak samplers for per-sample injections.

    Parameters
    ----------
    fault_freqs_physics
        Mapping from fault name to physics-guided frequency list (MCSA).
    rng_seed
        Seed for the base RNG used to spawn per-fault generators.
    margin_bins
        Minimum number of bins to keep away from FFT boundaries.
    random_peak_count_range
        Optional (min, max) range for the number of random peaks per fault.

    Returns
    -------
    dict
        Mapping from fault name to ``RandomPeakSampler`` instances.
    """
    if rng_seed is None:
        logger.warning(
            "Random peak samplers requested without rng_seed; results will be non-deterministic."
        )
        seed_seq = np.random.SeedSequence()
    else:
        seed_seq = np.random.SeedSequence(int(rng_seed))

    fault_items = list(fault_freqs_physics.items())
    child_seqs = seed_seq.spawn(len(fault_items))

    out: Dict[str, RandomPeakSampler] = {}
    for (fault, freqs_list), child_seq in zip(fault_items, child_seqs):
        peak_count = None if random_peak_count_range else len(freqs_list)
        sampler = RandomPeakSampler(
            peak_count=peak_count,
            count_range=(
                tuple(random_peak_count_range) if random_peak_count_range else None
            ),
            rng=np.random.default_rng(child_seq),
            margin_bins=margin_bins,
            label=fault,
        )
        out[fault] = sampler
    return out


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
    random_peak_count_range: Sequence[int] | None = None,
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
    random_peak_count_range
        Optional (min, max) range for the number of random peaks per fault.
        Only used when ``peak_mode="random"``. If provided, each fault's peak
        count is sampled uniformly from this inclusive range.

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

    if random_peak_count_range is not None:
        if len(random_peak_count_range) != 2:
            raise ValueError(
                "random_peak_count_range must be a (min, max) pair when provided."
            )
        min_peaks, max_peaks = (int(random_peak_count_range[0]), int(random_peak_count_range[1]))
        if min_peaks < 1 or max_peaks < min_peaks:
            raise ValueError(
                "random_peak_count_range must be >= 1 and min <= max."
            )
    else:
        min_peaks = max_peaks = None

    if rng_seed is None:
        logger.warning(
            "Random peak mode requested without rng_seed; results will be non-deterministic."
        )
    rng = np.random.default_rng(rng_seed)

    valid_idx = _resolve_valid_indices(freqs, margin_bins)

    out: Dict[str, np.ndarray] = {}
    for fault, freqs_list in fault_freqs_physics.items():
        if min_peaks is not None and max_peaks is not None:
            n_peaks = int(rng.integers(min_peaks, max_peaks + 1))
        else:
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
