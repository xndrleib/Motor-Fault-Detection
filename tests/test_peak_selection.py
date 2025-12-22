"""Tests for SGDA peak selection helpers."""

import random

import numpy as np

from src.sgda_peak_selection import (
    DEFAULT_PEAK_SEED_OFFSET,
    resolve_peak_location_seed,
    select_peak_frequencies,
)


def test_resolve_peak_location_seed() -> None:
    assert resolve_peak_location_seed(10, 99) == 99
    assert resolve_peak_location_seed(10, None) == 10 + DEFAULT_PEAK_SEED_OFFSET
    assert resolve_peak_location_seed(None, None) is None


def test_select_peak_frequencies_random_deterministic() -> None:
    fft_freqs = np.arange(0.0, 100.0, 1.0)
    fault_freqs = {"fault_a": [10.0, 20.0], "fault_b": [30.0]}

    out1 = select_peak_frequencies(
        fault_freqs, fft_freqs, "random", rng_seed=123, margin_bins=3
    )
    out2 = select_peak_frequencies(
        fault_freqs, fft_freqs, "random", rng_seed=123, margin_bins=3
    )

    assert np.array_equal(out1["fault_a"], out2["fault_a"])
    assert np.array_equal(out1["fault_b"], out2["fault_b"])
    assert len(out1["fault_a"]) == len(fault_freqs["fault_a"])
    assert len(out1["fault_b"]) == len(fault_freqs["fault_b"])


def test_select_peak_frequencies_random_margin() -> None:
    fft_freqs = np.arange(0.0, 50.0, 1.0)
    fault_freqs = {"fault_a": [1.0, 2.0, 3.0]}
    margin = 5

    out = select_peak_frequencies(
        fault_freqs, fft_freqs, "random", rng_seed=7, margin_bins=margin
    )
    assert out["fault_a"].min() >= fft_freqs[margin]
    assert out["fault_a"].max() <= fft_freqs[-margin - 1]


def test_select_peak_frequencies_mcsa_passthrough() -> None:
    fft_freqs = np.arange(0.0, 10.0, 1.0)
    fault_freqs = {"fault_a": [1.5, 2.5]}
    out = select_peak_frequencies(fault_freqs, fft_freqs, "mcsa", rng_seed=1)
    assert np.allclose(out["fault_a"], np.asarray(fault_freqs["fault_a"], dtype=float))


def test_no_rng_bleed() -> None:
    fft_freqs = np.arange(0.0, 100.0, 1.0)
    fault_freqs = {"fault_a": [10.0, 20.0]}

    random_state = random.getstate()
    np_state = np.random.get_state()

    _ = select_peak_frequencies(
        fault_freqs, fft_freqs, "random", rng_seed=123, margin_bins=3
    )

    assert random.getstate() == random_state
    assert np.random.get_state()[1].tolist() == np_state[1].tolist()
