"""Tests for anomaly injectors."""

import numpy as np

from src.anomaly_injector import GaussianPeakInjector


def test_gaussian_injector_callable_fault_freqs() -> None:
    fft_freqs = np.linspace(0.0, 10.0, 11)
    segment = np.zeros_like(fft_freqs)
    injector = GaussianPeakInjector(
        peak_segment=1,
        amplitude_range=(1.0, 1.0),
        sigma_range=(1.0, 1.0),
        negative=False,
        random_peak_position=False,
    )

    calls = {"count": 0}

    def sampler(freqs: np.ndarray) -> np.ndarray:
        calls["count"] += 1
        return np.array([freqs[5]])

    out = injector.inject(segment, fft_freqs, sampler)
    assert calls["count"] == 1
    assert np.any(out != 0.0)
