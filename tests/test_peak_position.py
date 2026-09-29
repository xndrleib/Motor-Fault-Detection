import hashlib
import random

import numpy as np
import pytest

from src.anomaly_injector import DiagnosticGaussianPeakInjector, GaussianPeakInjector


def fitted_center(freqs, delta):
    """Measure the Gaussian centre from samples, independently of its generator."""
    indices = np.argsort(np.abs(delta))[-3:]
    assert np.all(np.abs(delta[indices]) > 0)
    origin = freqs[np.argmax(np.abs(delta))]
    a, b, _ = np.polyfit(freqs[indices] - origin, np.log(np.abs(delta[indices])), 2)
    assert a < 0
    return origin - b / (2 * a)


@pytest.mark.parametrize("frequency", [0.0, 26.83, 50.0, 126.83, 249.978])
@pytest.mark.parametrize("sigma", [0.5, 1.5])
def test_continuous_center_measured_from_float32_samples(frequency, sigma):
    freqs = np.arange(611) * 0.4098
    source = np.linspace(-65, 50, len(freqs), dtype=np.float32)
    random.seed(17)
    gen = DiagnosticGaussianPeakInjector(4, (0.5, 0.5), (sigma, sigma), True)
    result = gen.inject(source, freqs, [frequency])
    delta = result.astype(float) - source.astype(float)
    assert abs(fitted_center(freqs, delta) - frequency) < 0.08
    outside = np.abs(freqs - frequency) > 4
    assert np.array_equal(result[outside], source[outside])
    assert result.dtype == source.dtype


def test_repeatability_and_superposition():
    freqs = np.arange(611) * 0.4098
    x = np.linspace(-20, 30, len(freqs), dtype=np.float32)
    gen = DiagnosticGaussianPeakInjector(4, negative=True)
    random.seed(42)
    first = gen.inject(x, freqs, [26.83, 28.1, 35.4])
    random.seed(42)
    second = x.copy()
    for f in [26.83, 28.1, 35.4]:
        second = gen.inject(second, freqs, [f])
    assert np.array_equal(first, second)
    random.seed(42)
    assert np.array_equal(first, gen.inject(x, freqs, [26.83, 28.1, 35.4]))


def test_training_injector_preserves_frozen_output():
    random.seed(42)
    freqs = np.arange(611) * 0.4098
    x = np.linspace(-20, 30, len(freqs), dtype=np.float32)
    result = GaussianPeakInjector(10, (0.5, 20), (0.5, 1.5), True, True).inject(
        x, freqs, [26.83, 50.0, 73.17, 126.83, 150.0, 173.17])
    assert hashlib.sha256(result.tobytes()).hexdigest() == (
        "eedc14986fa476d6cf39e266e7b3714bf312c9be69526c69b3fbef89dcfdc2fa")


def test_invalid_frequency_grid_and_ranges_are_rejected():
    gen = DiagnosticGaussianPeakInjector()
    for freqs, target in [([0.0, 1.0, 3.0], 1), ([0.0, 1.0, 2.0], 3),
                          ([0.0, 1.0, 2.0], float("nan"))]:
        with pytest.raises(ValueError):
            gen.inject(np.zeros(3, dtype=np.float32), freqs, [target])
    with pytest.raises(ValueError):
        DiagnosticGaussianPeakInjector(amplitude_range=(0, 1))
    with pytest.raises(ValueError):
        DiagnosticGaussianPeakInjector(sigma_range=(1, 0.5))
