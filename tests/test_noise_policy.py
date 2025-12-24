"""Tests for noise policy utilities."""

import numpy as np

from src.noise_policy import NoisePolicy, SNRStage


def test_noise_policy_uniform_range() -> None:
    policy = NoisePolicy(p_clean=0.0, snr_min=10.0, snr_max=20.0)
    rng = np.random.default_rng(123)
    signal = np.ones(1000, dtype=float)
    noisy, snr_db = policy.apply(signal, rng=rng, epoch=1, total_epochs=10)
    assert snr_db is not None
    assert 10.0 <= snr_db <= 20.0
    assert noisy.shape == signal.shape


def test_noise_policy_curriculum_bounds() -> None:
    curriculum = [
        SNRStage(start_pct=0.0, end_pct=0.5, snr_min=30.0, snr_max=40.0),
        SNRStage(start_pct=0.5, end_pct=1.0, snr_min=15.0, snr_max=25.0),
    ]
    policy = NoisePolicy(p_clean=0.0, snr_min=10.0, snr_max=20.0, curriculum=curriculum)
    rng = np.random.default_rng(7)
    signal = np.ones(1000, dtype=float)

    _, snr_db_early = policy.apply(signal, rng=rng, epoch=1, total_epochs=10)
    assert snr_db_early is not None
    assert 30.0 <= snr_db_early <= 40.0

    _, snr_db_late = policy.apply(signal, rng=rng, epoch=9, total_epochs=10)
    assert snr_db_late is not None
    assert 15.0 <= snr_db_late <= 25.0
