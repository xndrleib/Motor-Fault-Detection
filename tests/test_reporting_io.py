import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from reporting.input_files import read_current_signal
from reporting.prediction_export import probabilities_from_logits
from reporting.core import prepare


def test_softmax_scores_are_finite_and_normalized_for_large_logits():
    logits = np.array([[1000, 1000], [-1000, 1000], [0, np.log(3)]])
    got = probabilities_from_logits(logits)
    assert np.isfinite(got).all()
    assert np.allclose(got.sum(axis=1), 1)
    assert np.allclose(got, [[0.5, 0.5], [0, 1], [0.25, 0.75]])
    assert np.array_equal(got.argmax(axis=1), logits.argmax(axis=1))


def test_time_axis_rejects_wrong_rate_gaps_and_wrong_units(tmp_path):
    path = tmp_path / "signal.csv"
    t = np.arange(10000) / 4098
    current = np.sin(2*np.pi*50*t)
    for times in [t*2, t*1000, np.r_[t[:5000], t[5000:]+1/4098]]:
        pd.DataFrame({"Time": times, "Current": current}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="Частота|Неравномерная"):
            read_current_signal(path, sampling_hz=4098)
    # An explicit unit declaration supports otherwise identical millisecond data.
    pd.DataFrame({"Time": t*1000, "Current": current}).to_csv(path, index=False)
    signal = read_current_signal(path, sampling_hz=4098, time_axis="milliseconds")
    assert signal.time_axis["observed_sampling_hz"] == pytest.approx(4098)


def test_engine2_time_axis_preserves_historical_analysis_frequency(tmp_path):
    path = tmp_path / "historical.csv"
    values = np.linspace(-1, 1, 16384)
    pd.DataFrame({"0": np.arange(16384)*1000/4096, "1": values}).to_csv(path)
    with pytest.raises(ValueError, match="Частота"):
        read_current_signal(path, sampling_hz=4098, time_axis="milliseconds")
    got = read_current_signal(path, sampling_hz=4098, time_axis="engine2-legacy")
    assert np.allclose(got.current, values)
    assert got.time_axis["observed_sampling_hz"] == 4096
    assert got.time_axis["analysis_sampling_hz"] == 4098
    assert got.time_axis["resampled"] is False


@pytest.mark.parametrize("sampling_hz", [4098, 10000])
def test_prepare_matches_across_csv_ascii_and_channel_layouts(tmp_path, sampling_hz):
    t = np.arange(10000) / sampling_hz
    current = np.sin(2*np.pi*50*t) + 0.02*np.cos(2*np.pi*31*t)
    # Keep FFT bins above round-off: text encoders may differ by ~1 ulp at
    # exact Fourier zeros, where dB comparison is not a meaningful IO oracle.
    current += np.random.default_rng(7).normal(scale=0.001, size=len(t))
    cfg = {"processing_parameters": {
        "segment_length": 10000, "shift": 20, "f_sampling": sampling_hz, "cutoff_freq": 250,
        "db": True, "fault_types_to_use": ["rotor bar defect"],
        "phases_to_use": [1, 2], "loads_to_use": [100]}}
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    observed = []
    for i, (suffix, three, phase) in enumerate([
        (".CSV", False, 1), (".CSV", True, 2), (".TXT", False, 1), (".TXT", True, 2)
    ]):
        file = tmp_path / f"signal-{i}{suffix}"
        if suffix == ".CSV":
            columns = {"Time": t, "I1": 2*current, "I2": current, "I3": -current} if three else {"Time": t, "Current": current}
            pd.DataFrame(columns).to_csv(file, index=False)
        else:
            np.savetxt(file, np.column_stack([t, 2*current, current, -current] if three else [t, current]), delimiter=";")
        metadata = tmp_path / f"meta-{i}.csv"
        pd.DataFrame([{"measurement_id": "m", "base_id": "b", "state": "normal",
                       "phase": float(phase), "load_condition": 100, "experiment": "fixture",
                       "file_path": str(file)}]).to_csv(metadata, index=False)
        out = tmp_path / f"prepared-{i}"
        receipt = prepare(metadata, config, out, tmp_path)
        assert receipt["time_axis_checks"][0]["observed_sampling_hz"] == pytest.approx(sampling_hz)
        observed.append(np.load(out / "segments.npy"))
    assert all(np.allclose(observed[0], x, atol=1e-5, rtol=1e-5) for x in observed[1:])
