import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from reporting.core import sha256, synthesize, predict
from reporting.output_views import format_result


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / "prepared"; root.mkdir()
    frequencies = np.linspace(0, 250, 611)
    spectra = np.stack([np.sin(frequencies), np.cos(frequencies)]).astype(np.float32)
    np.save(root / "segments.npy", spectra); np.save(root / "freqs.npy", frequencies)
    pd.DataFrame({"base_id": ["a", "b"], "measurement_id": ["a_1", "b_1"],
                  "segment_idx": [0, 0], "state": ["normal", "normal"]}).to_csv(root / "segments_metadata.csv", index=False)
    processing = {"include_negative_peaks": True, "peak_segment": 4, "amplitude_range": [.5, 20],
                  "sigma_range": [.5, 1.5], "random_peak_position": True, "noise_factor": 0,
                  "db": True, "f_sampling": 4098, "segment_length": 10000, "shift": 20,
                  "cutoff_freq": 250, "fault_types_to_use": ["inter-turn short circuits", "rotor bar defect"],
                  "phases_to_use": [1], "loads_to_use": [100]}
    manifest = {"processing_parameters": processing, "window_seconds": 10000 / 4098,
                "segments_sha256": sha256(root / "segments.npy"),
                "metadata_output_sha256": sha256(root / "segments_metadata.csv"),
                "frequencies_sha256": sha256(root / "freqs.npy")}
    (root / "manifest.json").write_text(json.dumps(manifest))
    engine = tmp_path / "engine.yml"; engine.write_text("f1: 50\nf_r: 20\n")
    return root, engine, spectra


def test_generation_override_is_local_and_preserves_default(prepared, tmp_path):
    root, engine, _ = prepared
    before = (root / "manifest.json").read_bytes()
    synthesize(root, engine, "inter-turn short circuits", tmp_path / "default-a", count=20)
    config = tmp_path / "generation.yaml"; config.write_text("include_negative_peaks: false\n")
    result = synthesize(root, engine, "inter-turn short circuits", tmp_path / "positive", count=20, synthesis_config=config)
    generated = np.load(tmp_path / "positive/synthetic_segments.npy")
    source = np.load(tmp_path / "positive/source_segments.npy")
    assert np.all(generated >= source)
    assert result["include_negative_peaks"] is False
    assert result["synthesis_config_sha256"] == sha256(config)
    assert (root / "manifest.json").read_bytes() == before
    synthesize(root, engine, "inter-turn short circuits", tmp_path / "default-b", count=20)
    assert np.array_equal(np.load(tmp_path / "default-a/synthetic_segments.npy"),
                          np.load(tmp_path / "default-b/synthetic_segments.npy"))
    assert np.any(np.load(tmp_path / "default-a/synthetic_segments.npy") < source)
    preview = pd.read_csv(tmp_path / "positive/spectrum_preview.csv")
    assert np.allclose(preview.synthetic_spectrum_db, generated[0])
    assert np.allclose(preview.source_spectrum_db, source[0])


@pytest.mark.parametrize("content", ["include_negative_peaks: yesplease", "f_sampling: 10000", "{}"])
def test_generation_configuration_cannot_change_preprocessing(prepared, tmp_path, content):
    root, engine, _ = prepared
    config = tmp_path / "generation.yaml"; config.write_text(content)
    with pytest.raises(ValueError, match="include_negative_peaks"):
        synthesize(root, engine, "inter-turn short circuits", tmp_path / "output", synthesis_config=config)
    assert not (tmp_path / "output").exists()


def test_predict_keeps_strict_processing_guard(prepared, tmp_path):
    root, _, _ = prepared
    manifest = json.loads((root / "manifest.json").read_text())
    processing = dict(manifest["processing_parameters"]); processing["include_negative_peaks"] = False
    model = tmp_path / "model"; model.mkdir()
    (model / "training_config.yaml").write_text(yaml.safe_dump({"task": "binary", "processing_parameters": processing}))
    with pytest.raises(ValueError, match="различаются"):
        predict(root, model, tmp_path / "evaluation")
    assert not (tmp_path / "evaluation").exists()


def test_human_summary_has_inputs_metrics_and_outputs():
    result = {"task": "binary", "evaluated_windows": 4032,
              "segment_metrics": {"accuracy": .98, "macro_f1": .97, "false_positive_rate": .01},
              "historical_prediction_agreement": 1}
    text = format_result("predict", result, {"prepared": "results/prepared", "run": "models/binary", "output": "results/binary"})
    for required in ("results/prepared", "models/binary", "4032", "98.00", "0.9700", "1.00", "metrics.csv", "predictions.csv"):
        assert required in text
    assert "historical" not in text and "совпадение" not in text and "верно" not in text


def test_json_cli_preserves_machine_output(monkeypatch, capsys):
    from reporting import cli
    result = {"operation": "predict", "task": "binary", "segment_metrics": {"accuracy": .98}}
    monkeypatch.setattr(cli, "predict", lambda **arguments: result)
    monkeypatch.setattr("sys.argv", ["sgda-motordiag", "predict", "--prepared", "input",
                                    "--run", "model", "--output", "result", "--format", "json"])
    cli.main()
    assert json.loads(capsys.readouterr().out) == result
