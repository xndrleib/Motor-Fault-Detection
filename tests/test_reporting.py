import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reporting.core import (classification_metrics, diagnostic_frequencies,
    new_output, validate_measurements, validate_processing, sha256)
from src.electrical_signature_frequencies import get_eccentricity_freqs
from src.experiment_logging import LocalExperiment

ENGINE = {"f1":50, "f_r":20, "s":0.05, "n":8, "D_pit":40,
          "D_ball":10, "beta":0, "p":2, "R_s":None}


def test_legacy_prediction_labels_have_one_semantic_mapping():
    from reporting.core import decode_prediction_labels
    classes=["inter-turn short circuits","normal","rotor bar defect"]
    assert decode_prediction_labels(["normal","rotor bar defect","inter-turn short circuits"],classes).tolist()==[1,2,0]
    assert decode_prediction_labels([1,"2",0.0],classes).tolist()==[1,2,0]
    for invalid in ["unknown",3,0.5,float("nan")]:
        with pytest.raises(ValueError,match="метка"):
            decode_prediction_labels([invalid],classes)


@pytest.mark.parametrize("fault,expected", [
    ("rotor bar defect", [35,40,45,55,60,65]),
    ("inter-turn short circuits", [30,50,70,130,150,170]),
    ("bearing defect", [37.5,60,100]),
])
def test_frequency_formula_known_values(fault, expected):
    assert diagnostic_frequencies(ENGINE, fault) == expected


def test_simple_eccentricity_does_not_require_slot_geometry():
    assert get_eccentricity_freqs(ENGINE, [1], method="simple") == [30,70]
    assert diagnostic_frequencies(ENGINE, "air-gap eccentricity", [1], "simple") == [30,70]
    with pytest.raises(ValueError, match="R_s"):
        diagnostic_frequencies(ENGINE, "air-gap eccentricity", [1], "slot-based")
    with pytest.raises(ValueError, match="method"):
        get_eccentricity_freqs(ENGINE, [1], method="typo")


def test_empty_engine_section_is_a_controlled_input_error():
    with pytest.raises(ValueError,match="mcsa"):
        diagnostic_frequencies({"mcsa":None},"rotor bar defect")


@pytest.mark.parametrize("key,value", [("f1",float("nan")),("f1",-1),("s",1)])
def test_invalid_engine_rejected(key,value):
    with pytest.raises(ValueError):
        diagnostic_frequencies({**ENGINE,key:value}, "rotor bar defect")


def test_macro_f1_and_fpr_not_weighted_f1():
    got=classification_metrics([0,0,1,1],[0,1,1,1],[0,1])
    assert got["accuracy"] == 0.75
    assert got["macro_f1"] == pytest.approx((2/3+4/5)/2)
    assert got["false_positive_rate"] == 0.5
    imbalanced=classification_metrics([0,0,0,1],[0,0,0,0],[0,1])
    assert imbalanced["macro_f1"] == pytest.approx(3/7)


def test_output_atomic_and_no_overwrite(tmp_path):
    p=tmp_path/"result"
    with pytest.raises(ValueError):
        with new_output(p) as out:
            (out/"partial").write_text("not complete")
            raise ValueError("failure")
    assert not p.exists()
    with new_output(p) as out:
        (out/"complete").write_text("ok")
    with pytest.raises(ValueError, match="существует"):
        with new_output(p):
            pass
    assert (p/"complete").read_text() == "ok"


def test_nan_rejected_unless_legacy_reproduction_requested(tmp_path):
    p=tmp_path/"input.csv"
    pd.DataFrame({"Time":[0,1,2,3], "Current":[1,np.nan,2,3]}).to_csv(p)
    meta=pd.DataFrame({"measurement_id":["a"],"file_path":[str(p)]}).set_index("measurement_id")
    with pytest.raises(ValueError,match="Некорректная"):
        validate_measurements(meta,2)
    rows=validate_measurements(meta,2,"legacy-drop")
    assert rows == [{"measurement_id":"a","missing_current_rows":[1]}]


def test_invalid_measurement_shape_rejected(tmp_path):
    p=tmp_path/"input.csv"
    pd.DataFrame({"OnlyOneColumn":[1,2,3]}).to_csv(p)
    meta=pd.DataFrame({"measurement_id":["a"],"file_path":[str(p)]}).set_index("measurement_id")
    with pytest.raises(ValueError,match="CSV"):
        validate_measurements(meta,2)


def test_offline_logging_writes_locally_without_credentials(tmp_path):
    from experiments.train import start_experiment
    e,_=start_experiment({},online=False,name="test",local_dir=tmp_path)
    e.log_parameter("seed",42)
    e.log_metric("accuracy",0.75)
    e.end()
    records=[json.loads(s) for s in (tmp_path/"experiment.jsonl").read_text().splitlines()]
    assert [r["event"] for r in records] == ["parameter","metric","end"]


@pytest.mark.parametrize("field,value", [("shift",0),("segment_length",-1),("cutoff_freq",3000),("f_sampling",8000)])
def test_invalid_fft_configuration_rejected(field,value):
    p={"shift":20,"segment_length":10000,"f_sampling":4098,"cutoff_freq":250,"fault_types_to_use":["rotor bar defect"]}
    with pytest.raises(ValueError):
        validate_processing({"processing_parameters":{**p,field:value}})


def test_malformed_yaml_has_file_context(tmp_path):
    from reporting.core import load_config
    p=tmp_path/"bad.yaml"
    p.write_text("broken: [1,")
    with pytest.raises(ValueError,match="bad.yaml"):
        load_config(p)


def test_new_file_formats_and_current_checkpoint_agree(tmp_path):
    import torch
    import yaml
    from src.models import CNN
    from reporting.diagnose import diagnose
    run=tmp_path/"run";(run/"checkpoints").mkdir(parents=True)
    cfg={"task":"binary","model":"CNN","model_parameters":{"dropout":0.2,"attention_module":False},
         "dataset_parameters":{"normalization_method":"min-max","normalization_mode":"global"},
         "processing_parameters":{"segment_length":10000,"shift":20,"f_sampling":4098,"cutoff_freq":250,
                                  "db":True,"fault_types_to_use":["inter-turn short circuits","rotor bar defect"]}}
    (run/"training_config.yaml").write_text(yaml.safe_dump(cfg))
    (run/"normalizer_binary.json").write_text('{"min": -40, "max": 80}')
    torch.manual_seed(42)
    model=CNN(num_classes=2,dropout_rate=0.2)
    torch.save({"state_dict":model.state_dict()},run/"checkpoints/best_binary.pth")
    time_values=np.arange(10000)/4098
    current=np.sin(2*np.pi*50*time_values)
    ordinary=tmp_path/"ordinary.csv"
    pd.DataFrame({"Time":time_values,"Current":current}).to_csv(ordinary,index=False)
    three=tmp_path/"three.csv"
    pd.DataFrame({"Time":time_values,"I1":2*current,"I2":current,"I3":-current}).to_csv(three,index=False)
    ascii_path=tmp_path/"ascii.txt"
    np.savetxt(ascii_path,np.column_stack([time_values,current]),delimiter=";")
    results=[]
    for i,(path,phase) in enumerate([(ordinary,1),(three,2),(ascii_path,1)]):
        result=diagnose(path,run,tmp_path/f"out-{i}",phase=phase)
        assert result["segments"] == 1
        assert result["metrics"] is None
        results.append(result["prediction"])
    assert len(set(results)) == 1
    np.savetxt(ascii_path,np.column_stack([time_values[::-1],current]),delimiter=";")
    with pytest.raises(ValueError,match="возрастать"):
        diagnose(ascii_path,run,tmp_path/"bad-time")
    (run/"normalizer_binary.json").write_text('{"min": NaN, "max": 80}')
    with pytest.raises(ValueError,match="Нормализация"):
        diagnose(ordinary,run,tmp_path/"bad-normalizer")
    assert not (tmp_path/"bad-normalizer").exists()


@pytest.mark.parametrize("suffix",[".CSV",".TXT"])
def test_prepare_accepts_declared_formats_case_insensitively(tmp_path,suffix):
    import yaml
    from reporting.core import prepare
    current=np.sin(np.arange(10000)/13)
    time_values=np.arange(10000)/4098
    path=tmp_path/("input"+suffix)
    if suffix==".CSV":
        pd.DataFrame({"Time":time_values,"Current":current}).to_csv(path)
    else:
        np.savetxt(path,np.column_stack([time_values,current]),delimiter=";")
    meta=tmp_path/"metadata.csv"
    pd.DataFrame([{"measurement_id":"test","base_id":"test","state":"normal","phase":1,
                   "load_condition":100,"experiment":"test","file_path":str(path)}]).to_csv(meta,index=False)
    config=tmp_path/"config.yaml"
    config.write_text(yaml.safe_dump({"processing_parameters":{
        "segment_length":10000,"shift":20,"f_sampling":4098,"cutoff_freq":250,
        "db":True,"fault_types_to_use":["rotor bar defect"],"phases_to_use":[1],"loads_to_use":[100]}}))
    receipt=prepare(meta,config,tmp_path/"prepared",tmp_path)
    assert receipt["shape"]==[1,611]
    assert np.isfinite(np.load(tmp_path/"prepared/segments.npy")).all()
