"""Input configuration changes must preserve the numerical training inputs."""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from reporting.core import prepare
from reporting.input_files import resolve_input_parameters


def experiment(tmp_path, *, configured=True):
    current = np.random.default_rng(7).normal(size=16384)
    current[1234] = np.nan
    signal = tmp_path/'signal.csv'
    pd.DataFrame({'0':np.arange(16384)*1000/4096,'1':current}).to_csv(signal)
    meta = tmp_path/'metadata.csv'
    pd.DataFrame([{'measurement_id':'m','base_id':'b','state':'normal','phase':1,
                   'load_condition':100,'experiment':'example','file_path':signal.name}]).to_csv(meta,index=False)
    cfg = {'processing_parameters':{'segment_length':10000,'shift':20,'f_sampling':4098,
        'cutoff_freq':250,'db':True,'fault_types_to_use':['rotor bar defect'],
        'phases_to_use':[1],'loads_to_use':[100]}}
    if configured:
        cfg['input_parameters']={'time_unit':'milliseconds','sampling_hz':4096,'missing_current':'drop'}
    path = tmp_path/'experiment.yaml'
    path.write_text(yaml.safe_dump(cfg))
    return signal,meta,path,cfg


def test_experiment_input_settings_preserve_arrays_metadata_and_source(tmp_path):
    signal,meta,path,cfg = experiment(tmp_path)
    original = signal.read_bytes()
    explicit = prepare(meta,path,tmp_path/'explicit',tmp_path,
                       missing_current='legacy-drop',time_axis='engine2-legacy')
    configured = prepare(meta,path,tmp_path/'configured',tmp_path)
    for name in ['segments.npy','freqs.npy']:
        assert np.array_equal(np.load(tmp_path/'explicit'/name),np.load(tmp_path/'configured'/name))
    assert (tmp_path/'explicit/segments_metadata.csv').read_bytes()==(tmp_path/'configured/segments_metadata.csv').read_bytes()
    assert signal.read_bytes()==original
    assert configured['input_parameters']==cfg['input_parameters']
    assert configured['processing_parameters']==explicit['processing_parameters']
    assert configured['dropped_current_rows']==[{'measurement_id':'m','missing_current_rows':[1234]}]
    assert configured['time_axis_checks'][0]['observed_sampling_hz']==4096
    assert configured['time_axis_checks'][0]['analysis_sampling_hz']==4098
    assert configured['time_axis_checks'][0]['resampled'] is False


def test_strict_override_is_not_weakened_by_drop_in_config(tmp_path):
    _,meta,path,_ = experiment(tmp_path)
    with pytest.raises(ValueError,match='NaN'):
        prepare(meta,path,tmp_path/'rejected',tmp_path,missing_current='reject')
    assert not (tmp_path/'rejected').exists()
    with pytest.raises(ValueError,match='Частота'):
        prepare(meta,path,tmp_path/'wrong-unit',tmp_path,time_axis='seconds')
    assert not (tmp_path/'wrong-unit').exists()


@pytest.mark.parametrize('values',[
    None,[],{'unexpected':True},{'missing_current':'ignore'},
    {'time_unit':'guess'},{'sampling_hz':True},{'sampling_hz':0},
    {'sampling_hz':float('nan')},{'sampling_hz':float('inf')},
])
def test_bad_input_config_is_rejected(values):
    with pytest.raises(ValueError):
        resolve_input_parameters({'input_parameters':values})


def test_missing_input_block_keeps_seconds_and_reject_defaults():
    values = resolve_input_parameters({})
    assert values.time_unit=='seconds'
    assert values.sampling_hz is None
    assert values.missing_current=='reject'


def test_cli_reads_experiment_config_without_input_flags(tmp_path):
    _,meta,path,_ = experiment(tmp_path)
    output = tmp_path/'cli-output'
    process = subprocess.run([sys.executable,'-m','reporting.cli','prepare',
        '--metadata',str(meta),'--config',str(path),'--path-base',str(tmp_path),
        '--output',str(output),'--format','json'],capture_output=True,text=True)
    assert process.returncode==0,process.stderr
    result = json.loads(process.stdout)
    assert result['input_parameters']['missing_current']=='drop'
    assert result['shape']==[320,611]


def test_diagnose_uses_experiment_input_settings_and_checks_model_compatibility(tmp_path):
    import torch
    from reporting.diagnose import diagnose
    from src.models import CNN
    run=tmp_path/'model'
    (run/'checkpoints').mkdir(parents=True)
    config={'task':'binary','model':'CNN','model_parameters':{'dropout':.2,'attention_module':False},
        'dataset_parameters':{'normalization_method':'min-max','normalization_mode':'global'},
        'processing_parameters':{'segment_length':10000,'shift':20,'f_sampling':4098,'cutoff_freq':250,
            'db':True,'fault_types_to_use':['inter-turn short circuits','rotor bar defect']}}
    (run/'training_config.yaml').write_text(yaml.safe_dump(config))
    (run/'normalizer_binary.json').write_text('{"min": -80, "max": 80}')
    torch.manual_seed(42)
    torch.save({'state_dict':CNN(num_classes=2,dropout_rate=.2).state_dict()},run/'checkpoints/best_binary.pth')
    current=np.random.default_rng(7).normal(size=10001)
    current[100]=np.nan
    signal=tmp_path/'record.csv'
    pd.DataFrame({'Time':np.arange(10001)*1000/4096,'Current':current}).to_csv(signal,index=False)
    declared={**config,'input_parameters':{'time_unit':'milliseconds','sampling_hz':4096,'missing_current':'drop'}}
    experiment_path=tmp_path/'input.yaml'
    experiment_path.write_text(yaml.safe_dump(declared))
    result=diagnose(signal,run,tmp_path/'diagnosis',input_config=experiment_path)
    assert result['segments']==1
    assert result['dropped_current_rows']==[100]
    assert result['input_parameters']==declared['input_parameters']
    assert result['time_axis_check']['resampled'] is False
    declared['processing_parameters']={**config['processing_parameters'],'shift':21}
    experiment_path.write_text(yaml.safe_dump(declared))
    with pytest.raises(ValueError,match='не совпадают'):
        diagnose(signal,run,tmp_path/'incompatible',input_config=experiment_path)
    assert not (tmp_path/'incompatible').exists()
