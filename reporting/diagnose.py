"""Inference on a new unlabelled current file using an existing trained model."""
import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from .core import load_config, build_model, new_output, sha256, write_json, runtime, validate_processing
from src.data_pipeline import segment_signal, perform_fft_on_segments, read_oscilloscope_data
from src.normalization import Normalizer


def diagnose(input, run, output, phase=1, device="cpu", batch_size=128):
    begin=time.perf_counter()
    path=Path(input)
    if phase not in [1,2,3] or batch_size < 1:
        raise ValueError("Требуются phase=1/2/3 и положительный batch_size.")
    if not path.is_file() or path.stat().st_size==0:
        raise ValueError("Файл тока отсутствует или пуст.")
    root=Path(run)
    cfg=load_config(root/"training_config.yaml")
    p=validate_processing(cfg)
    if path.suffix.lower()==".csv":
        df=pd.read_csv(path)
        if "Current" in df and "Time" in df:
            time_values=pd.to_numeric(df.Time,errors="raise").to_numpy()
            current=pd.to_numeric(df.Current,errors="raise").to_numpy()
        elif f"I{phase}" in df and "Time" in df:
            time_values=pd.to_numeric(df.Time,errors="raise").to_numpy()
            current=pd.to_numeric(df[f"I{phase}"],errors="raise").to_numpy()
        else:
            df=pd.read_csv(path,index_col=0).dropna(axis=1,how="all")
            if df.shape[1]!=2:
                raise ValueError("Ожидается CSV Time,Current; Time,I1,I2,I3; либо исторический CSV с индексом.")
            values=df.apply(pd.to_numeric,errors="raise").to_numpy()
            time_values,current=values[:,0],values[:,1]
    elif path.suffix.lower()==".txt":
        df=read_oscilloscope_data(path)
        time_values=np.asarray(df.Time,dtype=float)
        current=np.asarray(df.Data,dtype=float)
    else:
        raise ValueError("Поддерживаются .csv и .txt.")
    if len(current)<p["segment_length"] or not np.isfinite(current).all() or not np.isfinite(time_values).all():
        raise ValueError("Недостаточно отсчётов или обнаружены NaN/Inf.")
    if np.any(np.diff(time_values)<=0):
        raise ValueError("Отсчёты времени должны строго возрастать.")
    prep_start=time.perf_counter()
    windows=segment_signal(current,p["segment_length"],step=p["shift"],apply_window=False)
    spectra,freqs=perform_fft_on_segments(windows,f_sampling=p["f_sampling"],cutoff_freq=p["cutoff_freq"],db=p["db"])
    task=cfg["task"]
    classes=["normal","anomalous"] if task=="binary" else sorted(p["fault_types_to_use"]+["normal"])
    norm_path=root/f"normalizer_{task}.json"
    normalizer=Normalizer(cfg["dataset_parameters"]["normalization_method"],cfg["dataset_parameters"]["normalization_mode"])
    if normalizer.mode=="global": normalizer.load(norm_path)
    spectra=normalizer.transform(spectra).astype(np.float32)
    if not np.isfinite(spectra).all():
        raise ValueError("Нормализация создала NaN/Inf; проверьте статистики обучающей выборки.")
    prep_elapsed=time.perf_counter()-prep_start
    model=build_model(cfg,len(classes),root)
    checkpoint=root/"checkpoints"/f"best_{task}.pth"
    pkg=torch.load(checkpoint,map_location="cpu",weights_only=True)
    model.load_state_dict(pkg.get("state_dict",pkg.get("state",pkg)),strict=True)
    model.to(device).eval()
    torch.set_num_threads(4)
    predictions=[]
    def sync():
        if device=="cuda": torch.cuda.synchronize()
        elif device=="mps": torch.mps.synchronize()
    sync(); inference_start=time.perf_counter()
    with torch.inference_mode():
        for offset in range(0,len(spectra),batch_size):
            data=torch.from_numpy(spectra[offset:offset+batch_size]).unsqueeze(1).to(device)
            logits=model(data)
            if not torch.isfinite(logits).all():
                raise ValueError("Модель вернула NaN/Inf; диагностический результат не сформирован.")
            predictions.extend(logits.argmax(1).cpu().tolist())
    sync(); inference_elapsed=time.perf_counter()-inference_start
    predictions=np.array(predictions,dtype=int)
    vote=int(np.bincount(predictions,minlength=len(classes)).argmax())
    starts=np.arange(len(spectra))*p["shift"]/p["f_sampling"]
    result={"operation":"diagnose","task":task,"input_sha256":sha256(path),
            "checkpoint_sha256":sha256(checkpoint),"config_sha256":sha256(root/"training_config.yaml"),
            "normalizer_sha256":sha256(norm_path) if norm_path.exists() else None,
            "classes":classes,"phase":phase,"segments":len(spectra),
            "input_seconds_by_sample_count":len(current)/p["f_sampling"],
            "segment_seconds":p["segment_length"]/p["f_sampling"],
            "prediction":classes[vote],"voting":"majority; smallest class index on ties",
            "preprocessing_seconds_excluding_read":prep_elapsed,
            "inference_seconds_including_transfer":inference_elapsed,
            "total_seconds_including_loading":time.perf_counter()-begin,
            "device":device,"runtime":runtime(),
            "metrics":None,"metrics_reason":"input is unlabelled; no accuracy is inferred"}
    with new_output(output) as out:
        pd.DataFrame({"segment_idx":np.arange(len(spectra)),"start_seconds":starts,
                      "end_seconds":starts+p["segment_length"]/p["f_sampling"],
                      "prediction":predictions,"prediction_state":[classes[i] for i in predictions]}).to_csv(out/"predictions.csv",index=False)
        write_json(out/"diagnosis.json",result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",required=True)
    p.add_argument("--run",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--phase",type=int,default=1)
    p.add_argument("--device",choices=["cpu","cuda","mps"],default="cpu")
    p.add_argument("--batch-size",type=int,default=128)
    try:
        print(__import__("json").dumps(diagnose(**vars(p.parse_args())),ensure_ascii=False,indent=2))
    except (ValueError,KeyError,OSError,TypeError,RuntimeError) as e:
        p.exit(2,f"Ошибка: {e}\n")


if __name__=="__main__": main()
