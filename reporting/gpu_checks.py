"""Bounded V100 checks: fixed-seed repetitions and explicit latency boundaries."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from reporting.core import build_model, load_config, predict, runtime, sha256, write_json


def latency(run, out):
    c=load_config(run/"training_config.yaml")
    model=build_model(c, 2 if c["task"]=="binary" else 3, run)
    pkg=torch.load(run/"checkpoints"/f"best_{c['task']}.pth",map_location="cpu",weights_only=True)
    model.load_state_dict(pkg.get("state_dict",pkg.get("state",pkg)),strict=True)
    model.to("cuda").eval()
    results=[]
    with torch.inference_mode():
        for length,seconds in [(250,1.0),(611,10000/4098)]:
            x=torch.zeros((1,1,length),device="cuda")
            for _ in range(20): model(x)
            torch.cuda.synchronize()
            values=[]
            for _ in range(200):
                begin=time.perf_counter()
                model(x)
                torch.cuda.synchronize()
                values.append((time.perf_counter()-begin)*1000)
            results.append({"frequency_bins":length,"represented_seconds":seconds,"batch":1,
                            "input":"zeros for timing only","boundary":"resident GPU input to synchronized logits; excludes FFT, I/O, transfers",
                            "mean_ms":float(np.mean(values)),"median_ms":float(np.median(values)),
                            "p95_ms":float(np.percentile(values,95)),"repetitions":len(values)})
    write_json(out/"latency.json",results)
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--workspace",required=True)
    a=parser.parse_args()
    root=Path(a.workspace).resolve()
    os.chdir(root)
    receipt=json.loads((root/"source-receipt.json").read_text())
    for path,digest in receipt["files"].items():
        if sha256(root/path)!=digest:
            raise RuntimeError(f"Source digest mismatch: {path}")
    if not torch.cuda.is_available(): raise RuntimeError("No allocated CUDA device")
    device=torch.cuda.get_device_name(0)
    if "V100" not in device: raise RuntimeError(f"Expected V100, got {device}")
    torch.set_num_threads(2)
    out=root/"gpu-results"
    out.mkdir(exist_ok=False)
    write_json(out/"environment.json",{"runtime":runtime(),"gpu":device,
        "slurm_job_id":os.environ.get("SLURM_JOB_ID"),"source_commit":receipt["commit"],
        "source_receipt_sha256":sha256(root/"source-receipt.json"),
        "imports":{"reporting":__import__("reporting.core",fromlist=["x"]).__file__},
        "nvidia_smi":subprocess.check_output(["nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv,noheader"],text=True)})
    summary={}
    for task in ["binary","multiclass"]:
        task_results=[]
        for repetition in [1,2]:
            name=f"{task}-seed42-repeat{repetition}"
            cmd=[sys.executable,"-m","experiments.train","--cfg",str(root/"configs"/f"{task}.yaml"),
                 "--offline","--device","cuda","--dataset-dir",str(root/"dataset"/"engine_2"),
                 "--prepared-dir",str(root/"prepared"),"--runs-dir",str(out/"runs"),"--run-name",name]
            with (out/(name+".log")).open("w") as log:
                subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
            run=out/"runs"/name
            result=predict(root/"prepared",run,out/(name+"-evaluation"),device="cuda",batch_size=128)
            task_results.append(result)
        timing=latency(out/"runs"/f"{task}-seed42-repeat1",out/f"{task}-seed42-repeat1-evaluation")
        summary[task]={"repetitions":task_results,"latency":timing,
            "accuracy_range":abs(task_results[0]["segment_metrics"]["accuracy"]-task_results[1]["segment_metrics"]["accuracy"]),
            "macro_f1_range":abs(task_results[0]["segment_metrics"]["macro_f1"]-task_results[1]["segment_metrics"]["macro_f1"]),
            "scope":"two fixed-seed repeats, same allocation; not five-seed variability or universal determinism"}
        write_json(out/"summary.json",summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
