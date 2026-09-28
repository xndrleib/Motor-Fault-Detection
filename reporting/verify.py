"""Functional checks on actual prepared normal spectra."""
import argparse
import json
from pathlib import Path
import random
import numpy as np
from reporting.core import load_prepared, synthesize, write_json, sha256, runtime
from src.anomaly_injector import GaussianPeakInjector


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--prepared",required=True)
    p.add_argument("--engine",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=False)
    x,meta,freqs,manifest=load_prepared(a.prepared)
    checks=[]
    for slug,fault,orders,method in [
        ("itsc","inter-turn short circuits",[1,2,3],"slot-based"),
        ("rotor","rotor bar defect",[1,2,3],"slot-based"),
        ("bearing","bearing defect",[1,2,3],"slot-based"),
        ("mechanical","other mechanical defects",[1],"slot-based"),
        ("eccentricity-simple","air-gap eccentricity",[1],"simple")]:
        r1=synthesize(a.prepared,a.engine,fault,out/(slug+"-a"),seed=42,count=20,
                      orders=orders,eccentricity_method=method)
        r2=synthesize(a.prepared,a.engine,fault,out/(slug+"-b"),seed=42,count=20,
                      orders=orders,eccentricity_method=method)
        original=np.load(out/(slug+"-a")/"source_segments.npy")
        result=np.load(out/(slug+"-a")/"synthetic_segments.npy")
        inside=np.zeros(len(freqs),dtype=bool)
        half=r1["half_window_bins"]
        for frequency in r1["diagnostic_frequencies_hz"]:
            center=int(np.abs(freqs-frequency).argmin())
            inside[max(0,center-half):min(len(freqs),center+half+1)]=True
        checks.append({"fault":fault,"method":method,"orders":orders,
            "bitwise_identical":r1["array_sha256"]==r2["array_sha256"],
            "outside_windows_identical":bool(np.array_equal(original[:,~inside],result[:,~inside])),
            "finite":bool(np.isfinite(result).all()),"shape":list(result.shape),
            "frequencies":r1["diagnostic_frequencies_hz"]})
    performance=synthesize(a.prepared,a.engine,"inter-turn short circuits",out/"performance-10000",
                           seed=42,count=10000)
    # Isolate one injected peak from the real signal background for measurement.
    half=int(np.ceil(4/(freqs[1]-freqs[0])))
    generator=GaussianPeakInjector(half,(10,10),(1,1),negative=False,random_peak_position=True)
    deviations=[]
    random.seed(42)
    for _ in range(100):
        delta=generator.inject(np.zeros(len(freqs)),freqs,[50.0])
        deviations.append(abs(float(freqs[np.argmax(delta)])-50.0))
    report={"runtime":runtime(),"prepared_manifest_sha256":sha256(Path(a.prepared)/"manifest.json"),
            "functional_checks":checks,"performance_10000":performance,
            "peak_localization":{"repetitions":100,"frequency_hz":50,"peak_segment_half_width_hz":4,
                "interpretation":"1% of full 8 Hz injection band; comparison requires owner confirmation",
                "tolerance_hz":0.08,"maximum_deviation_hz":max(deviations),
                "within_tolerance":sum(d<=0.08 for d in deviations),
                "verdict_under_stated_interpretation": "pass" if max(deviations)<=0.08 else "fail",
                "algorithm_changed":False}}
    write_json(out/"functional-report.json",report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
