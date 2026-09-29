"""Functional checks on actual prepared normal spectra."""
import argparse
import json
from pathlib import Path
import random
import numpy as np
from reporting.core import load_prepared, synthesize, write_json, sha256, runtime
from src.anomaly_injector import DiagnosticGaussianPeakInjector


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
    # Recover each component's continuous centre from its actual sampled change.
    # Log-magnitude quadratic fitting is independent of the generating formula.
    generator=DiagnosticGaussianPeakInjector(4,(0.5,20),(0.5,1.5),negative=True)
    deviations=[]; measured=[]
    targets=sorted({float(f) for row in checks for f in row["frequencies"]}
                   | {float(freqs[0]), float(freqs[-1])})
    normal_rows=np.flatnonzero(meta.state.eq("normal"))
    random.seed(42)
    for target in targets:
        for trial in range(100):
            source=np.asarray(x[normal_rows[trial % len(normal_rows)]])
            result=generator.inject(source,freqs,[target])
            delta=result.astype(float)-source.astype(float)
            indices=np.argsort(np.abs(delta))[-3:]
            if np.any(np.abs(delta[indices]) == 0):
                raise ValueError("Insufficient nonzero samples to measure a centre")
            origin=freqs[np.argmax(np.abs(delta))]
            qa,qb,_=np.polyfit(freqs[indices]-origin,np.log(np.abs(delta[indices])),2)
            if qa >= 0:
                raise ValueError("Measured component is not a Gaussian peak")
            center=float(origin-qb/(2*qa))
            deviations.append(abs(center-target))
            measured.append({"frequency_hz":target,"trial":trial,"fitted_center_hz":center,
                             "error_hz":abs(center-target)})
    report={"runtime":runtime(),"prepared_manifest_sha256":sha256(Path(a.prepared)/"manifest.json"),
            "functional_checks":checks,"performance_10000":performance,
            "peak_localization":{"repetitions_per_frequency":100,"frequencies_hz":targets,
                "total_measurements":len(measured),"peak_segment_half_width_hz":4,
                "peak_position":"diagnostic",
                "center_definition":"continuous Gaussian centre reconstructed from the log absolute added component",
                "interpretation":"1% of the configured full 8 Hz support band",
                "input":"actual float32 normal spectra; each component measured separately; noise disabled",
                "tolerance_hz":0.08,"maximum_deviation_hz":max(deviations),
                "within_tolerance":sum(d<=0.08 for d in deviations),
                "verdict_under_stated_interpretation": "pass" if max(deviations)<=0.08 else "fail",
                "measurements":measured}}
    write_json(out/"functional-report.json",report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
