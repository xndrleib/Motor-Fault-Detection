"""Validated file boundaries around the existing SGDA implementation.

The processing, network and injection algorithms remain in src. Output folders
are committed only after the operation succeeds. Source datasets are read-only.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import shutil
import tempfile
import time

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

from src.anomaly_injector import GaussianPeakInjector, NoiseInjector, CompositeAnomalyInjector
from src.data_pipeline import preprocessing
from src.electrical_signature_frequencies import ANOMALY_FREQS, get_eccentricity_freqs
from src.models import CNN, ResNet, ResidualBlock
from src.normalization import Normalizer

FAULTS = ["inter-turn short circuits", "rotor bar defect"]
REQUIRED_META = {"measurement_id", "base_id", "state", "phase", "load_condition", "experiment", "file_path"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def load_config(path):
    try:
        with open(path) as f:
            c = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ValueError(f"Некорректный YAML в {path}: {exc}") from exc
    if not isinstance(c, dict):
        raise ValueError("Конфигурация должна быть YAML-словарём.")
    return c


@contextmanager
def new_output(path):
    destination = Path(path).resolve()
    if destination.exists():
        raise ValueError(f"Каталог результата уже существует: {destination}. Укажите новый путь.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=".sgda-", dir=destination.parent))
    try:
        yield scratch
        scratch.rename(destination)
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)


def runtime():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__,
            "cuda_runtime": torch.version.cuda}


def validate_processing(c):
    p = c["processing_parameters"]
    for key in ("segment_length", "shift"):
        v = p[key]
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            raise ValueError(f"{key} должен быть положительным целым числом.")
    fs, cutoff = float(p["f_sampling"]), float(p["cutoff_freq"])
    if fs not in {4098, 10000}:
        raise ValueError("Поддерживаемые частоты дискретизации: 4098 и 10000 Гц.")
    if not math.isfinite(fs) or fs <= 0 or not 0 < cutoff <= fs / 2:
        raise ValueError("Требуются f_sampling > 0 и 0 < cutoff_freq <= f_sampling/2.")
    if not p.get("fault_types_to_use"):
        raise ValueError("Не задан список fault_types_to_use.")
    return p


def resolve_metadata(metadata, path_base, processing):
    df = pd.read_csv(metadata)
    missing = REQUIRED_META - set(df)
    if missing:
        raise ValueError(f"В metadata отсутствуют колонки: {sorted(missing)}")
    if df.empty or df.measurement_id.isna().any() or df.measurement_id.duplicated().any():
        raise ValueError("metadata пуста либо measurement_id пустые/неуникальные.")
    classes = processing["fault_types_to_use"] + ["normal"]
    keep = (df.state.isin(classes) & df.phase.isin(processing["phases_to_use"])
            & df.load_condition.isin(processing["loads_to_use"]))
    df = df.loc[keep].copy()
    if df.empty:
        raise ValueError("После фильтрации не осталось записей.")
    base = Path(path_base).resolve()
    df["file_path"] = [str((base / str(p)).resolve()) for p in df.file_path]
    for path in df.file_path:
        p = Path(path)
        if not p.is_file() or p.stat().st_size == 0:
            raise ValueError(f"Входной файл отсутствует или пуст: {p}")
        if p.suffix.lower() not in {".csv", ".txt"}:
            raise ValueError(f"Неподдерживаемый формат: {p.suffix}")
    return df.set_index("measurement_id")


def validate_measurements(df, segment_length, missing_current="reject"):
    # Mirrors the supported historical CSV contract: index, Time, Current.
    # Reject invalid input before the legacy loader can silently drop NaNs.
    from src.data_pipeline import load_measurement, read_oscilloscope_data
    dropped = []
    if missing_current not in {"reject", "legacy-drop"}:
        raise ValueError("Неизвестная политика пропусков тока.")
    for measurement_id, row in df.iterrows():
        try:
            if Path(row.file_path).suffix.lower() == ".csv":
                raw = pd.read_csv(row.file_path, header=0, index_col=0).dropna(axis=1, how="all")
                if raw.shape[1] != 2:
                    raise ValueError("CSV должен содержать индекс и два поля Time, Current.")
                arr = raw.apply(pd.to_numeric, errors="raise").to_numpy()
                if not np.isfinite(arr[:, 0]).all() or np.isinf(arr[:, 1]).any():
                    raise ValueError("Некорректное время или бесконечный ток.")
                missing = np.isnan(arr[:, 1])
                if missing.any() and missing_current == "legacy-drop":
                    dropped.append({"measurement_id": str(measurement_id),
                                    "missing_current_rows": np.flatnonzero(missing).tolist()})
                    arr = arr[~missing]
            else:
                raw = read_oscilloscope_data(row.file_path)
                arr = raw[["Data"]].to_numpy()
            if len(arr) < segment_length or not np.isfinite(arr).all():
                raise ValueError("Недостаточно отсчётов либо найдены NaN/Inf.")
        except (ValueError, KeyError, TypeError, pd.errors.ParserError) as e:
            raise ValueError(f"Некорректная запись {measurement_id}: {e}") from e
    return dropped


def prepare(metadata, config, output, path_base, missing_current="reject"):
    c = load_config(config)
    p = validate_processing(c)
    df = resolve_metadata(metadata, path_base, p)
    dropped = validate_measurements(df, p["segment_length"], missing_current)
    with new_output(output) as out:
        x, meta, freqs = preprocessing(
            df, str(out), segment_length=p["segment_length"], step=p["shift"],
            f_sampling=p["f_sampling"], cutoff_freq=p["cutoff_freq"], apply_window=False, db=p["db"])
        # Preserve the filtered legacy row ordering used by saved split indices.
        meta = meta.reset_index()
        meta.to_csv(out / "segments_metadata.csv", index=False)
        if not np.isfinite(x).all() or len(x) != len(meta):
            raise ValueError("Предобработка создала некорректный массив.")
        shutil.copy2(config, out / "training_config.yaml")
        manifest = {"operation": "prepare", "runtime": runtime(),
                    "config_sha256": sha256(config), "metadata_sha256": sha256(metadata),
                    "processing_parameters": p, "shape": list(x.shape),
                    "raw_records": len(df), "window_seconds": p["segment_length"] / p["f_sampling"],
                    "missing_current_policy": missing_current, "legacy_dropped_rows": dropped,
                    "inputs": [{"path": r.file_path, "sha256": sha256(r.file_path)}
                               for _, r in df.iterrows()],
                    "segments_sha256": sha256(out / "segments.npy"),
                    "metadata_output_sha256": sha256(out / "segments_metadata.csv"),
                    "frequencies_sha256": sha256(out / "freqs.npy")}
        write_json(out / "manifest.json", manifest)
    return manifest


def load_prepared(path):
    root = Path(path)
    m = json.loads((root / "manifest.json").read_text())
    for name, key in (("segments.npy", "segments_sha256"),
                      ("segments_metadata.csv", "metadata_output_sha256"),
                      ("freqs.npy", "frequencies_sha256")):
        if sha256(root / name) != m[key]:
            raise ValueError(f"Контрольная сумма не совпадает: {name}")
    x = np.load(root / "segments.npy", mmap_mode="r", allow_pickle=False)
    meta = pd.read_csv(root / "segments_metadata.csv")
    freqs = np.load(root / "freqs.npy", allow_pickle=False)
    if len(x) != len(meta) or x.ndim != 2 or x.shape[1] != len(freqs):
        raise ValueError("Размерности массива и метаданных не совпадают.")
    return x, meta, freqs, m


def diagnostic_frequencies(engine, fault, orders=(1, 2, 3), eccentricity_method="slot-based"):
    if not isinstance(engine, dict):
        raise ValueError("Конфигурация двигателя должна быть словарём.")
    c = engine.get("mcsa", engine)
    if not isinstance(c, dict):
        raise ValueError("Раздел mcsa должен содержать параметры двигателя в виде словаря.")
    required = {"rotor bar defect": ["f1", "s"], "inter-turn short circuits": ["f1", "f_r"],
                "bearing defect": ["f_r", "n", "D_pit", "D_ball", "beta"],
                "air-gap eccentricity": ["f1", "f_r"],
                "other mechanical defects": ["f1", "f_r"]}
    if fault not in required:
        raise ValueError(f"Неизвестный тип дефекта: {fault}")
    keys = required[fault] + (["R_s", "p"] if fault == "air-gap eccentricity" and eccentricity_method == "slot-based" else [])
    for k in keys:
        v = c.get(k)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            raise ValueError(f"Параметр двигателя {k} отсутствует или не является конечным числом.")
        if k != "beta" and (v < 0 or (k != "s" and v == 0)):
            raise ValueError(f"Недопустимое значение параметра {k}.")
    if "s" in keys and c["s"] >= 1:
        raise ValueError("Скольжение s должно принадлежать [0, 1).")
    if fault == "bearing defect" and (c["n"] != int(c["n"]) or c["D_ball"] >= c["D_pit"]):
        raise ValueError("Требуются целое n и 0 < D_ball < D_pit.")
    if not orders or any(not isinstance(n, int) or n <= 0 for n in orders):
        raise ValueError("Порядки гармоник должны быть положительными целыми.")
    if fault == "air-gap eccentricity":
        return get_eccentricity_freqs(c, n_range=orders, method=eccentricity_method)
    if fault in {"rotor bar defect", "other mechanical defects"}:
        return ANOMALY_FREQS[fault](c, n_range=orders)
    return ANOMALY_FREQS[fault](c)


def synthesize(prepared, engine, fault, output, seed=42, count=10,
               orders=(1, 2, 3), eccentricity_method="slot-based", normalizer_path=None):
    operation_start = time.perf_counter()
    x, meta, freqs, manifest = load_prepared(prepared)
    if count < 1:
        raise ValueError("count должен быть положительным.")
    p = manifest["processing_parameters"]
    fault_freqs = diagnostic_frequencies(load_config(engine), fault, orders, eccentricity_method)
    outside = [f for f in fault_freqs if not freqs[0] <= f <= freqs[-1]]
    if outside:
        raise ValueError(f"Диагностические частоты вне представленного диапазона: {outside}. "
                         "Явно выберите подходящие orders или диапазон FFT; частоты не обрезаются молча.")
    normal = np.flatnonzero(meta.state.eq("normal"))
    if not len(normal):
        raise ValueError("В подготовленных данных нет нормальных записей.")
    source_rows = np.resize(normal, count)
    half_bins = int(np.ceil(p["peak_segment"] / (freqs[1] - freqs[0])))
    injector = CompositeAnomalyInjector({
        "peak": GaussianPeakInjector(peak_segment=half_bins,
                amplitude_range=tuple(p["amplitude_range"]), sigma_range=tuple(p["sigma_range"]),
                negative=p["include_negative_peaks"], random_peak_position=p["random_peak_position"]),
        "noise": NoiseInjector(p["noise_factor"])})
    if p["peak_segment"] <= 0 or p["sigma_range"][0] <= 0 or p["noise_factor"] < 0:
        raise ValueError("Некорректные параметры инжектора.")
    for k in ("amplitude_range", "sigma_range"):
        if len(p[k]) != 2 or not all(math.isfinite(v) for v in p[k]) or p[k][0] > p[k][1]:
            raise ValueError(f"Некорректный диапазон {k}.")
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        start = time.perf_counter()
        generated = np.stack([injector.inject(np.asarray(x[i]), freqs, fault_freqs) for i in source_rows])
        elapsed = time.perf_counter() - start
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
    if not np.isfinite(generated).all():
        raise ValueError("Синтез создал NaN/Inf.")
    with new_output(output) as out:
        np.save(out / "synthetic_segments.npy", generated)
        np.save(out / "source_segments.npy", x[source_rows])
        np.save(out / "freqs.npy", freqs)
        selected = meta.iloc[source_rows].copy()
        selected["source_row"] = source_rows
        selected["synthetic_fault"] = fault
        selected["seed"] = seed
        selected["amplitude_min"] = p["amplitude_range"][0]
        selected["amplitude_max"] = p["amplitude_range"][1]
        selected["sigma_min_bins"] = p["sigma_range"][0]
        selected["sigma_max_bins"] = p["sigma_range"][1]
        selected["peak_half_width_hz"] = p["peak_segment"]
        selected.to_csv(out / "synthetic_metadata.csv", index=False)
        if normalizer_path:
            cfg = load_config(Path(prepared) / "training_config.yaml")
            norm = Normalizer(cfg["dataset_parameters"]["normalization_method"],
                              cfg["dataset_parameters"]["normalization_mode"])
            norm.load(normalizer_path)
            normalized = norm.transform(generated).astype(np.float32)
            if not np.isfinite(normalized).all():
                raise ValueError("Нормализация создала NaN/Inf; проверьте статистики обучающей выборки.")
            np.save(out / "synthetic_normalized.npy", normalized)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(freqs, x[source_rows[0]], label="Normal", linewidth=1)
        ax.plot(freqs, generated[0], label="SGDA", linewidth=1, alpha=0.85)
        ax.set(xlabel="Frequency, Hz", ylabel="Spectrum, dB" if p["db"] else "FFT magnitude")
        ax.legend(); fig.tight_layout()
        fig.savefig(out / "spectrum.png", dpi=180)
        fig.savefig(out / "spectrum.pdf")
        plt.close(fig)
        receipt = {"operation": "synthesize", "seed": seed, "fault": fault, "count": count,
                   "orders": list(orders), "eccentricity_method": eccentricity_method,
                   "diagnostic_frequencies_hz": fault_freqs, "half_window_bins": half_bins,
                   "shape": list(generated.shape), "synthesis_seconds": elapsed,
                   "milliseconds_per_segment": elapsed * 1000 / count,
                   "window_seconds": manifest["window_seconds"], "runtime": runtime(),
                   "engine_sha256": sha256(engine),
                   "normalizer_sha256": sha256(normalizer_path) if normalizer_path else None,
                   "normalization": "additional synthetic_normalized.npy with provided training statistics" if normalizer_path else "raw FFT representation; no fitted normalizer supplied",
                   "generation_and_export_seconds": time.perf_counter() - operation_start,
                   "prepared_manifest_sha256": sha256(Path(prepared) / "manifest.json"),
                   "array_sha256": sha256(out / "synthetic_segments.npy")}
        write_json(out / "manifest.json", receipt)
    return receipt


def classification_metrics(y, pred, labels):
    y, pred = np.asarray(y, dtype=int), np.asarray(pred, dtype=int)
    if y.shape != pred.shape or not len(y):
        raise ValueError("Пустые или несогласованные метки/предсказания.")
    if not set(np.unique(y)).issubset(labels) or not set(np.unique(pred)).issubset(labels):
        raise ValueError("Неизвестные метки.")
    result = {"n": len(y), "accuracy": float(accuracy_score(y, pred)),
              "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
              "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
              "label_order": list(labels)}
    if labels == [0, 1]:
        normal = y == 0
        result["false_positive_rate"] = float((pred[normal] == 1).mean()) if normal.any() else None
    return result


def decode_prediction_labels(values, classes):
    """Read both numeric and textual labels used by historical CSV exports."""
    mapping = {name: index for index, name in enumerate(classes)}
    codes = []
    for value in values:
        if isinstance(value, str) and value in mapping:
            codes.append(mapping[value])
            continue
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Неизвестная сохранённая метка: {value}") from exc
        if not math.isfinite(number) or number != int(number) or not 0 <= number < len(classes):
            raise ValueError(f"Неизвестная сохранённая метка: {value}")
        codes.append(int(number))
    return np.asarray(codes, dtype=np.int64)


def build_model(config, num_classes, run):
    mp = config["model_parameters"]
    prior = None
    if mp.get("attention_module"):
        mask = torch.load(Path(run) / "mask.pt", map_location="cpu", weights_only=True)
        prior = dict(mask=mask, w_in_init=1.0, w_out_init=0.0, learnable_in=True, learnable_out=False)
    kw = dict(num_classes=num_classes, dropout_rate=mp["dropout"], prior_kwargs=prior)
    if config["model"] == "ResNet":
        return ResNet(ResidualBlock, [2, 2, 2, 2], **kw)
    if config["model"] == "CNN":
        return CNN(**kw)
    raise ValueError("Поддерживаются сохранённые модели CNN и ResNet.")


def predict(prepared, run, output, device="cpu", batch_size=128):
    root = Path(run)
    cfg_path = root / "training_config.yaml"
    c = load_config(cfg_path)
    task = c["task"]
    if task not in {"binary", "multiclass"} or batch_size < 1:
        raise ValueError("Неверная задача или размер batch.")
    x, meta, freqs, manifest = load_prepared(prepared)
    if c["processing_parameters"] != manifest["processing_parameters"]:
        raise ValueError("Параметры предобработки и сохранённого запуска различаются.")
    index_path = root / "indices" / f"test_idx_{task}.npy"
    idx = np.load(index_path, allow_pickle=False)
    if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer) or len(np.unique(idx)) != len(idx):
        raise ValueError("Индексы test должны быть уникальным целочисленным вектором.")
    if idx.min() < 0 or idx.max() >= len(meta):
        raise ValueError("Индексы test выходят за пределы подготовленных данных.")
    test_meta = meta.iloc[idx].copy().reset_index(drop=True)
    pred_path = root / f"segments_metadata_test_{task}_pred.csv"
    old = pd.read_csv(pred_path, index_col=0) if pred_path.exists() else None
    if old is not None:
        if len(old) != len(idx) or not np.array_equal(old.index.to_numpy(), idx):
            raise ValueError("Порядок test_idx отличается от исторического CSV.")
        for key in ("base_id", "state", "phase", "load_condition", "start_time", "end_time"):
            if key in old:
                a, b = old[key].to_numpy(), test_meta[key].to_numpy()
                ok = np.allclose(a, b, rtol=0, atol=1e-8) if np.issubdtype(a.dtype, np.number) else np.array_equal(a, b)
                if not ok:
                    raise ValueError(f"Состав test не совпадает с историческим CSV: {key}")
    classes = ["normal", "anomalous"] if task == "binary" else sorted(c["processing_parameters"]["fault_types_to_use"] + ["normal"])
    y = (test_meta.state != "normal").astype(int).to_numpy() if task == "binary" else test_meta.state.map({s:i for i,s in enumerate(classes)}).to_numpy()
    norm_path = root / f"normalizer_{task}.json"
    normalizer = Normalizer(method=c["dataset_parameters"]["normalization_method"],
                            mode=c["dataset_parameters"]["normalization_mode"])
    if normalizer.mode == "global":
        normalizer.load(norm_path)
    data = normalizer.transform(np.asarray(x[idx]))
    if not np.isfinite(data).all():
        raise ValueError("Нормализация создала NaN/Inf; проверьте статистики обучающей выборки.")
    checkpoint = root / "checkpoints" / f"best_{task}.pth"
    model = build_model(c, len(classes), root)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state" in state:
        state = state["state"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    torch.set_num_threads(4)
    loader = DataLoader(TensorDataset(torch.from_numpy(data.astype(np.float32)).unsqueeze(1)),
                        batch_size=batch_size, shuffle=False)
    def sync():
        if device.startswith("cuda"): torch.cuda.synchronize()
        elif device == "mps": torch.mps.synchronize()
    outputs = []
    sync()
    start = time.perf_counter()
    with torch.inference_mode():
        for (inputs,) in loader:
            outputs.append(model(inputs.to(device)).cpu().numpy())
    sync()
    elapsed = time.perf_counter() - start
    logits = np.concatenate(outputs)
    if not np.isfinite(logits).all():
        raise ValueError("Модель вернула NaN/Inf; диагностический результат не сформирован.")
    preds = logits.argmax(axis=1)
    labels = list(range(len(classes)))
    result = {"operation": "predict", "task": task, "classes": classes,
              "segment_metrics": classification_metrics(y, preds, labels),
              "runtime": runtime(), "device": device, "batch_size": batch_size,
              "inference_seconds_including_transfer": elapsed,
              "window_seconds": manifest["window_seconds"], "by_load": {},
              "checkpoint_sha256": sha256(checkpoint), "config_sha256": sha256(cfg_path),
              "indices_sha256": sha256(index_path), "normalizer_sha256": sha256(norm_path) if norm_path.exists() else None,
              "prepared_manifest_sha256": sha256(Path(prepared) / "manifest.json")}
    test_meta["true_label"], test_meta["prediction"] = y, preds
    test_meta["test_index"] = idx
    for load, group in test_meta.groupby("load_condition"):
        result["by_load"][str(load)] = classification_metrics(group.true_label, group.prediction, labels)
    if old is not None:
        reference = decode_prediction_labels(old[f"{task}_prediction"], classes)
        result["historical_prediction_agreement"] = float(np.mean(reference == preds))
        result["historical_prediction_sha256"] = sha256(pred_path)
    records = []
    for base, group in test_meta.groupby("base_id", sort=True):
        if group.true_label.nunique() != 1:
            raise ValueError(f"Неоднозначная метка записи {base}.")
        # Ties use the smallest class index, made explicit in the receipt.
        vote = int(np.bincount(group.prediction, minlength=len(classes)).argmax())
        records.append({"base_id":base, "true_label":int(group.true_label.iloc[0]),
                        "prediction":vote, "segments":len(group)})
    records = pd.DataFrame(records)
    result["record_metrics"] = classification_metrics(records.true_label, records.prediction, labels)
    result["voting"] = {"group":"base_id", "rule":"majority", "ties":"smallest class index",
                        "population":"records represented in segment test split, not unseen-record validation"}
    with new_output(output) as out:
        test_meta.to_csv(out / "predictions.csv", index=False)
        records.to_csv(out / "record_predictions.csv", index=False)
        np.save(out / "logits.npy", logits)
        write_json(out / "metrics.json", result)
    return result
