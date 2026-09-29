"""Current-file parsing and time-axis validation; no resampling or signal repair."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TIME_AXES = ("seconds", "milliseconds", "engine2-legacy")


@dataclass
class CurrentSignal:
    time: np.ndarray
    current: np.ndarray
    dropped_rows: list[int]
    time_axis: dict
    source_time: np.ndarray
    source_current: np.ndarray


def _ascii_columns(path, phase):
    rows = []
    width = None
    with path.open(encoding="latin") as stream:
        for number, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            if ";" not in line:
                if rows:
                    raise ValueError(f"строка {number}: ожидается разделитель ';'")
                continue  # Oscilloscope metadata before the numeric table.
            fields = line.split(";")
            if not rows and fields[0].strip().lower() == "time":
                continue
            if len(fields) not in (2, 4) or (width is not None and len(fields) != width):
                raise ValueError(f"строка {number}: ожидаются время и один или три канала")
            try:
                rows.append([float(value.strip()) for value in fields])
            except ValueError as exc:
                raise ValueError(f"строка {number}: нечисловое значение") from exc
            width = len(fields)
    if not rows:
        raise ValueError("пустая таблица ASCII")
    data = np.asarray(rows)
    return data[:, 0], data[:, 1 if data.shape[1] == 2 else phase]


def validate_time_axis(time, sampling_hz, profile):
    if profile not in TIME_AXES:
        raise ValueError(f"Неизвестный time_axis: {profile}")
    if len(time) < 2 or not np.isfinite(time).all():
        raise ValueError("Нужны хотя бы два конечных отсчёта времени")
    scale = 1.0 if profile == "seconds" else 0.001
    seconds = time * scale
    steps = np.diff(seconds)
    if np.any(steps <= 0):
        raise ValueError("Отсчёты времени должны строго возрастать")
    observed = float((len(seconds) - 1) / (seconds[-1] - seconds[0]))
    if sampling_hz is not None:
        if profile == "engine2-legacy":
            if sampling_hz != 4098:
                raise ValueError("engine2-legacy требует историческую конфигурацию FFT 4098 Гц")
            expected_hz = 4096.0
        else:
            expected_hz = float(sampling_hz)
        # A 0.5% per-step allowance admits microsecond text rounding at 4098 Hz.
        # The stricter whole-record bound rejects a different acquisition rate.
        if not np.isclose(observed, expected_hz, rtol=1e-4, atol=0):
            raise ValueError(f"Частота по временной оси {observed:.9g} Гц не соответствует {expected_hz:g} Гц")
        if not np.allclose(steps, 1 / expected_hz, rtol=0.005, atol=1e-9):
            raise ValueError("Неравномерная временная сетка или пропущенные отсчёты времени")
    return {"profile": profile, "input_time_unit": "s" if scale == 1 else "ms",
            "observed_sampling_hz": observed, "analysis_sampling_hz": sampling_hz,
            "legacy_analysis_frequency_preserved": profile == "engine2-legacy",
            "resampled": False}


def read_current_signal(path, phase=1, sampling_hz=None, time_axis="seconds",
                        missing_current="reject"):
    path = Path(path)
    try:
        if phase not in (1, 2, 3):
            raise ValueError("Фаза должна быть 1, 2 или 3")
        phase = int(phase)
        if missing_current not in ("reject", "legacy-drop"):
            raise ValueError("Неизвестная политика пропусков тока")
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError("Файл тока отсутствует или пуст")
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path).dropna(axis=1, how="all")
            if "Time" in frame and "Current" in frame:
                selected = frame[["Time", "Current"]]
            elif "Time" in frame and all(f"I{i}" in frame for i in (1, 2, 3)):
                selected = frame[["Time", f"I{phase}"]]
            elif len(frame.columns) == 3 and str(frame.columns[0]).startswith("Unnamed:"):
                selected = frame.iloc[:, 1:]  # Historical pandas index + two numeric columns.
            else:
                raise ValueError("Ожидается CSV Time,Current; Time,I1,I2,I3; либо исторический CSV с индексом")
            data = selected.apply(pd.to_numeric, errors="raise").to_numpy()
            times, current = data[:, 0], data[:, 1]
        elif path.suffix.lower() == ".txt":
            times, current = _ascii_columns(path, phase)
        else:
            raise ValueError("Поддерживаются .csv и .txt")
        timing = validate_time_axis(times, sampling_hz, time_axis)
        if np.isinf(current).any():
            raise ValueError("Бесконечные значения тока")
        missing = np.flatnonzero(np.isnan(current)).tolist()
        if missing and missing_current != "legacy-drop":
            raise ValueError("NaN в значениях тока")
        keep = np.isfinite(current)
        return CurrentSignal(times[keep], current[keep], missing, timing, times, current)
    except (ValueError, TypeError, OSError, pd.errors.ParserError) as exc:
        raise ValueError(f"Некорректный файл {path}: {exc}") from exc
