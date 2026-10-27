"""Readable views of actual arrays and metrics; no model or signal processing."""
from pathlib import Path

import numpy as np
import pandas as pd


def save_prepared_preview(output, spectra, frequencies, metadata, db):
    normal = np.flatnonzero(metadata.state.eq("normal"))
    row = int(normal[0]) if len(normal) else 0
    value = "spectrum_db" if db else "fft_magnitude"
    pd.DataFrame({"frequency_hz": frequencies, value: spectra[row]}).to_csv(
        Path(output) / "spectrum_preview.csv", index=False)
    return {"row": row, "base_id": str(metadata.iloc[row].base_id),
            "segment_idx": int(metadata.iloc[row].segment_idx),
            "state": str(metadata.iloc[row].state), "file": "spectrum_preview.csv"}


def save_synthetic_view(output, source, synthetic, frequencies, metadata, fault_frequencies, db):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    value = "spectrum_db" if db else "fft_magnitude"
    pd.DataFrame({"frequency_hz": frequencies, "source_" + value: source,
                  "synthetic_" + value: synthetic}).to_csv(
        Path(output) / "spectrum_preview.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax in axes:
        ax.plot(frequencies, source, label="Исходный спектр", linewidth=1.2)
        ax.plot(frequencies, synthetic, label="Синтетический спектр", linewidth=1.2, alpha=.85)
        ax.set(xlabel="Частота, Гц", ylabel="Уровень спектра, дБ" if db else "Амплитуда БПФ")
        ax.grid(alpha=.2)
    axes[0].set_title("Полный диапазон")
    centre = float(fault_frequencies[0])
    axes[1].set_xlim(max(float(frequencies[0]), centre - 6), min(float(frequencies[-1]), centre + 6))
    window = np.abs(frequencies - centre) <= 6
    low = min(float(source[window].min()), float(synthetic[window].min()))
    high = max(float(source[window].max()), float(synthetic[window].max()))
    margin = max((high - low) * .15, 1.)
    axes[1].set_ylim(low - margin, high + margin)
    axes[1].set_title(f"Область диагностической частоты {centre:.2f} Гц")
    axes[0].legend(loc="best")
    fig.suptitle(f"Запись {metadata.base_id}, окно {int(metadata.segment_idx)}")
    fig.tight_layout()
    fig.savefig(Path(output) / "spectrum.png", dpi=180)
    fig.savefig(Path(output) / "spectrum.pdf")
    plt.close(fig)


def save_metric_table(output, metrics):
    rows = [{"metric": "accuracy", "value": metrics["accuracy"] * 100, "unit": "%"},
            {"metric": "macro_f1", "value": metrics["macro_f1"], "unit": "0..1"}]
    if "false_positive_rate" in metrics:
        rate = metrics["false_positive_rate"]
        rows.append({"metric": "false_positive_rate", "value": None if rate is None else rate * 100, "unit": "%"})
    pd.DataFrame(rows).to_csv(Path(output) / "metrics.csv", index=False)


def format_result(operation, result, arguments):
    output = Path(arguments["output"])
    lines = []
    if operation == "prepare":
        lines += ["Подготовка токовых записей", f'Перечень записей: {arguments["metadata"]}',
                  f'Конфигурация обработки: {arguments["config"]}',
                  f'Обработано записей: {result["raw_records"]}',
                  f'Получено окон: {result["shape"][0]}',
                  f'Частотных отсчётов в окне: {result["shape"][1]}',
                  f'Запись для просмотра: {result["preview"]["base_id"]}, окно {result["preview"]["segment_idx"]}',
                  f'Таблица окон: {output / "segments_metadata.csv"}',
                  f'Просмотр спектра: {output / "spectrum_preview.csv"}']
    elif operation == "synthesize":
        names = {"inter-turn short circuits": "межвитковое замыкание", "rotor bar defect": "дефект стержней ротора"}
        lines += ["Синтез спектральных признаков", f'Исходные спектры: {arguments["prepared"]}',
                  f'Параметры двигателя: {arguments["engine"]}',
                  f'Тип неисправности: {names.get(result["fault"], result["fault"])}',
                  f'Сформировано спектров: {result["shape"][0]}',
                  f'Частотных отсчётов в спектре: {result["shape"][1]}',
                  f'Сравнение спектров: {output / "spectrum.png"}',
                  f'Числовые значения рисунка: {output / "spectrum_preview.csv"}']
    else:
        metrics = result["segment_metrics"]
        title = "бинарной" if result["task"] == "binary" else "многоклассовой"
        lines += [f"Применение {title} модели", f'Подготовленные данные: {arguments["prepared"]}',
                  f'Каталог модели: {arguments["run"]}',
                  f'Оценено окон: {result["evaluated_windows"]}',
                  f'Доля верных ответов: {100 * metrics["accuracy"]:.2f} %',
                  f'Макро-F1: {metrics["macro_f1"]:.4f}']
        if "false_positive_rate" in metrics:
            value = metrics["false_positive_rate"]
            lines.append("Доля ложных срабатываний: не определена" if value is None else f"Доля ложных срабатываний: {100 * value:.2f} %")
        lines += [f'Метрики: {output / "metrics.csv"}', f'Предсказания: {output / "predictions.csv"}']
    lines.append(f"Каталог результатов: {output}")
    return "\n".join(lines)


def format_diagnosis(result, arguments):
    names = {"normal": "нормальное состояние", "anomalous": "неисправность",
             "inter-turn short circuits": "межвитковое замыкание",
             "rotor bar defect": "дефект стержней ротора"}
    return "\n".join(["Диагностика токовой записи", f'Входной файл: {arguments["input"]}',
                      f'Каталог модели: {arguments["run"]}', f'Обработано окон: {result["segments"]}',
                      f'Распознанное состояние: {names.get(result["prediction"], result["prediction"])}',
                      f'Результат: {Path(arguments["output"]) / "diagnosis.json"}'])


def format_verification(report, arguments):
    checks = report["functional_checks"]
    count = len(checks)
    peak = report["peak_localization"]
    return "\n".join(["Функциональная проверка синтеза", f'Подготовленные данные: {arguments["prepared"]}',
        f'Проверено групп признаков: {count}',
        f'Побитно совпавших повторов: {sum(x["bitwise_identical"] for x in checks)} из {count}',
        f'Групп с сохранением спектра вне полос добавления: {sum(x["outside_windows_identical"] for x in checks)} из {count}',
        f'Групп с конечными значениями: {sum(x["finite"] for x in checks)} из {count}',
        f'Измерений центра в допуске: {peak["within_tolerance"]} из {peak["total_measurements"]}',
        f'Допуск: {peak["tolerance_hz"]:.2f} Гц',
        f'Максимальное отклонение: {peak["maximum_deviation_hz"]:.8f} Гц',
        f'Полный отчёт: {Path(arguments["output"]) / "functional-report.json"}'])
