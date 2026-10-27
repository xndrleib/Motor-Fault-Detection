"""Command-line operations for preparing, synthesizing and classifying spectra."""
import argparse
import json
import sys
from .core import prepare, synthesize, predict
from .input_files import TIME_AXES
from .output_views import format_result


def main():
    parser = argparse.ArgumentParser(description="SGDA-MotorDiag: подготовка токовых записей, синтез и классификация спектров.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--metadata", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--path-base", required=True, help="База относительных file_path из metadata.")
    p.add_argument("--time-axis", choices=TIME_AXES, default="seconds",
                   help="Единицы Time; engine2-legacy проверяет исходную сетку и сохраняет историческую FFT-конфигурацию.")
    p.add_argument("--missing-current", choices=["reject", "legacy-drop"], default="reject",
                   help="legacy-drop явно воспроизводит dropna исходного конвейера; пропуски записываются в manifest.")
    p.add_argument("--output", required=True)
    s = sub.add_parser("synthesize")
    s.add_argument("--prepared", required=True)
    s.add_argument("--engine", required=True)
    s.add_argument("--fault", required=True)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--count", type=int, default=10)
    s.add_argument("--peak-position", choices=["diagnostic", "configured"], default="diagnostic",
                   help="diagnostic: центр гауссовой компоненты в расчётной частоте; configured: позиция из конфигурации.")
    s.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3])
    s.add_argument("--eccentricity-method", choices=["slot-based", "simple"], default="slot-based")
    s.add_argument("--output", required=True)
    s.add_argument("--normalizer-path", help="Статистики обучающей выборки для дополнительного нормализованного экспорта.")
    s.add_argument("--synthesis-config", help="Отдельная конфигурация генерации; не изменяет подготовленные данные и настройки модели.")
    p = sub.add_parser("predict")
    p.add_argument("--prepared", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu", choices=["cpu","cuda","mps"])
    p.add_argument("--batch-size", type=int, default=128)
    for operation in sub.choices.values():
        operation.add_argument("--format", choices=["text", "json"], default="text",
                               help="Формат вывода в терминал; полный JSON также сохраняется в каталоге результата.")
    args = vars(parser.parse_args())
    command = args.pop("command")
    output_format = args.pop("format")
    try:
        result = {"prepare":prepare, "synthesize":synthesize, "predict":predict}[command](**args)
    except (ValueError, KeyError, OSError, TypeError, RuntimeError) as exc:
        parser.exit(2, f"Ошибка: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2) if output_format == "json"
          else format_result(command, result, args))


if __name__ == "__main__":
    main()
