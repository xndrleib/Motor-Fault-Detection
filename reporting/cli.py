"""Command-line interface for reproducible SGDA reporting examples."""
import argparse
import json
import sys
from .core import prepare, synthesize, predict


def main():
    parser = argparse.ArgumentParser(description="SGDA-MotorDiag: подготовка, синтез, повторный инференс.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--metadata", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--path-base", required=True, help="База относительных file_path из metadata.")
    p.add_argument("--missing-current", choices=["reject", "legacy-drop"], default="reject",
                   help="legacy-drop явно воспроизводит dropna исходного конвейера; пропуски записываются в manifest.")
    p.add_argument("--output", required=True)
    s = sub.add_parser("synthesize")
    s.add_argument("--prepared", required=True)
    s.add_argument("--engine", required=True)
    s.add_argument("--fault", required=True)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--count", type=int, default=10)
    s.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3])
    s.add_argument("--eccentricity-method", choices=["slot-based", "simple"], default="slot-based")
    s.add_argument("--output", required=True)
    s.add_argument("--normalizer-path", help="Статистики обучающей выборки для дополнительного нормализованного экспорта.")
    p = sub.add_parser("predict")
    p.add_argument("--prepared", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu", choices=["cpu","cuda","mps"])
    p.add_argument("--batch-size", type=int, default=128)
    args = vars(parser.parse_args())
    command = args.pop("command")
    try:
        result = {"prepare":prepare, "synthesize":synthesize, "predict":predict}[command](**args)
    except (ValueError, KeyError, OSError, TypeError, RuntimeError) as exc:
        parser.exit(2, f"Ошибка: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
