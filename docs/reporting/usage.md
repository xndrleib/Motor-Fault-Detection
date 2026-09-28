# Отчётная версия SGDA MotorDiag 0.1.1

Основной сценарий качества: Normal / ITSC / RBD. Дополнительные группы проверяются отдельно как функции синтеза. Матрица требований и открытые трактовки: [requirements-matrix.md](requirements-matrix.md), [acceptance-clarifications.md](acceptance-clarifications.md). Режим эксплуатации не отправляет данные в Comet при --offline.

## Установка

Python 3.11. Создайте отдельную среду, установите соответствующий requirements-reporting-*.txt и пакет:

    python3.11 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements-reporting-linux.txt
    python -m pip install --no-deps .
    sgda-motordiag --help

macOS CPU-профиль закреплён отдельно. Linux-профиль соответствует существующей HPC-среде; CUDA-сборка и устройство должны быть записаны в конкретном run receipt.

## Минимальное воспроизведение

Для уже имеющегося бинарного запуска:

    bash examples/run_reporting_demo.sh \
      /path/to/Motor-Fault-Detection \
      /path/to/selected-binary-run \
      /path/to/new-output

Первый аргумент содержит dataset/engine_2 и исходные file_path из metadata. Второй содержит training_config.yaml, normalizer_binary.json, checkpoints/best_binary.pth, indices/test_idx_binary.npy и, для сверки, исторический predictions CSV. Каталог результата должен быть новым.

Сценарий подготавливает спектры, экспортирует 10000 синтетических ITSC-примеров и воспроизводит test-предсказания. Raw-пример engine_2 содержит четыре одиночных пропуска тока в выбранных 71 записях; флаг legacy-drop явно сохраняет историческую обработку. Для новых данных действует отклонение NaN/Inf по умолчанию.

## Команды и выходы

- prepare: raw metadata/config → segments.npy, freqs.npy, metadata, hash manifest.
- synthesize: подготовленные нормальные спектры + engine config → синтетические массивы, metadata, PNG/PDF. При --normalizer-path сохраняется также synthetic_normalized.npy; статистики не оцениваются на test.
- predict: точный saved run → сегментные метрики, голосование по записи и сравнение со старыми предсказаниями.
- python -m reporting.diagnose: новый неразмеченный CSV/ASCII → сегментные классы и общее решение. Метрики качества рассчитываются только для размеченного набора.
- python -m experiments.train --offline: полный исходный алгоритм обучения с локальными журналами. --dataset-dir и --prepared-dir отделяют входы от выходов.

Новые CSV могут иметь Time,Current либо Time,I1,I2,I3; --phase выбирает канал. Для синтеза остальных групп используется тот же API. Пример эксцентриситета без неизвестного R_s:

    python -m reporting.cli synthesize \
      --prepared /path/to/prepared \
      --engine dataset/engine_2/engine.yml \
      --fault "air-gap eccentricity" \
      --eccentricity-method simple --orders 1 \
      --count 20 --seed 42 --output /path/to/new-eccentricity

Это явно упрощённая существующая формула; slot-based требует R_s. Частоты вне сетки не обрезаются и не перемещаются на край молча.

## Проверки

    python -m pytest tests -q
    python -m reporting.verify --prepared /path/to/prepared \
      --engine dataset/engine_2/engine.yml --output /path/to/new-functional

Исторические checkpoints с полем state и новые с state_dict загружаются через weights_only=True и строгую проверку параметров модели. Каталоги reporting-результатов атомарны, существующие результаты не перезаписываются.

## Границы доказательств

Локальный инференс двух ResNet seed42 полностью повторил старые предсказания. Это не новые пять обучений. Сегментный split допускает общие исходные записи между частями; двигатели состояния в engine_2 различны. Показатели не характеризуют автоматически перенос на новый двигатель.

Оригинальный SGDA допускает случайное смещение пика. Проверка буквального допуска 1% полосы дала несоответствие; метод не изменён автоматически. GPU повторяемость/скорость и все неоднозначности ТЗ отражаются в отдельном протоколе, без утверждения о полной приёмке только по тестам.
