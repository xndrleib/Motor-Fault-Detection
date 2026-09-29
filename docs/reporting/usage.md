# Отчётная версия SGDA MotorDiag 0.1.3

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

Для исходного Engine 2 скрипт явно использует --time-axis engine2-legacy: временная координата записана в миллисекундах и соответствует 4096 отсчётам/с; исторические FFT-параметры 4098 Гц сохраняются для точного воспроизведения. Полученные массивы должны совпадать с прежними побайтно.

Сценарий подготавливает спектры, экспортирует 10000 синтетических ITSC-примеров и воспроизводит test-предсказания. Raw-пример engine_2 содержит четыре одиночных пропуска тока в выбранных 71 записях; флаг legacy-drop явно сохраняет историческую обработку. Для новых данных действует отклонение NaN/Inf по умолчанию.

## Команды и выходы

- prepare: raw metadata/config → segments.npy, freqs.npy, metadata, hash manifest.
- synthesize: подготовленные нормальные спектры + engine config → синтетические массивы, metadata, PNG/PDF. Центр гауссовой компоненты задаётся расчётной диагностической частотой в герцах, поддержка ограничивается заданной полушириной. При --normalizer-path сохраняется также synthetic_normalized.npy; статистики не оцениваются на test.
- predict: точный saved run → сегментные метрики, голосование по записи и сравнение со старыми предсказаниями.
- python -m reporting.diagnose: новый неразмеченный CSV/ASCII → сегментные классы и общее решение. Метрики качества рассчитываются только для размеченного набора.
- python -m experiments.train --offline: полный исходный алгоритм обучения с локальными журналами. --dataset-dir и --prepared-dir отделяют входы от выходов.

Команды prepare и diagnose принимают CSV Time,Current, CSV Time,I1,I2,I3 и ASCII с разделителем «;». Для трёхфазного prepare канал задаёт поле phase в metadata; для diagnose — --phase. Time задаётся в секундах по умолчанию или явно в миллисекундах через --time-axis milliseconds. Проверяются частота и равномерность временной сетки; ресемплинг не выполняется. Для синтеза остальных групп используется тот же API. Пример эксцентриситета без неизвестного R_s:

    python -m reporting.cli synthesize \
      --prepared /path/to/prepared \
      --engine dataset/engine_2/engine.yml \
      --fault "air-gap eccentricity" \
      --eccentricity-method simple --orders 1 \
      --count 20 --seed 42 --output /path/to/new-eccentricity

Это явно упрощённая существующая формула; slot-based требует R_s. Частоты вне сетки не обрезаются и не перемещаются на край молча.

## Контракт экспорта предсказаний

predictions.csv содержит прежние метки и дополнительные пары logit_i/probability_i для каждого класса. Порядок i задан в score_export.class_index внутри JSON. logits.npy сохраняет исходные выходы модели, probabilities.npy — устойчиво вычисленный softmax.

record_predictions.csv содержит решение по большинству, mean_logit_i, mean_probability_i и vote_share_i. Средние оценки описывают сегменты записи и не заменяют голосование; vote_share — доля голосов, а не откалиброванная вероятность.

## Проверки

    python -m pytest tests -q
    python -m reporting.verify --prepared /path/to/prepared \
      --engine dataset/engine_2/engine.yml --output /path/to/new-functional

Исторические checkpoints с полем state и новые с state_dict загружаются через weights_only=True и строгую проверку параметров модели. Каталоги reporting-результатов атомарны, существующие результаты не перезаписываются.

## Границы доказательств

Локальный инференс двух ResNet seed42 полностью повторил старые предсказания. Это не новые пять обучений. Сегментный split допускает общие исходные записи между частями; двигатели состояния в engine_2 различны. Показатели не характеризуют автоматически перенос на новый двигатель.

Проверка центра пика восстанавливает вершину гауссовой компоненты по трём
наибольшим по модулю отсчётам разности синтетического и исходного спектров.
Для поддержки ±4 Гц допуск равен 0.08 Гц. В проверке версии 0.1.3 все
1700 измерений на 17 частотах уложились в допуск; наибольшая ошибка
составила 0.0000222 Гц. В manifest записывается peak_position.
Параметр --peak-position configured применяет параметры положения из
переданной конфигурации. Обучение и оценка используют конфигурацию,
сохранённую вместе с соответствующей моделью.
