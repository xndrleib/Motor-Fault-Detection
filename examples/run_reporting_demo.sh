#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo "Usage: bash examples/run_reporting_demo.sh SOURCE_REPO BINARY_RUN NEW_OUTPUT" >&2
  exit 2
fi
sgda_source="$1"
sgda_run="$2"
sgda_output="$3"
python -m reporting.cli prepare \
  --metadata "$sgda_source/dataset/engine_2/metadata.csv" \
  --config "$sgda_run/training_config.yaml" \
  --path-base "$sgda_source/experiments" \
  --missing-current legacy-drop \
  --time-axis engine2-legacy \
  --output "$sgda_output/prepared"
python -m reporting.cli synthesize \
  --prepared "$sgda_output/prepared" \
  --engine "$sgda_source/dataset/engine_2/engine.yml" \
  --fault "inter-turn short circuits" \
  --normalizer-path "$sgda_run/normalizer_binary.json" \
  --count 10000 --seed 42 --output "$sgda_output/synthetic"
python -m reporting.cli predict \
  --prepared "$sgda_output/prepared" --run "$sgda_run" \
  --device cpu --output "$sgda_output/evaluation"
