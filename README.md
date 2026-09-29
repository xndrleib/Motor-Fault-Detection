# SGDA MotorDiag reporting release

Version 0.1.2. This branch packages the existing SGDA implementation for reproducible
preparation, synthesis and inference. The evaluated benchmark uses Engine 2,
phase 1, load 100%, and Normal / ITSC / RBD.

## Installation

Use Python 3.11 and the dependency profile for the target environment:

~~~sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-reporting-linux.txt
python -m pip install --no-deps .
python -m reporting.cli --help
~~~

For the verified macOS CPU environment use requirements-reporting-macos.txt.
The Linux profile records the available cluster environment; a successful
clean-install/target-platform check must be recorded separately. Windows has
not yet been validated.

## Data and saved models

Source code and scientific assets are supplied separately. The asset archive
contains sgda-assets/prepared, sgda-assets/runs/binary-seed42 (and other seeds),
sgda-assets/runs/multiclass-seed42, and the selected Engine 2 raw files.
Run commands from this code directory and replace the example absolute paths.

~~~sh
python -m reporting.cli predict --prepared /path/to/sgda-assets/prepared --run /path/to/sgda-assets/runs/binary-seed42 --device cpu --output /path/to/new-results/binary
python -m reporting.cli prepare --metadata /path/to/sgda-assets/dataset/engine_2/metadata.csv --config training_configs/reporting/binary.yaml --path-base /path/to/sgda-assets/experiments --missing-current legacy-drop --time-axis engine2-legacy --output /path/to/new-results/prepared
~~~

The legacy time-axis profile verifies the original Engine 2 millisecond
coordinate at 4096 samples/s while preserving the historical FFT configuration
of 4098 Hz. It does not resample or change the model input. Ordinary new inputs
use seconds by default; declare --time-axis milliseconds when appropriate.

## Commands and outputs

- prepare validates input formats and time coordinates, then invokes the existing FFT pipeline.
- synthesize exports SGDA spectra, configuration metadata and plots.
- predict restores an existing test split and exports metrics, segment scores and majority votes.
- python -m reporting.diagnose classifies a new unlabelled current file.
- python -m experiments.train --offline runs the existing training workflow.

Prediction CSV files include logit_i and probability_i in the class order
recorded by the JSON receipt. Record summaries contain mean scores and vote
fractions; the decision remains majority voting. Softmax values are not a
claim of empirical probability calibration.

See [the Russian usage guide](docs/reporting/usage.md) for detailed commands
and artifact contracts. Existing checkpoints, split indices and normalizers
must remain together.

## Verification

~~~sh
python -m pytest tests -q
~~~

The reporting release preserves the historical scientific method. A successful
replay verifies the saved benchmark; the acceptance protocol identifies the
specific data, model, preprocessing and metric scope.
