"""Local-only experiment logging used by the reporting release."""
import json
from pathlib import Path


class LocalExperiment:
    def __init__(self, path):
        self.path = Path(path)

    def _write(self, event, data):
        with self.path.open("a") as stream:
            stream.write(json.dumps({"event": event, "data": data}, default=str) + "\n")

    def log_parameter(self, name, value):
        self._write("parameter", {name: value})

    def log_parameters(self, values):
        self._write("parameters", values)

    def log_metric(self, name, value, **kwargs):
        self._write("metric", {name: value, **kwargs})

    def log_metrics(self, values, **kwargs):
        self._write("metrics", {**values, **kwargs})

    def log_figure(self, **kwargs):
        self._write("figure", {"name": kwargs.get("figure_name")})

    def log_asset(self, path, **kwargs):
        self._write("asset", {"path": str(path), **kwargs})

    def log_model(self, **kwargs):
        self._write("model", kwargs)

    def end(self):
        self._write("end", {})
