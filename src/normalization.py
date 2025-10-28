# normalization.py
import numpy as np
import json
from pathlib import Path

class Normalizer:
    """
    Handles both *per-sample* and *global* normalisation.

    mode='per'    : z-score or min-max *per window*
    mode='global' : one μ/σ (or min/max) fitted on an array of windows
    """
    def __init__(self, method='min-max', mode='global'):
        assert method in ('z-score', 'min-max')
        assert mode   in ('global', 'per')
        self.method, self.mode = method, mode
        self.stats = None                     # populated by .fit()

    def fit(self, windows: np.ndarray) -> "Normalizer":
        """Compute global statistics on *any* array of shape (N, L)."""
        if self.mode == 'per':
            raise RuntimeError("per-window mode doesn't need .fit()")
        if self.method == 'z-score':
            mu  = windows.mean()
            std = windows.std()
            self.stats = {'mean': float(mu), 'std': float(std)}
        else:
            lo  = np.percentile(windows,  2.5)   # robust to outliers
            hi  = np.percentile(windows, 97.5)
            self.stats = {'min': float(lo), 'max': float(hi)}
        return self

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.stats, indent=2))

    def load(self, path: str | Path) -> "Normalizer":
        self.stats = json.loads(Path(path).read_text())
        return self
    
    def _apply(self, x: np.ndarray) -> np.ndarray:
        if self.method == 'z-score':
            mu  = self.stats['mean']  if self.mode=='global' else x.mean()
            std = self.stats['std']   if self.mode=='global' else x.std()
            return (x - mu) / (std + 1e-8)
        else:                                      # min-max
            lo  = self.stats['min']   if self.mode=='global' else x.min()
            hi  = self.stats['max']   if self.mode=='global' else x.max()
            return (x - lo) / (hi - lo + 1e-8)

    def transform(self, windows: np.ndarray) -> np.ndarray:
        if self.mode == 'per':
            return np.stack([self._apply(w) for w in windows])
        return self._apply(windows)

    def fit_transform(self, windows: np.ndarray) -> np.ndarray:
        return self.fit(windows).transform(windows)
