"""Noise policy utilities for time-domain augmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def add_awgn(signal: np.ndarray, snr_db: float, rng: np.random.Generator, eps: float) -> np.ndarray:
    """Add AWGN to a 1D signal at a target SNR (dB).

    Parameters
    ----------
    signal : np.ndarray
        1D time-domain signal.
    snr_db : float
        Target SNR in dB, defined against the recorded signal power.
    rng : np.random.Generator
        RNG for reproducible noise generation.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    np.ndarray
        Noisy signal with additive Gaussian noise.
    """
    power = float(np.mean(signal ** 2))
    power = max(power, eps)
    noise_power = power / (10 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=signal.shape)
    return signal + noise


@dataclass(frozen=True)
class SNRStage:
    """Curriculum stage for SNR range scheduling."""

    start_pct: float
    end_pct: float
    snr_min: float
    snr_max: float


class NoisePolicy:
    """Noise sampling policy for time-domain augmentation."""

    def __init__(
        self,
        *,
        p_clean: float = 0.3,
        snr_min: float = 20.0,
        snr_max: float = 40.0,
        distribution: str = "uniform",
        knee_range: Optional[Tuple[float, float]] = None,
        knee_weight: float = 0.5,
        bin_edges: Optional[Iterable[float]] = None,
        bin_weights: Optional[Iterable[float]] = None,
        curriculum: Optional[List[SNRStage]] = None,
        eps: float = 1e-12,
    ) -> None:
        self.p_clean = float(p_clean)
        self.snr_min = float(snr_min)
        self.snr_max = float(snr_max)
        self.distribution = str(distribution).lower()
        self.knee_range = knee_range
        self.knee_weight = float(knee_weight)
        self.bin_edges = list(bin_edges) if bin_edges is not None else None
        self.bin_weights = list(bin_weights) if bin_weights is not None else None
        self.curriculum = curriculum or []
        self.eps = float(eps)

        if not (0.0 <= self.p_clean <= 1.0):
            raise ValueError("p_clean must be between 0 and 1.")
        if self.snr_max < self.snr_min:
            raise ValueError("snr_max must be >= snr_min.")
        if self.distribution not in {"uniform", "mixture", "bin_weighted"}:
            raise ValueError(
                "distribution must be one of: 'uniform', 'mixture', 'bin_weighted'."
            )
        if self.distribution == "mixture":
            if self.knee_range is None or len(self.knee_range) != 2:
                raise ValueError("mixture distribution requires knee_range=(min, max).")
            if not (0.0 <= self.knee_weight <= 1.0):
                raise ValueError("knee_weight must be between 0 and 1.")
        if self.distribution == "bin_weighted":
            if not self.bin_edges or not self.bin_weights:
                raise ValueError("bin_weighted requires bin_edges and bin_weights.")
            if len(self.bin_edges) != len(self.bin_weights) + 1:
                raise ValueError("bin_edges must have length len(bin_weights) + 1.")
            if any(b2 <= b1 for b1, b2 in zip(self.bin_edges, self.bin_edges[1:])):
                raise ValueError("bin_edges must be strictly increasing.")

    @classmethod
    def from_config(cls, cfg: Dict[str, object]) -> "NoisePolicy":
        """Construct a NoisePolicy from a config dictionary."""
        p_clean = float(cfg.get("p_clean", 0.3))

        snr_cfg = cfg.get("snr_db", {}) or {}
        snr_min = float(snr_cfg.get("snr_min", 20.0))
        snr_max = float(snr_cfg.get("snr_max", 40.0))
        distribution = str(snr_cfg.get("distribution", "uniform")).lower()
        knee_range = snr_cfg.get("knee_range", None)
        knee_weight = float(snr_cfg.get("knee_weight", 0.5))

        bin_edges = snr_cfg.get("bin_edges", None)
        bin_weights = snr_cfg.get("bin_weights", None)

        curriculum_cfg = cfg.get("curriculum", {}) or {}
        curriculum = []
        if curriculum_cfg.get("enabled", False):
            for stage in curriculum_cfg.get("schedule", []):
                curriculum.append(
                    SNRStage(
                        start_pct=float(stage["start_pct"]),
                        end_pct=float(stage["end_pct"]),
                        snr_min=float(stage["snr_min"]),
                        snr_max=float(stage["snr_max"]),
                    )
                )

        eps = float(cfg.get("epsilon", 1e-12))
        return cls(
            p_clean=p_clean,
            snr_min=snr_min,
            snr_max=snr_max,
            distribution=distribution,
            knee_range=tuple(knee_range) if knee_range is not None else None,
            knee_weight=knee_weight,
            bin_edges=bin_edges,
            bin_weights=bin_weights,
            curriculum=curriculum,
            eps=eps,
        )

    def _snr_bounds(self, epoch: int, total_epochs: Optional[int]) -> Tuple[float, float]:
        if not self.curriculum or total_epochs is None or total_epochs <= 0:
            return self.snr_min, self.snr_max

        progress = (epoch - 1) / float(total_epochs)
        for stage in self.curriculum:
            if stage.start_pct <= progress < stage.end_pct:
                return stage.snr_min, stage.snr_max
        # fallback to last stage if progress >= end
        last = self.curriculum[-1]
        return last.snr_min, last.snr_max

    def _sample_snr_db(
        self, rng: np.random.Generator, epoch: int, total_epochs: Optional[int]
    ) -> float:
        snr_min, snr_max = self._snr_bounds(epoch, total_epochs)

        if self.distribution == "uniform":
            return float(rng.uniform(snr_min, snr_max))

        if self.distribution == "mixture":
            knee_min, knee_max = self.knee_range  # type: ignore[misc]
            if rng.random() < self.knee_weight:
                return float(rng.uniform(knee_min, knee_max))
            return float(rng.uniform(snr_min, snr_max))

        # bin_weighted
        weights = np.array(self.bin_weights, dtype=float)
        weights = weights / weights.sum()
        bin_idx = int(rng.choice(len(weights), p=weights))
        low = float(self.bin_edges[bin_idx])
        high = float(self.bin_edges[bin_idx + 1])
        return float(rng.uniform(low, high))

    def apply(
        self,
        signal: np.ndarray,
        *,
        rng: np.random.Generator,
        epoch: int,
        total_epochs: Optional[int],
    ) -> Tuple[np.ndarray, Optional[float]]:
        """Apply noise policy to a signal.

        Returns the (possibly) noisy signal and the sampled SNR in dB, or
        ``None`` if the signal was kept clean.
        """
        if rng.random() < self.p_clean:
            return signal, None

        snr_db = self._sample_snr_db(rng, epoch, total_epochs)
        noisy = add_awgn(signal, snr_db, rng=rng, eps=self.eps)
        return noisy, snr_db
