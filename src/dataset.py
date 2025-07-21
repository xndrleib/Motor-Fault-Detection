"""Frankenstein spectrogram dataset for PyTorch.

This module provides a dataset that loads pre–FFT segments produced by
:func:`src.pipeline.preprocessing`, interpolates them onto a **canonical
frequency grid**, and globally normalises the resulting magnitudes.

Quick-start
-----------
>>> from pathlib import Path
>>> from src.dataset import FrankensteinDataset
>>> root = Path("../dataset/engine_2/artefacts/")            # contains segments.npy etc.
>>> ds = FrankensteinDataset(root, f_out=512)
>>> spec, load, label = ds[0]            # first sample
>>> spec.shape
torch.Size([512])

Public API
----------
`interpolate_canonical`, `FrankensteinDataset`
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.normalizer import Method, Mode, Normalizer

__all__ = ["interpolate_canonical", "FrankensteinDataset"]


def interpolate_canonical(
    spectra: np.ndarray,
    freqs: np.ndarray,
    f_cut: float,
    f_out: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate magnitude spectra onto a canonical frequency grid.

    Parameters
    ----------
    spectra : ndarray, shape (N, F)
        Batch of magnitude spectra. **Must be 2-D** and of dtype
        ``float32`` or convertible to it.
    freqs : ndarray, shape (F,)
        One-dimensional frequency axis (Hz) corresponding to the columns
        of `spectra`.
    f_cut : float
        Upper frequency limit (Hz) to keep after interpolation.
    f_out : int
        Desired number of frequency bins in the output grid.

    Returns
    -------
    spectra_interp : ndarray, shape (N, f_out)
        Spectra resampled onto the canonical grid.
    freq_axis : ndarray, shape (f_out,)
        The canonical, linearly spaced frequency axis.

    Raises
    ------
    ValueError
        If input shapes do not match the requirements.

    See Also
    --------
    numpy.interp : Underlying interpolation routine.

    Notes
    -----
    The canonical grid is produced by::

        np.linspace(0.0, f_cut, f_out, endpoint=False, dtype=np.float32)

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> spectra = np.random.rand(3, 1024).astype(np.float32)
    >>> freqs = np.linspace(0, 20_000, 1024, dtype=np.float32)
    >>> spec_i, f_axis = interpolate_canonical(spectra, freqs, f_cut=10_000, f_out=512)
    >>> spec_i.shape
    (3, 512)
    """
    if spectra.ndim != 2:
        raise ValueError("`spectra` must be 2-D (N, F)")
    if freqs.ndim != 1:
        raise ValueError("`freqs` must be 1-D (F,)")
    if spectra.shape[1] != freqs.size:
        raise ValueError("Dimension mismatch between `spectra` and `freqs`")

    f_star = np.linspace(0.0, f_cut, f_out, endpoint=False, dtype=np.float32)
    out = np.empty((spectra.shape[0], f_out), dtype=np.float32)
    for i, row in enumerate(spectra):
        out[i] = np.interp(f_star, freqs, row).astype(np.float32)
    return out, f_star


class FrankensteinDataset(Dataset):
    """
    PyTorch dataset yielding **normalised** magnitude spectra.

    The dataset expects three artifact files inside *artefact_dir*:

    * ``segments.npy`` – raw FFT magnitude segments, shape (N, F)
    * ``freqs.npy`` – frequency axis, shape (F,)
    * ``segments_metadata.csv`` – per-segment metadata with at least the
      columns ``state`` and ``load_condition``

    During initialisation the following preprocessing steps are applied:

    1. Interpolation to a canonical grid via :func:`interpolate_canonical`.
    2. Global min-max scaling to :math:`[0, 1]` using
       :class:`src.normalizer.Normalizer`.

    Parameters
    ----------
    artefact_dir : str or Path
        Directory containing the three artifact files listed above.
    f_out : int, default=512
        Number of frequency bins after interpolation.

    Notes
    -----
    *Label encoding*: every unique ``state`` value in the metadata
    column is mapped to an integer starting at 0, stored in
    ``self.state2idx``.

    Examples
    --------
    Basic usage:

    >>> from src.dataset import FrankensteinDataset
    >>> root = Path("../dataset/engine_2/artefacts/")
    >>> ds = FrankensteinDataset(root, f_out=512)
    >>> spec, load, label = ds[0]
    >>> spec.shape[0]
    512
    """

    def __init__(self, artefact_dir: str | Path, f_out: int = 512) -> None:
        self.artefact_dir = Path(artefact_dir)
        seg_path = self.artefact_dir / "segments.npy"
        freq_path = self.artefact_dir / "freqs.npy"
        meta_path = self.artefact_dir / "segments_metadata.csv"

        self._segments = np.load(seg_path).astype(np.float32)
        freqs = np.load(freq_path).astype(np.float32)
        self.meta = pd.read_csv(meta_path)

        # Step 1: canonical interpolation
        self._segments, self.freqs = interpolate_canonical(
            self._segments, freqs, freqs.max(), f_out
        )

        # Step 2: global min-max normalisation
        self.norm = Normalizer(method=Method.MIN_MAX, mode=Mode.GLOBAL)
        self._segments = self.norm.fit_transform(self._segments)

        # Label mapping
        states = sorted(self.meta["state"].unique())
        self.state2idx = {s: i for i, s in enumerate(states)}

    # ------------------------------------------------------------------ #
    # PyTorch dataset interface
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:  # pragma: no cover
        """
        int: Number of FFT segments in the dataset.
        """
        return self._segments.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float, int]:
        """
        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        spec : torch.Tensor, shape (f_out,)
            Normalised magnitude spectrum on the canonical grid.
        load : float
            Load condition normalised to :math:`[0, 1]`.
        label : int
            Integer-encoded machine state.
        """
        row = self.meta.iloc[idx]
        load = float(row["load_condition"]) / 100.0
        label = self.state2idx[row["state"]]
        spec = torch.from_numpy(self._segments[idx])
        return spec, load, label
