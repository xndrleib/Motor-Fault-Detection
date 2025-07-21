"""
src.pipeline
End-to-end time-series segmentation and FFT pipeline
====================================================

This module provides utilities to load raw current measurements,
slice them into overlapping frames, convert each frame to a
magnitude spectrum through a real-valued Fast Fourier Transform
(RFFT), and save both the spectra and rich metadata for downstream
machine-learning tasks such as fault diagnosis.

The core steps are:

1. **Segmentation** – :func:`sliding_segment` turns a 1-D signal into a
   sequence of (optionally windowed) overlapping frames.

2. **Spectral analysis** – :func:`rfft_mag` converts many frames at once
   into magnitude spectra and optionally expresses them in the decibel
   scale.

3. **Dataset orchestration** – :func:`preprocessing` walks over a folder
   hierarchy produced by the electric-motor benchmark, applies the
   segmentation + FFT pipeline to every measurement, and stores the
   result alongside a per-frame metadata table.

Examples
--------
>>> from pathlib import Path
>>> from src import pipeline as pl
>>> root = Path("../dataset/engine_2")
>>> meta = pl.create_metadata_df(root)
>>> seg, seg_meta, freqs = pl.preprocessing(
...     meta,
...     out_dir=root / "artefacts",
...     segment_length=10_000,
...     step=20,
...     fs=4096,
...     cutoff_hz=250,
... )
>>> seg.shape

Notes
-----
The implementation is streaming-friendly: heavy numerical arrays are
created only once per measurement and can be deleted immediately
after persisting to disk.  The metadata table preserves the mapping
between a spectral window and its provenance.
"""

import glob
import logging
import os
from pathlib import Path
from typing import Any, List, Tuple, Union, Optional, Sequence

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy import fft as sp_fft
from scipy.signal import get_window
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def sliding_segment(
    signal: np.ndarray,
    segment_length: int,
    step: int,
    window: str | None = None,
) -> np.ndarray:
    """
    Slice a one-dimensional array into equidistant, possibly windowed,
    overlapping frames.

    Parameters
    ----------
    signal : ndarray of shape (n_samples,)
        Input signal. Must be one-dimensional.
    segment_length : int
        Length (in samples) of each frame.
    step : int
        Number of samples between the starts of consecutive frames.
    window : str or None, default=None
        Name of a SciPy window function to apply to every frame
        (e.g., ``'hann'``, ``'blackmanharris'``).  If *None* no
        tapering window is applied.

    Returns
    -------
    windows : ndarray of shape (n_frames, segment_length)
        A view into *signal* (and therefore zero-copy) containing
        all extracted frames.  If *window* is not *None* the returned
        array is a copy with the window applied.

    Raises
    ------
    ValueError
        If *signal* is not one-dimensional, *segment_length* exceeds
        *signal* length, or *step* is not strictly positive.

    Examples
    --------
    >>> x = np.arange(10)
    >>> sliding_segment(x, segment_length=4, step=2)
    array([[0, 1, 2, 3],
           [2, 3, 4, 5],
           [4, 5, 6, 7]])
    """
    if signal.ndim != 1:
        raise ValueError("`signal` must be 1-D.")
    if segment_length > signal.size:
        raise ValueError("`segment_length` longer than signal.")
    if step <= 0:
        raise ValueError("`step` must be positive.")

    windows = np.lib.stride_tricks.sliding_window_view(
        signal, window_shape=segment_length
    )[::step]

    if window:
        win = get_window(window, segment_length, fftbins=True).astype(signal.dtype)
        windows = windows * win  # broadcast

    return windows


def rfft_mag(
    frames: np.ndarray,
    fs: float,
    cutoff_hz: float,
    to_db: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the magnitude of the one-sided (real) FFT for a batch of
    frames.

    The transformation is vectorised along the first axis and therefore
    much faster than looping in pure Python.

    Parameters
    ----------
    frames : ndarray of shape (n_frames, n_samples)
        Time-domain frames, typically produced by :func:`sliding_segment`.
    fs : float
        Sampling frequency in Hertz.
    cutoff_hz : float
        Upper frequency (exclusive) to retain.  All bins at
        ``f >= cutoff_hz`` are discarded.
    to_db : bool, default=True
        If *True* convert the magnitude to a decibel (20·log10) scale.
        A machine-epsilon offset is added to avoid :pyfunc:`numpy.log10`
        singularities.

    Returns
    -------
    mag : ndarray of shape (n_frames, n_freqs)
        Magnitude (linear or dB) spectra.
    freqs : ndarray of shape (n_freqs,)
        Centre frequency of each column in *mag*.

    Raises
    ------
    ValueError
        If *frames* is not two-dimensional.

    Examples
    --------
    >>> frames = np.random.randn(8, 1024)
    >>> mag, freqs = rfft_mag(frames, fs=4096, cutoff_hz=1000)
    >>> mag.shape
    (8, 251)
    """
    if frames.ndim != 2:
        raise ValueError("`frames` must be 2-D.")

    n_samples = frames.shape[1]
    freqs_full = sp_fft.rfftfreq(n_samples, d=1.0 / fs)
    keep = freqs_full < cutoff_hz

    spectra = sp_fft.rfft(frames, axis=1)[:, keep]
    mag = np.abs(spectra)

    if to_db:
        eps = np.finfo(np.float64).eps
        mag = 20.0 * np.log10(mag + eps)

    freqs = freqs_full[keep].astype(np.float32, copy=False)
    return mag, freqs


def _leading_trailing_run(mask: np.ndarray, *, from_start: bool = True) -> int:
    """Length of a leading (or trailing) True run in *mask*."""
    if not mask.any():
        return 0
    it = range(len(mask)) if from_start else range(len(mask) - 1, -1, -1)
    run = 0
    for idx in it:
        if mask[idx]:
            run += 1
        else:
            break
    return run


def _load_measurement(
    path: Union[str, Path],
    *,
    header: int = 0,
    use_cols: Optional[Sequence[int]] = None,
    scale_time: float | int = 1.0,
    tiny_gap_limit: int = 2,
    clock_tolerance: float = 0.01,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Load, validate and tiny‑gap‑impute a sensor measurement CSV.

    Parameters
    ----------
    path
        Path to the measurement file (CSV).
    header
        Row number for the header (passed to :pyfunc:`pandas.read_csv`).
    use_cols
        Column indices to read.  Defaults to the first two columns ``[0, 1]``.
    scale_time
        Factor multiplying the *Time* column (e.g. ms → s).
    tiny_gap_limit
        Maximum consecutive NaNs to patch with PCHIP interpolation.
    clock_tolerance
        Allowed relative deviation from the median sampling interval.
    logger
        Optional :pyclass:`logging.Logger`.  Falls back to
        ``logging.getLogger(__name__)``.

    Returns
    -------
    pd.DataFrame
        Index: *float64* time grid, strictly increasing.
        Column ``'Current'``: *float64*.
        Any gap longer than *tiny_gap_limit* remains ``NaN``.
    """

    use_cols = [0, 1] if use_cols is None else list(use_cols)
    logger = logger or logging.getLogger(__name__)
    path = Path(path).expanduser().resolve()

    # ---------- 1. CSV ingest ----------
    try:
        df = pd.read_csv(
            path,
            header=header,
            names=["Time", "Current"],
            usecols=use_cols,
            dtype={"Time": "float64", "Current": "float64"},
        )
    except Exception as e:
        raise RuntimeError(f"Failed to read {path}: {e}") from e

    # ---------- 2. Basic schema validation ----------
    df["Time"] *= scale_time
    df.set_index("Time", inplace=True)
    df.sort_index(inplace=True)

    if not df.index.is_monotonic_increasing:
        raise ValueError(
            f"{path}: Time column must be strictly increasing after scaling."
        )

    # ---------- 3. Handle leading NaNs by dropping them ----------
    leading_run = _leading_trailing_run(
        df["Current"].isna().to_numpy(), from_start=True
    )
    if leading_run:
        dropped_times = df.index[:leading_run].tolist()
        df = df.iloc[leading_run:]
        logger.info(
            f"{path}: dropped {leading_run} leading NaN row(s) at times {dropped_times}."
        )

    # ---------- 4. Diagnostics for remaining NaNs ----------
    missing_mask = df["Current"].isna()
    if missing_mask.any():
        rows = np.where(missing_mask)[0].tolist()
        times = df.index[missing_mask].tolist()
        logger.warning(
            f"{path}: {len(rows)} NaN(s) remain at rows/time -> {list(zip(rows, times))}"
        )

    # ---------- 5. Uniform‑clock check ----------
    deltas = np.diff(df.index.values)
    if deltas.size >= 2:
        median_step = np.median(deltas)
        jitter_mask = np.abs(deltas - median_step) > clock_tolerance * median_step
        if jitter_mask.any():
            # Row positions are offset by +1 because deltas are between rows i‑1 and i
            row_jitter = (np.where(jitter_mask)[0] + 1).tolist()
            time_jitter = df.index[row_jitter].tolist()
            max_dev = np.abs(deltas[jitter_mask]).max()
            logger.warning(
                f"{path}: non‑uniform sampling (max deviation {max_dev:.3g}); "
                f"offending rows/time -> {list(zip(row_jitter, time_jitter))}"
            )

    # ---------- 6. Tiny‑gap interpolation ----------
    before = missing_mask.sum()
    if before:
        df["Current"] = df["Current"].interpolate(
            method="pchip",
            limit=tiny_gap_limit,
            limit_direction="both",
            limit_area="inside",
        )
        after = df["Current"].isna().sum()
        patched = before - after
        if patched:
            logger.info(f"{path}: filled {patched} tiny‑gap NaNs with PCHIP.")
        if after:
            logger.warning(f"{path}: {after} NaNs remain (gaps > {tiny_gap_limit}).")

    return df


def load_measurement(
    measurement_id: str, meta: pd.DataFrame, load_cfg: dict | None = None
) -> pd.Series:
    """
    Load a single measurement CSV file and return its *Current* signal.

    Parameters
    ----------
    measurement_id : str
        Index label identifying the row in *meta* corresponding to the
        desired measurement.
    meta : DataFrame
        Metadata table as returned by :func:`create_metadata_df`. Must
        contain a ``'file_path'`` column.

    Returns
    -------
    Series
        Time-indexed current signal with name ``'Current'``.

    Notes
    -----
    * Only the last non-all-NaN data column of the CSV is kept.
    * If the CSV file has no index name, it is set to ``'Time'``.
    """
    path = Path(meta.at[measurement_id, "file_path"])

    # todo: add load_cfg
    df = _load_measurement(path)
    return df["Current"]


def process_file(
    file_path: str | Path,
    *,
    segment_length: int,
    step: int,
    fs: int,
    cutoff_hz: float,
    window: str | None = None,
    to_db: bool = True,
    load_cfg: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return FFT frames for a single measurement file.

    Parameters
    ----------
    file_path
        Path to the measurement CSV.
    segment_length
        Frame size in samples.
    step
        Hop size in samples.
    fs
        Sampling rate in Hertz.
    cutoff_hz
        Highest frequency to keep (see :func:`rfft_mag`).
    window
        Optional window function passed to :func:`sliding_segment`.
    to_db
        Output magnitude in dB when *True*.

    Returns
    -------
    fft_frames
        Magnitude spectra of shape ``(n_frames, n_freqs)``.
    freqs
        Frequency axis shared by all frames.
    """
    if load_cfg:
        df = _load_measurement(Path(file_path), **load_cfg)
    else:
        df = _load_measurement(Path(file_path))
    sig = df["Current"].values
    frames = sliding_segment(sig, segment_length, step, window=window)
    fft_frames, freqs = rfft_mag(frames, fs, cutoff_hz, to_db)
    return fft_frames, freqs


def process_measurement(
    m_id: str,
    meta_row: pd.Series,
    *,
    segment_length: int,
    step: int,
    fs: int,
    cutoff_hz: float,
    window: str | None = None,
    to_db: bool = True,
) -> tuple[np.ndarray, list[dict], np.ndarray]:
    """Return FFT frames and metadata for one measurement.

    Parameters
    ----------
    m_id : str
        Measurement identifier (index label in ``metadata_df``).
    meta_row : Series
        Row from the metadata table corresponding to ``m_id``.
    segment_length : int
        Frame size in samples.
    step : int
        Hop size in samples.
    fs : int
        Sampling rate in Hertz.
    cutoff_hz : float
        Highest frequency to keep (see :func:`rfft_mag`).
    window : str or None, optional
        Optional window function passed to :func:`sliding_segment`.
    to_db : bool, default=True
        Output magnitude in dB when *True*.

    Returns
    -------
    fft_frames : ndarray of shape (n_frames, n_freqs)
        Magnitude spectra of the measurement.
    rows : list of dict
        Per-segment metadata rows for ``fft_frames``.
    freqs : ndarray of shape (n_freqs,)
        Frequency axis shared by all frames.
    """

    df = _load_measurement(Path(meta_row["file_path"]))
    current = df["Current"]
    frames = sliding_segment(current.values, segment_length, step, window=window)
    fft_frames, freqs = rfft_mag(frames, fs, cutoff_hz, to_db)

    rows = [
        {
            "measurement_id": m_id,
            "base_id": meta_row["base_id"],
            "segment_idx": i,
            "start_time": current.index[i * step],
            "end_time": current.index[i * step + segment_length - 1],
            "state": meta_row["state"],
            "binary_label": 0 if meta_row["state"] == "normal" else 1,
            "multiclass_label": meta_row["state"],
            "phase": int(meta_row["phase"]),
            "load_condition": meta_row["load_condition"],
            "experiment": meta_row["experiment"],
        }
        for i in range(fft_frames.shape[0])
    ]

    return fft_frames, rows, freqs


def preprocessing(
    metadata_df: pd.DataFrame,
    out_dir: str | Path,
    segment_length: int,
    step: int,
    fs: int,
    cutoff_hz: float,
    window: str | None = None,
    to_db: bool = True,
    intermediate_dir: str | Path | None = None,
    experiment: Any | None = None,
) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Run the segmentation + FFT pipeline for **all** measurements in
    *metadata_df* and serialise the result.

    For every measurement the function:

    1. Reads the CSV (via :func:`load_measurement`).
    2. Splits the signal into overlapping frames.
    3. Computes the magnitude spectrum of each frame.
    4. Appends spectra and per-frame metadata to in-memory lists.
    5. Saves three artefacts to *out_dir*:
       ``segments.npy``, ``freqs.npy``, ``segments_metadata.csv``.

    Parameters
    ----------
    metadata_df : DataFrame
        Metadata table produced by :func:`create_metadata_df`.
    out_dir : str or Path
        Destination directory for the persisted artefacts.
    segment_length : int
        Frame size in samples.
    step : int
        Hop size in samples.
    fs : int
        Sampling rate of the raw signal in Hertz.
    cutoff_hz : float
        Highest frequency to keep (see :func:`rfft_mag`).
    window : str or None, default=None
        Optional window function (passed to :func:`sliding_segment`).
    to_db : bool, default=True
        Output magnitude in dB when *True*.
    intermediate_dir : str or Path or None, optional
        If given, save per-measurement FFT arrays and metadata inside this
        directory.
    experiment : optional
        Comet experiment for logging statistics.

    Returns
    -------
    segments : ndarray of shape (N, n_freqs)
        Stack of all magnitude spectra in the same order as *seg_meta*.
    seg_meta : DataFrame
        Per-segment metadata indexed by ``['measurement_id', 'segment_idx']``.
    freqs : ndarray of shape (n_freqs,)
        Frequency axis shared by every row in *segments*.

    Examples
    --------
    >>> segments, seg_meta, freqs = preprocessing(
    ...     metadata_df, "./artefacts", 10_000, 20, 4096, 250
    ... )
    """
    seg_rows: list[dict] = []
    seg_arrays: list[np.ndarray] = []
    freqs: np.ndarray | None = None

    if intermediate_dir is not None:
        intermediate_dir = Path(intermediate_dir)
        intermediate_dir.mkdir(parents=True, exist_ok=True)

    if experiment is not None:
        experiment.log_parameters(
            {
                "segment_length": segment_length,
                "step": step,
                "fs": fs,
                "cutoff_hz": cutoff_hz,
                "window": window,
                "to_db": to_db,
            }
        )

    for m_id, meta in tqdm(
        metadata_df.iterrows(), total=len(metadata_df), desc="FFT-preprocessing"
    ):
        fft_frames, rows, freqs = process_measurement(
            m_id,
            meta,
            segment_length=segment_length,
            step=step,
            fs=fs,
            cutoff_hz=cutoff_hz,
            window=window,
            to_db=to_db,
        )

        seg_rows.extend(rows)
        seg_arrays.append(fft_frames)

        if intermediate_dir is not None:
            np.save(intermediate_dir / f"{m_id}.npy", fft_frames)
            pd.DataFrame(rows).to_csv(
                intermediate_dir / f"{m_id}_meta.csv", index=False
            )

    segments = np.vstack(seg_arrays).astype(np.float32)
    seg_meta = pd.DataFrame(seg_rows).set_index(["measurement_id", "segment_idx"])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "segments.npy", segments)
    np.save(out_dir / "freqs.npy", freqs)
    seg_meta.to_csv(out_dir / "segments_metadata.csv")

    if experiment is not None:
        experiment.log_metric("num_segments", segments.shape[0])

    return segments, seg_meta, freqs


def create_metadata_df(
    base_dir: str | Path, state2name: dict | None = None
) -> pd.DataFrame:
    """
    Scan the dataset folder hierarchy and build a per-measurement
    metadata table.

    The expected folder layout is compatible with the *Electro-Mechanical
    Fault Detection* benchmark::

        base_dir/
        └── experiment_*/
            └── current/
                └── <load_condition>/
                    └── <phase>/
                        └── *.csv

    Parameters
    ----------
    base_dir : str or Path
        Root directory of the dataset.
    state2name : dict, optional
        Mapping from integer state labels (extracted from the
        ``<load_condition>`` folder) to human-readable names.

    Returns
    -------
    DataFrame
        Table with one row per measurement and index ``measurement_id``.
        Columns include (but are not limited to)::

            'experiment', 'state', 'load_condition', 'phase',
            'num_observations', 'file_path', 'engine_cfg_path'
    """
    base_dir = Path(base_dir)
    csv_pattern = os.path.join(base_dir, "experiment_*", "current", "*", "*", "*.csv")
    csv_files = glob.glob(csv_pattern)

    metadata_list = []

    for file_path in csv_files:
        # Expected structure:
        # base_dir/experiment_*/current/<load_condition>/<phase>/<filename>.csv
        parts = file_path.split(os.sep)
        try:
            experiment = parts[-5]  # e.g., "experiment_1"
            load_folder = parts[-3]  # e.g., "1st_load_80"
            phase = parts[-2]  # e.g., "1"
            base_id = os.path.splitext(os.path.basename(file_path))[0]
            measurement_id = f"{base_id}_{phase}"
        except IndexError as e:
            logger.error(
                "File path %s does not match expected structure: %s", file_path, e
            )
            continue

        # Parse the load_folder to extract state and load_condition.
        try:
            load_parts = load_folder.split("_")
            if len(load_parts) >= 3:
                state = load_parts[0][0]  # e.g., from "1st" take '1'
                load_condition = load_parts[-1]  # e.g., "80"
            else:
                state = None
                load_condition = load_folder
        except Exception as e:
            logger.error(
                "Error parsing load folder %s in file %s: %s", load_folder, file_path, e
            )
            state, load_condition = None, None

        # Read the CSV file to count the number of observations.
        try:
            df = pd.read_csv(file_path, header=0, index_col=0)
        except Exception as e:
            logger.error("Error reading file %s: %s", file_path, e)
            continue

        # Drop any columns that are completely NaN (e.g., from trailing delimiters).
        df = df.dropna(axis=1, how="all")

        num_observations = df.shape[0]

        if state2name is not None:
            try:
                state = state2name[int(state)]
            except KeyError:
                logger.warning("Fault type %s not found in mapping.", state)
                state = None

        # Create the metadata entry.
        meta_entry = {
            "measurement_id": measurement_id,
            "base_id": base_id,
            "experiment": experiment,
            "state": state,
            "load_condition": load_condition,
            "phase": phase,
            "num_observations": num_observations,
            "file_path": file_path,
            "engine_cfg_path": base_dir / "engine.yml",
        }
        metadata_list.append(meta_entry)

    if metadata_list:
        metadata_df = pd.DataFrame(metadata_list)
        metadata_df.set_index("measurement_id", inplace=True)
        metadata_df.sort_index(inplace=True)
    else:
        metadata_df = pd.DataFrame(
            columns=[
                "experiment",
                "state",
                "load_condition",
                "phase",
                "num_observations",
                "file_path",
                "engine_cfg_path",
            ]
        )

    return metadata_df


def filter_segments(
    seg_meta: pd.DataFrame,
    segments: np.ndarray,
    states: List[str],
    loads: List[str],
    phases: List[str],
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Select a subset of spectral windows and keep the array and metadata
    in sync.

    Parameters
    ----------
    seg_meta : DataFrame
        Metadata table produced by :func:`preprocessing`.
    segments : ndarray of shape (N, n_freqs)
        Magnitude spectra aligned with *seg_meta*.
    states : list of str
        Desired fault states.
    loads : list of str
        Desired load conditions.
    phases : list of str
        Desired motor phases.

    Returns
    -------
    seg_meta_filt : DataFrame
        Filtered metadata with consecutive integer index.
    segments_filt : ndarray
        Spectra whose rows correspond to *seg_meta_filt*.

    Examples
    --------
    >>> seg_meta_sub, seg_sub = filter_segments(
    ...     seg_meta, segments,
    ...     states=['normal', '1'],
    ...     loads=['80'],
    ...     phases=['1']
    ... )
    """
    mask = (
        seg_meta["state"].isin(states)
        & seg_meta["load_condition"].isin(loads)
        & seg_meta["phase"].isin(phases)
    )
    return seg_meta.loc[mask].reset_index(drop=True), segments[mask.values]
