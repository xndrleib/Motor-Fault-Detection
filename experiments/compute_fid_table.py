#!/usr/bin/env python3
"""
Compute FID and normalized FID (nFID) between real and synthetic embeddings per
(fault_code, load), with support for multiple negative baselines. Produces a
**single detailed (wide) table** and saves it in **CSV and Parquet** formats.

Embeddings are expected to be produced by `embed_fid_from_run.py` into:
    res/fid_embs/<run_id>/*_embs.npy

Filename conventions
--------------------
    real_<LOAD>_<FAULTCODE>_embs.npy
    synth_<LOAD>_<FAULTCODE>_embs.npy

Normalization (nFID)
--------------------
For each (fault, load) we compute:
  • Floor (FID_RR): split-half real↔real distance.
  • Negative baselines (FID_RW):
      - 'real-other-fault'        : FID( real(f,L),  real(g≠f,L) )
      - 'real-synth-other-fault'  : FID( real(f,L), synth(g≠f,L) )
      - 'synth-other-fault'       : FID( synth(f,L), synth(g≠f,L) )

Then, for each chosen baseline:
    nFID = (FID_RS - FID_RR) / (FID_RW - FID_RR)

Outputs
-------
A single, detailed **wide** table saved twice next to the embeddings:
    1) <emb_dir>/<out-csv>      (CSV)
    2) <emb_dir>/<out-parquet>  (Parquet)

Columns include:
    ['run_id','fault_code','load','n_real','n_synth','dim','fid','fid_rr',
     'fid_rw__<mode1>','nfid__<mode1>',
     'fid_rw__<mode2>','nfid__<mode2>', ... ]

Notes
-----
• Robustness: split-half baseline uses repeated random splits; negative baselines can be
  aggregated across multiple "other faults" using a percentile (default: median).
• Important: FID is **not clamped** to be non-negative; small negative values due to
  round-off are preserved as-is.

Example
-------
python experiments/compute_fid_table.py \
  --run-dir res/runs/2025-06-03_02-14-40_train_full-data-removeES-42-16 \
  --emb-root res/fid_embs \
  --out-csv fid_table.csv \
  --out-parquet fid_table.parquet \
  --rr-repeats 10 --seed 123 \
  --neg-baselines real-other-fault real-synth-other-fault synth-other-fault \
  --neg-percentile 0.5
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Iterable

import numpy as np
import pandas as pd
from scipy import linalg as la


# =============================================================================
# Parsing utilities
# =============================================================================

STEM_REGEX = re.compile(
    r"^(?P<kind>real|synth)_(?P<load>\d+)_(?P<fault>[A-Za-z0-9]+)_embs$",
    re.IGNORECASE,
)


def _parse_stem(stem: str) -> Optional[Tuple[str, int, str]]:
    """
    Parse an embeddings filename stem.

    Parameters
    ----------
    stem : str
        Filename stem without extension, e.g. "real_40_RBD_embs".

    Returns
    -------
    tuple of (str, int, str) or None
        A tuple ``(kind, load, fault_code)`` where:
        - kind is "real" or "synth" (lowercase),
        - load is an integer (e.g., 40),
        - fault_code is uppercase (e.g., "RBD").
        Returns ``None`` if the stem does not match the expected pattern.

    Examples
    --------
    >>> _parse_stem("real_40_RBD_embs")
    ('real', 40, 'RBD')
    """
    m = STEM_REGEX.match(stem)
    if not m:
        return None
    kind = m.group("kind").lower()
    load = int(m.group("load"))
    fault = m.group("fault").upper()
    return kind, load, fault


def _discover_emb_dir(run_dir: Path, emb_root: Path) -> Path:
    """
    Locate the per-run embeddings directory.

    Parameters
    ----------
    run_dir : Path
        Path to the training run directory (used to derive run_id).
    emb_root : Path
        Root directory containing per-run embeddings subfolders.

    Returns
    -------
    Path
        The expected embeddings directory: ``emb_root / run_dir.name``.

    Raises
    ------
    FileNotFoundError
        If the inferred embeddings directory does not exist.
    """
    run_id = run_dir.name
    emb_dir = emb_root / run_id
    if not emb_dir.exists():
        raise FileNotFoundError(
            f"Embeddings dir not found: {emb_dir} "
            f"(did you run embed_fid_from_run.py first?)"
        )
    return emb_dir


def _load_embeddings(path: Path, cast: Optional[np.dtype] = np.float64) -> np.ndarray:
    """
    Load an embeddings matrix from ``.npy`` and optionally cast dtype.

    Parameters
    ----------
    path : Path
        Path to a ``.npy`` file containing a 2-D array (N, D).
    cast : numpy.dtype, optional
        If provided, cast the array to this dtype (default ``float64``).

    Returns
    -------
    ndarray of shape (N, D)
        Loaded embeddings matrix.

    Raises
    ------
    ValueError
        If the loaded array is not two-dimensional.
    """
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"{path.name}: expected 2-D (N, D), got {arr.shape}")
    return arr.astype(cast, copy=False) if cast is not None else arr


def _all_finite(X: np.ndarray) -> bool:
    """
    Check whether all entries in an array are finite.

    Parameters
    ----------
    X : ndarray
        Input array.

    Returns
    -------
    bool
        ``True`` if all elements are finite, otherwise ``False``.
    """
    return np.isfinite(X).all()


# =============================================================================
# Linear algebra (FID components)
# =============================================================================

def _covariance(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Compute an unbiased sample covariance with diagonal jitter.

    Parameters
    ----------
    X : ndarray of shape (N, D)
        Observations by rows.
    eps : float, default=1e-6
        Small non-negative number added to the diagonal for numerical
        stability, i.e., ``C := C + eps * I``.

    Returns
    -------
    ndarray of shape (D, D)
        Symmetric (approximately PSD) covariance matrix.

    Notes
    -----
    - Uses ``ddof=1`` (unbiased).
    - Symmetrizes the result by ``(C + C.T)/2``.
    - Requires ``N >= 2``; otherwise raises ``ValueError``.
    """
    if X.shape[0] < 2:
        raise ValueError("Need at least 2 samples to compute covariance.")
    C = np.cov(X, rowvar=False, ddof=1)
    C = (C + C.T) * 0.5
    return C + eps * np.eye(C.shape[0], dtype=C.dtype)


def _trace_sqrt_product(C1: np.ndarray, C2: np.ndarray) -> float:
    """
    Compute :math:`\\mathrm{Tr}(\\sqrt{C_1 C_2})` using SciPy ``sqrtm``.

    Parameters
    ----------
    C1, C2 : ndarray of shape (D, D)
        Symmetric positive semi-definite covariance matrices.

    Returns
    -------
    float
        Trace of the positive semi-definite square root of the product.

    Notes
    -----
    - We evaluate ``S = sqrtm(C1 @ C2)`` (equivalently
      ``sqrtm(C1^{1/2} C2 C1^{1/2})`` up to similarity).
    - Any tiny imaginary parts due to round-off are discarded
      (``S = S.real`` if complex).
    - The result is re-symmetrized before taking the trace.
    """
    S = la.sqrtm(C1 @ C2)
    if np.iscomplexobj(S):
        S = S.real
    S = (S + S.T) * 0.5
    return float(np.trace(S))


def fid_between(real: np.ndarray, synth: np.ndarray, eps: float = 1e-6) -> float:
    """
    Compute Fréchet distance (FID) between two Gaussian fits.

    Parameters
    ----------
    real : ndarray of shape (N, D)
        Real/sample A embeddings.
    synth : ndarray of shape (M, D)
        Synthetic/sample B embeddings.
    eps : float, default=1e-6
        Diagonal jitter added to covariances for stability.

    Returns
    -------
    float
        FID value. Returns ``NaN`` for invalid inputs (e.g., wrong shapes,
        ``N < 2`` or ``M < 2``, or non-finite values).

    Notes
    -----
    FID is defined as:

    .. math::
        \\|\\mu_r - \\mu_s\\|^2 + \\operatorname{Tr}(C_r + C_s
        - 2\\, (C_r^{1/2} C_s C_r^{1/2})^{1/2})

    where :math:`\\mu_r, C_r` and :math:`\\mu_s, C_s` are the means and
    covariances of the two sets (estimated with ``ddof=1``).

    Important
    ---------
    FID is **not clamped** to be non-negative; small negative values due to
    round-off are preserved.
    """
    # Basic guards
    if real.ndim != 2 or synth.ndim != 2:
        return float("nan")
    Nr, Dr = real.shape
    Ns, Ds = synth.shape
    if Nr < 2 or Ns < 2 or Dr != Ds:
        return float("nan")
    if not (_all_finite(real) and _all_finite(synth)):
        return float("nan")

    # Means and covariances
    mu_r = real.mean(axis=0, dtype=np.float64)
    mu_s = synth.mean(axis=0, dtype=np.float64)
    Cr = _covariance(real, eps=eps)
    Cs = _covariance(synth, eps=eps)

    # FID components
    mean_term = float(np.sum((mu_r - mu_s) ** 2))
    cross_trace = _trace_sqrt_product(Cr, Cs)
    fid = mean_term + float(np.trace(Cr + Cs)) - 2.0 * cross_trace
    return float(fid)  # preserve sign (no clamping)


# =============================================================================
# File discovery & pairing
# =============================================================================

def build_pairs(emb_dir: Path) -> Dict[Tuple[str, int], Dict[str, Path]]:
    """
    Pair available embeddings by (fault, load).

    Parameters
    ----------
    emb_dir : Path
        Directory containing ``*_embs.npy`` files.

    Returns
    -------
    dict
        Mapping ``(fault_code, load) -> {'real': Path, 'synth': Path}``.
        A key's dictionary only contains kinds that actually exist; callers
        must check for both 'real' and 'synth'.
    """
    pairs: Dict[Tuple[str, int], Dict[str, Path]] = {}
    for p in sorted(emb_dir.glob("*_embs.npy")):
        parsed = _parse_stem(p.stem)
        if not parsed:
            print(f"[skip] Not a FID embeddings file: {p.name}")
            continue
        kind, load, fault = parsed
        d = pairs.setdefault((fault, load), {})
        d[kind] = p
    return pairs


def build_real_map(emb_dir: Path) -> Dict[Tuple[str, int], Path]:
    """
    Map (fault, load) -> Path for 'real' embeddings (used by baselines).

    Parameters
    ----------
    emb_dir : Path
        Embeddings directory.

    Returns
    -------
    dict
        Mapping ``(fault_code, load) -> Path`` for ``real_*_embs.npy`` files.
    """
    out: Dict[Tuple[str, int], Path] = {}
    for p in sorted(emb_dir.glob("real_*_embs.npy")):
        parsed = _parse_stem(p.stem)
        if parsed:
            kind, load, fault = parsed
            if kind == "real":
                out[(fault, load)] = p
    return out


def build_synth_map(emb_dir: Path) -> Dict[Tuple[str, int], Path]:
    """
    Map (fault, load) -> Path for 'synth' embeddings (negative baselines).

    Parameters
    ----------
    emb_dir : Path
        Embeddings directory.

    Returns
    -------
    dict
        Mapping ``(fault_code, load) -> Path`` for ``synth_*_embs.npy`` files.
    """
    out: Dict[Tuple[str, int], Path] = {}
    for p in sorted(emb_dir.glob("synth_*_embs.npy")):
        parsed = _parse_stem(p.stem)
        if parsed:
            kind, load, fault = parsed
            if kind == "synth":
                out[(fault, load)] = p
    return out


# =============================================================================
# FID table (per fault, per load)
# =============================================================================

def compute_fid_table(emb_dir: Path, eps: float = 1e-6) -> pd.DataFrame:
    """
    Compute FID per (fault, load) for available real/synth pairs.

    Parameters
    ----------
    emb_dir : Path
        Directory ``<emb_root>/<run_id>`` with ``*_embs.npy`` files.
    eps : float, default=1e-6
        Diagonal jitter added to covariances for numerical stability.

    Returns
    -------
    pandas.DataFrame
        Columns:
            - run_id : str
            - fault_code : str
            - load : int
            - n_real : int
            - n_synth : int
            - dim : int or NaN
            - fid : float (NaN if not computable)

        The table is sorted by (fault_code, load).
    """
    run_id = emb_dir.name
    pairs = build_pairs(emb_dir)
    rows: List[dict] = []

    for (fault, load), d in pairs.items():
        real_p = d.get("real")
        synth_p = d.get("synth")
        if real_p is None or synth_p is None:
            missing = ", ".join(
                x for x, pth in (("real", real_p), ("synth", synth_p)) if pth is None
            )
            print(f"[skip] Missing {missing} for ({fault}, load={load})")
            continue

        # Load arrays; cast to float64 for robust covariance math
        try:
            R = _load_embeddings(real_p, cast=np.float64)
            S = _load_embeddings(synth_p, cast=np.float64)
        except Exception as e:
            print(f"[skip] Failed to load pair ({fault}, {load}): {e}")
            continue

        # Meta for reporting
        n_real = int(R.shape[0])
        n_synth = int(S.shape[0])
        dim = int(R.shape[1]) if R.shape[0] > 0 else (int(S.shape[1]) if S.shape[0] > 0 else 0)

        # Guardrails and compute
        if n_real < 2 or n_synth < 2:
            fid = float("nan")
            print(f"[skip] Too few samples for ({fault}, load={load}): n_real={n_real}, n_synth={n_synth}")
        elif R.shape[1] != S.shape[1]:
            fid = float("nan")
            dim = np.nan
            print(f"[skip] Dim mismatch for ({fault}, load={load}): real D={R.shape[1]} vs synth D={S.shape[1]}")
        elif not (_all_finite(R) and _all_finite(S)):
            fid = float("nan")
            print(f"[skip] Non-finite values for ({fault}, load={load})")
        else:
            fid = fid_between(R, S, eps=eps)

        rows.append(
            {
                "run_id": run_id,
                "fault_code": fault,
                "load": load,
                "n_real": n_real,
                "n_synth": n_synth,
                "dim": dim,
                "fid": fid,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["run_id", "fault_code", "load", "n_real", "n_synth", "dim", "fid"])

    df = pd.DataFrame(rows)
    df.sort_values(by=["fault_code", "load"], inplace=True, ignore_index=True)
    return df


# =============================================================================
# nFID computation (RR floor + configurable negative baselines)
# =============================================================================

def _split_half_rr(X: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a randomized split-half partition of rows.

    Parameters
    ----------
    X : ndarray of shape (N, D)
        Data matrix.
    rng : numpy.random.Generator
        RNG used to shuffle indices for reproducibility.

    Returns
    -------
    tuple of (ndarray, ndarray)
        Two disjoint halves (A, B) partitioning the rows of X as evenly as
        possible. Each half may differ by at most one row when N is odd.
    """
    N = X.shape[0]
    idx = np.arange(N)
    rng.shuffle(idx)
    mid = N // 2
    return X[idx[:mid]], X[idx[mid:]]


def _fid_rr_baseline(R: np.ndarray, repeats: int, rng: np.random.Generator, eps: float) -> float:
    """
    Estimate real↔real split-half baseline (FID_RR) via repeats.

    Parameters
    ----------
    R : ndarray of shape (N, D)
        Real embeddings for a single (fault, load).
    repeats : int
        Number of independent split-half evaluations to average.
    rng : numpy.random.Generator
        RNG used for shuffling in splits.
    eps : float
        Covariance jitter.

    Returns
    -------
    float
        Mean split-half FID over repeats. Returns NaN if there are not enough
        samples to form two halves with >=2 rows each.
    """
    if R.shape[0] < 4:  # need at least 2 samples per half
        return float("nan")
    vals: List[float] = []
    for _ in range(repeats):
        A, B = _split_half_rr(R, rng)
        if A.shape[0] < 2 or B.shape[0] < 2:
            continue
        vals.append(fid_between(A, B, eps=eps))
    return float(np.nan) if len(vals) == 0 else float(np.mean(vals))


def build_cache_key(fault: str, load: int, kind: str) -> Tuple[str, int, str]:
    """Helper to keep a consistent cache key."""
    return (fault, load, kind)


def _get_cached(
    key: Tuple[str, int],
    kind: str,
    *,
    real_map: Dict[Tuple[str, int], Path],
    synth_map: Dict[Tuple[str, int], Path],
    cache: Dict[Tuple[str, int, str], np.ndarray],
) -> Optional[np.ndarray]:
    """
    Cached loader for arrays keyed by (fault, load, kind).

    Parameters
    ----------
    key : tuple
        (fault_code, load)
    kind : {'real', 'synth'}
        Dataset kind to load.
    real_map, synth_map : dict
        Path maps for real/synth files.
    cache : dict
        Array cache to avoid re-loading from disk.

    Returns
    -------
    ndarray or None
        Loaded array (float64) or None if not available.
    """
    ck = build_cache_key(key[0], key[1], kind)
    if ck in cache:
        return cache[ck]
    src = (real_map if kind == "real" else synth_map).get(key)
    if src is None:
        return None
    X = _load_embeddings(src, cast=np.float64)
    cache[ck] = X
    return X


def _fid_negative_baseline(
    mode: str,
    fault: str,
    load: int,
    *,
    real_map: Dict[Tuple[str, int], Path],
    synth_map: Dict[Tuple[str, int], Path],
    cache: Dict[Tuple[str, int, str], np.ndarray],
    eps: float,
    percentile: float = 0.5,
) -> float:
    """
    Compute the negative baseline FID for (fault, load) given a mode.

    Parameters
    ----------
    mode : {'real-other-fault', 'real-synth-other-fault', 'synth-other-fault'}
        Baseline definition:
          - 'real-other-fault'        → FID( real(f,L),  real(g≠f,L) )
          - 'real-synth-other-fault'  → FID( real(f,L), synth(g≠f,L) )
          - 'synth-other-fault'       → FID( synth(f,L), synth(g≠f,L) )
    fault : str
        Reference fault code ``f``.
    load : int
        Load level ``L``.
    real_map, synth_map : dict
        Maps for locating arrays on disk.
    cache : dict
        Array cache keyed by (fault, load, kind).
    eps : float
        Covariance jitter.
    percentile : float, default=0.5
        If multiple "other faults" exist, take this percentile over their FIDs
        (0.5 = median). Robust against outliers and extensible beyond 2 faults.

    Returns
    -------
    float
        The aggregated negative baseline FID for (fault, load), or NaN if not
        computable (e.g., missing arrays).
    """
    key_f = (fault, load)

    # Select anchor and candidate set according to the chosen mode
    if mode == "real-other-fault":
        anchor = _get_cached(key_f, "real", real_map=real_map, synth_map=synth_map, cache=cache)
        candidates_keys = [
            ("real", (g, load)) for (g, ld) in real_map.keys() if ld == load and g != fault
        ]
    elif mode == "real-synth-other-fault":
        anchor = _get_cached(key_f, "real", real_map=real_map, synth_map=synth_map, cache=cache)
        candidates_keys = [
            ("synth", (g, load)) for (g, ld) in synth_map.keys() if ld == load and g != fault
        ]
    elif mode == "synth-other-fault":
        anchor = _get_cached(key_f, "synth", real_map=real_map, synth_map=synth_map, cache=cache)
        candidates_keys = [
            ("synth", (g, load)) for (g, ld) in synth_map.keys() if ld == load and g != fault
        ]
    else:
        raise ValueError(f"Unknown neg-baseline: {mode}")

    if anchor is None or not candidates_keys:
        return float("nan")

    fids: List[float] = []
    for kind_other, key_other in candidates_keys:
        other = _get_cached(key_other, kind_other, real_map=real_map, synth_map=synth_map, cache=cache)
        if other is None:
            continue
        fids.append(fid_between(anchor, other, eps=eps))

    fids = [v for v in fids if np.isfinite(v)]
    if not fids:
        return float("nan")

    # Aggregate via percentile for robustness (median by default)
    pct = float(np.clip(percentile, 0.0, 1.0))
    return float(np.quantile(np.array(fids, dtype=np.float64), pct))


def _normalize_modes(modes: Optional[Iterable[str]]) -> List[str]:
    """
    Normalize/validate the list of negative baseline modes.

    Parameters
    ----------
    modes : iterable of str or None
        Iterable of mode names; if None or empty, returns all supported modes.

    Returns
    -------
    list of str
        Validated list of modes in a stable order.
    """
    all_modes = ["real-other-fault", "real-synth-other-fault", "synth-other-fault"]
    if not modes:
        return all_modes
    allowed = set(all_modes)
    out = []
    for m in modes:
        if m not in allowed:
            raise ValueError(f"Unsupported neg-baseline: {m}")
        out.append(m)
    # keep provided order and remove duplicates
    seen = set()
    uniq = []
    for m in out:
        if m not in seen:
            uniq.append(m)
            seen.add(m)
    return uniq


def compute_nfid_table_wide(
    emb_dir: Path,
    fid_df: pd.DataFrame,
    rr_repeats: int = 10,
    seed: int = 123,
    eps: float = 1e-6,
    neg_baselines: Optional[Iterable[str]] = None,
    neg_percentile: float = 0.5,
) -> pd.DataFrame:
    """
    Build a wide normalized FID table using FID_RR and multiple negative baselines.

    Parameters
    ----------
    emb_dir : Path
        Embeddings directory (used to resolve run_id and to find real/synth files).
    fid_df : pandas.DataFrame
        Output of :func:`compute_fid_table`. Must include columns:
        ['run_id','fault_code', 'load', 'fid', 'n_real', 'n_synth', 'dim'].
    rr_repeats : int, default=10
        Number of split-half repeats for the real↔real (FID_RR) floor.
    seed : int, default=123
        RNG seed for split-half shuffles.
    eps : float, default=1e-6
        Covariance jitter for FID computations.
    neg_baselines : iterable of {'real-other-fault', 'real-synth-other-fault', 'synth-other-fault'}, optional
        Baselines to compute; if None or empty, computes **all**.
    neg_percentile : float, default=0.5
        Percentile to aggregate multiple negatives (0.5 = median).

    Returns
    -------
    pandas.DataFrame
        Wide table. Base columns:
            ['run_id','fault_code','load','n_real','n_synth','dim','fid','fid_rr']
        For each baseline ``mode`` (with dashes converted to underscores), adds:
            - f"fid_rw__{mode_tag}"
            - f"nfid__{mode_tag}"
    """
    run_id = emb_dir.name
    modes = _normalize_modes(neg_baselines)
    mode_tags = {m: m.replace("-", "_") for m in modes}

    rng = np.random.default_rng(seed)

    # Index available arrays and keep a small cache for loaded matrices
    real_map = build_real_map(emb_dir)
    synth_map = build_synth_map(emb_dir)
    cache: Dict[Tuple[str, int, str], np.ndarray] = {}

    rows: List[dict] = []
    for row in fid_df.itertuples(index=False):
        fault = row.fault_code
        load = int(row.load)
        fid = float(row.fid)

        # Floor (RR): split-half real vs real within (fault, load)
        key = (fault, load)
        if key in real_map:
            R = _get_cached(key, "real", real_map=real_map, synth_map=synth_map, cache=cache)
            fid_rr = _fid_rr_baseline(R, repeats=rr_repeats, rng=rng, eps=eps) if R is not None else float("nan")
        else:
            fid_rr = float("nan")

        rec = {
            "run_id": run_id,
            "fault_code": fault,
            "load": load,
            "n_real": int(row.n_real),
            "n_synth": int(row.n_synth),
            "dim": row.dim,
            "fid": fid,
            "fid_rr": fid_rr,
        }

        # Compute each requested negative baseline and normalized value
        for mode in modes:
            rw = _fid_negative_baseline(
                mode,
                fault,
                load,
                real_map=real_map,
                synth_map=synth_map,
                cache=cache,
                eps=eps,
                percentile=neg_percentile,
            )
            tag = mode_tags[mode]
            rec[f"fid_rw__{tag}"] = rw
            if np.isfinite(fid) and np.isfinite(fid_rr) and np.isfinite(rw) and (rw - fid_rr) > 1e-12:
                rec[f"nfid__{tag}"] = (fid - fid_rr) / (rw - fid_rr)
            else:
                rec[f"nfid__{tag}"] = float("nan")

        rows.append(rec)

    out = pd.DataFrame(rows).sort_values(by=["fault_code", "load"], ignore_index=True)
    return out


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    """
    Entry point: compute the single detailed FID/nFID table and save it as
    **both CSV and Parquet** next to the embeddings.

    The script writes:
        - CSV     : --out-csv      (default 'fid_table.csv')
        - Parquet : --out-parquet  (default 'fid_table.parquet')
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to a specific run under res/runs/… (used to derive run_id).",
    )
    ap.add_argument(
        "--emb-root",
        type=Path,
        default=Path("res/fid_embs"),
        help="Root folder containing <run_id> subfolder with _embs.npy files.",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("fid_table.csv"),
        help="CSV filename for the detailed (wide) table.",
    )
    ap.add_argument(
        "--out-parquet",
        type=Path,
        default=Path("fid_table.parquet"),
        help="Parquet filename for the detailed (wide) table.",
    )
    ap.add_argument(
        "--rr-repeats",
        type=int,
        default=10,
        help="Number of split-half repeats for FID_RR baseline.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=123,
        help="RNG seed for split-half baselines.",
    )
    ap.add_argument(
        "--eps",
        type=float,
        default=1e-6,
        help="Diagonal jitter added to covariances for numerical stability.",
    )
    ap.add_argument(
        "--neg-baselines",
        nargs="*",
        choices=["real-other-fault", "real-synth-other-fault", "synth-other-fault"],
        default=None,
        help=(
            "Space-separated list of negative baselines to compute. "
            "If omitted, computes ALL of them."
        ),
    )
    ap.add_argument(
        "--neg-percentile",
        type=float,
        default=0.5,
        help="Percentile over multiple negative candidates (0.5 = median).",
    )

    args = ap.parse_args()

    emb_dir = _discover_emb_dir(args.run_dir, args.emb_root)

    # 1) Plain FID per (fault, load)
    fid_df = compute_fid_table(emb_dir, eps=float(args.eps))

    # 2) Detailed (wide) computations table with per-baseline fid_rw & nfid
    modes = _normalize_modes(args.neg_baselines)
    detailed_df = compute_nfid_table_wide(
        emb_dir=emb_dir,
        fid_df=fid_df,
        rr_repeats=int(args.rr_repeats),
        seed=int(args.seed),
        eps=float(args.eps),
        neg_baselines=modes,                # None -> all modes (normalized inside)
        neg_percentile=float(args.neg_percentile),
    )

    # Save outputs in two formats
    out_csv = emb_dir / args.out_csv
    detailed_df.to_csv(out_csv, index=False)
    print(f"[✓] Wrote detailed FID/nFID table (CSV) → {out_csv}")

    out_parquet = emb_dir / args.out_parquet
    try:
        # Note: requires 'pyarrow' or 'fastparquet' to be installed.
        detailed_df.to_parquet(out_parquet, index=False)
        print(f"[✓] Wrote detailed FID/nFID table (Parquet) → {out_parquet}")
    except Exception as e:
        print(f"[warn] Failed to write Parquet at {out_parquet}: {e}")
        print("       Hint: install a Parquet engine, e.g. `pip install pyarrow`.")

    # Pretty print a compact view for quick inspection
    if len(detailed_df):
        with pd.option_context("display.max_rows", 20, "display.max_columns", None):
            subset_cols = ["run_id", "fault_code", "load", "fid", "fid_rr"]
            first_modes = modes[:1]
            if first_modes:
                tag = first_modes[0].replace("-", "_")
                subset_cols += [f"fid_rw__{tag}", f"nfid__{tag}"]
            subset_cols = [c for c in subset_cols if c in detailed_df.columns]
            print("\nDetailed FID/nFID (sample columns):")
            print(detailed_df[subset_cols].to_string(index=False))


if __name__ == "__main__":
    main()
