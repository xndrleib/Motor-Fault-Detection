# evaluation.py
from typing import Optional

import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

def compute_mse(original_segment, denoised_segment):
    """
    Computes Mean Squared Error (MSE) between original and denoised segments.
    """
    return np.mean((original_segment - denoised_segment) ** 2)

def compute_snr(original_segment, denoised_segment):
    """
    Computes the Signal-to-Noise Ratio (SNR) in dB for one segment.
    SNR = 20 * log10(||original|| / ||original - denoised||).
    """
    numerator = np.linalg.norm(original_segment)
    denominator = np.linalg.norm(original_segment - denoised_segment) + 1e-12
    return 20 * np.log10(numerator / denominator)

def compute_metrics(original_segments, denoised_segments):
    """
    Computes the average MSE and average SNR across all segments.
    Returns (avg_mse, avg_snr).
    """
    mses = []
    snrs = []
    for orig_seg, den_seg in zip(original_segments, denoised_segments):
        mses.append(compute_mse(orig_seg, den_seg))
        snrs.append(compute_snr(orig_seg, den_seg))
    return np.mean(mses), np.mean(snrs)


def calculate_metrics(true_labels, predictions):
    """
    Calculates and prints confusion matrix and classification metrics.
    
    Parameters:
      - true_labels: list or array of ground truth labels.
      - predictions: list or array of predicted labels.
      
    Returns:
      - cm: Confusion matrix (as a NumPy array).
      - accuracy: Overall accuracy in percent.
      - precision: Weighted precision.
      - recall: Weighted recall.
      - f1_score: Weighted F1 score.
    """
    cm = confusion_matrix(true_labels, predictions)
    accuracy = 100.0 * np.sum(np.array(true_labels) == np.array(predictions)) / len(true_labels)
    precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    return cm, accuracy, precision, recall, f1_score

def slice_metrics(df: pd.DataFrame,
                  y_true: str,
                  y_pred: str,
                  average: str = "binary") -> pd.DataFrame:
    """
    Return accuracy, precision, recall, F1 **and n** for every
    (phase, load_condition) slice.
    """
    records = []
    for (phase, load), g in df.groupby(["phase", "load_condition"]):
        n   = len(g)
        acc = accuracy_score(g[y_true], g[y_pred])
        prc, rec, f1, _ = precision_recall_fscore_support(
            g[y_true], g[y_pred],
            average=average,
            zero_division=0,
        )
        records.append({
            "phase":      phase,
            "load":       load,
            "n":          n,           
            "accuracy":   acc,
            "precision":  prc,
            "recall":     rec,
            "f1":         f1,
        })
    return (
        pd.DataFrame.from_records(records)
        .sort_values(["load", "phase"], ignore_index=True)
    )


def compute_fid(real_embeddings: np.ndarray,
               synthetic_embeddings: np.ndarray,
               eps: float = 1e-6) -> float:
    """Compute the Fréchet Inception Distance between two embedding matrices.

    Parameters
    ----------
    real_embeddings : np.ndarray
        Array of shape (n_real, n_features) containing embeddings for real data.
    synthetic_embeddings : np.ndarray
        Array of shape (n_synth, n_features) containing embeddings for synthetic data.
    eps : float, optional
        Diagonal regularisation term added to the covariance matrices to maintain
        numerical stability.

    Returns
    -------
    float
        The Fréchet distance between the empirical Gaussians induced by the
        real and synthetic embeddings.

    Raises
    ------
    ValueError
        If the embedding matrices are not two dimensional, do not share the same
        feature dimension or contain fewer than two samples.
    """

    real_embeddings = np.asarray(real_embeddings, dtype=np.float64)
    synthetic_embeddings = np.asarray(synthetic_embeddings, dtype=np.float64)

    if real_embeddings.ndim != 2 or synthetic_embeddings.ndim != 2:
        raise ValueError("Embeddings must be two-dimensional matrices.")

    if real_embeddings.shape[1] != synthetic_embeddings.shape[1]:
        raise ValueError("Real and synthetic embeddings must have the same number of features.")

    if real_embeddings.shape[0] < 2 or synthetic_embeddings.shape[0] < 2:
        raise ValueError("At least two samples are required in each embedding set to compute FID.")

    mu_real = real_embeddings.mean(axis=0)
    mu_synth = synthetic_embeddings.mean(axis=0)

    cov_real = np.atleast_2d(np.cov(real_embeddings, rowvar=False))
    cov_synth = np.atleast_2d(np.cov(synthetic_embeddings, rowvar=False))

    # Ensure covariance matrices are well-conditioned
    dim = cov_real.shape[0]
    cov_real += np.eye(dim) * eps
    cov_synth += np.eye(dim) * eps

    covmean, _ = sqrtm(cov_real @ cov_synth, disp=False)
    if not np.isfinite(covmean).all():
        # Fallback to higher regularisation if sqrtm produced NaNs
        identity = np.eye(dim)
        jitter = eps
        for scale in (10.0, 100.0, 1000.0):
            covmean, _ = sqrtm((cov_real + identity * jitter) @ (cov_synth + identity * jitter), disp=False)
            if np.isfinite(covmean).all():
                break
            jitter *= scale
        if not np.isfinite(covmean).all():
            raise ValueError("Failed to compute a stable covariance square root for FID calculation.")

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = np.sum((mu_real - mu_synth) ** 2)
    fid += np.trace(cov_real + cov_synth - 2.0 * covmean)

    # Numerical noise can produce tiny negative values
    return float(np.maximum(fid, 0.0))


def bootstrap_fid(real_embeddings: np.ndarray,
                  synthetic_embeddings: np.ndarray,
                  n_bootstrap: int = 1000,
                  random_state: Optional[int] = None) -> np.ndarray:
    """Estimate the sampling distribution of FID using bootstrap resampling.

    Parameters
    ----------
    real_embeddings : np.ndarray
        Array of shape (n_real, n_features) containing embeddings for real data.
    synthetic_embeddings : np.ndarray
        Array of shape (n_synth, n_features) containing embeddings for synthetic data.
    n_bootstrap : int, optional
        Number of bootstrap iterations.
    random_state : Optional[int], optional
        Seed for the random number generator to ensure reproducibility.

    Returns
    -------
    np.ndarray
        Array of length ``n_bootstrap`` with bootstrapped FID scores.
    """

    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be a positive integer.")

    real_embeddings = np.asarray(real_embeddings)
    synthetic_embeddings = np.asarray(synthetic_embeddings)

    n_real = real_embeddings.shape[0]
    n_synth = synthetic_embeddings.shape[0]

    if n_real < 2 or n_synth < 2:
        raise ValueError("At least two samples are required in each embedding set to bootstrap FID.")

    rng = np.random.default_rng(random_state)
    scores = np.empty(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        real_idx = rng.integers(0, n_real, size=n_real)
        synth_idx = rng.integers(0, n_synth, size=n_synth)
        scores[i] = compute_fid(real_embeddings[real_idx], synthetic_embeddings[synth_idx])

    return scores
