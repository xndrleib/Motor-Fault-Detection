import numpy as np
import pandas as pd
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