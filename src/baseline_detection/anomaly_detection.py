import numpy as np
import scipy.signal as sp_signal
from typing import Callable, Dict, Optional, Tuple, List

from src.electrical_signature_frequencies import ANOMALY_FREQS as ANOMALY_FREQS_ELECTRO


def get_anomaly_score(
    freqs: np.ndarray,
    signal: np.ndarray,
    anomaly_freqs: List[float],
    max_peak_distance_hz: float = 0.5,
    thresh: float = 0.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Calculate anomaly scores for specified frequencies.

    Args:
        freqs: Array of frequency values.
        signal: Signal amplitude at each frequency.
        anomaly_freqs: List of anomaly frequencies to check.
        max_peak_distance_hz: Maximum distance to consider peaks as neighbors.
        thresh: Threshold for anomaly scores.

    Returns:
        Tuple of anomaly frequencies and their scores if above threshold; None otherwise.
    """
    freqs = np.array(freqs)
    peaks, _ = sp_signal.find_peaks(signal, distance=5, prominence=10)
    prominences = sp_signal.peak_prominences(signal, peaks)[0]
    median_prominence = np.median(prominences)

    locations = []
    scores = []
    matched_anomaly_freqs = []

    for anomaly_freq in anomaly_freqs:
        neighbors_indices = [i for i in range(freqs.shape[0]) if abs(freqs[i] - anomaly_freq) < max_peak_distance_hz]
        neighboring_peaks = [i for i, peak in enumerate(peaks) if peak in neighbors_indices]

        if not neighboring_peaks:
            continue

        # Select the most prominent peak
        best_peak = max(neighboring_peaks, key=lambda idx: prominences[idx])
        prominence = prominences[best_peak]
        score = -1 + 2 / (1 + np.exp(-prominence / median_prominence))

        scores.append(score)
        matched_anomaly_freqs.append(anomaly_freq)

    scores = np.array(scores)
    matched_anomaly_freqs = np.array(matched_anomaly_freqs)

    mask = scores > thresh
    if np.any(mask):
        return matched_anomaly_freqs[mask], scores[mask]
    return None, None


def detect(
    freqs: np.ndarray,
    signal: np.ndarray,
    engine_config: Dict,
    anomaly_freqs_dict: Dict[str, Callable[[Dict], List[float]]],
    max_peak_distance_hz: float = 0.5,
    thresh_dict: Dict[str, float] = {},
) -> Dict[str, Dict[str, List[float]]]:
    """
    Detect anomalies based on anomaly frequency dictionaries.

    Args:
        freqs: Frequency values.
        signal: Signal amplitude.
        engine_config: Configuration for the engine.
        anomaly_freqs_dict: Dictionary mapping anomaly types to frequency functions.
        max_peak_distance_hz: Maximum peak distance for anomaly matching.
        thresh_dict: Threshold values for each anomaly type.

    Returns:
        Dictionary of detected anomalies with scores and locations.
    """
    results = {
        anomaly_type: get_anomaly_score(
            freqs, signal, anomaly_freqs_fn(engine_config), max_peak_distance_hz, thresh_dict.get(anomaly_type, 0)
        )
        for anomaly_type, anomaly_freqs_fn in anomaly_freqs_dict.items()
    }
    return {
        anomaly_type: {"scores": scores.tolist(), "locations": locations.tolist()}
        for anomaly_type, (locations, scores) in results.items()
        if locations is not None
    }


def detect_electro(
    freqs: np.ndarray,
    signal: np.ndarray,
    engine_config: Dict,
    max_peak_distance_hz: float = 0.5,
    thresh_dict: Dict[str, float] = {},
) -> Dict[str, Dict[str, List[float]]]:
    """
    Detect anomalies in electrical signals.

    Args:
        freqs: Frequency values.
        signal: Signal amplitude.
        engine_config: Configuration for the engine.
        max_peak_distance_hz: Maximum peak distance for anomaly matching.
        thresh_dict: Threshold values for each anomaly type.

    Returns:
        Dictionary of detected anomalies with scores and locations.
    """
    return detect(freqs, signal, engine_config, ANOMALY_FREQS_ELECTRO, max_peak_distance_hz, thresh_dict)


if __name__ == '__main__':
    # python src/baseline_detection/anomaly_detection.py "data/raw/холостой ход 11 01.txt" LIMAN
    from src.data_pipeline import read_oscilloscope_data, time_to_freq_transform
    
    import argparse
    import json
    import os
    import yaml
    import noisereduce as nr

    CONFIGS_DIR = 'engine_configs'    

    configs = [x.split('.')[0] for x in os.listdir(CONFIGS_DIR) if 'yml' in x]

    parser = argparse.ArgumentParser(description='anomalies detector')
    parser.add_argument('filename', type=str)
    parser.add_argument('config', type=str, choices=configs)
    parser.add_argument('--max_peak_distance_Hz', type=float, default=0.5)
    parser.add_argument('--cut_freq_Hz', type=float, default=200, help='Cut all frequencies above the value')
    parser.add_argument('--noise_reduction', type=bool, default=True, help='Enable noise reduction of the original signal')

    args = parser.parse_args()

    with open(f"{CONFIGS_DIR}/{args.config}.yml", "r") as stream:
        engine_config = yaml.safe_load(stream)

    df = read_oscilloscope_data(args.filename)
    f_sampling = engine_config['f_sampling']

    if args.noise_reduction:
        df['Data'] = nr.reduce_noise(df['Data'], int(f_sampling))

    yf, freqs = time_to_freq_transform(df, f_sampling, args.cut_freq_Hz)

    preds = detect_electro(freqs, yf, engine_config, max_peak_distance_hz=args.max_peak_distance_Hz)
    print(json.dumps(preds, ensure_ascii=False))