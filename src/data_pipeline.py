import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import glob

import random
from src.electrical_signature_frequencies import ANOMALY_FREQS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import List, Tuple, Optional, Dict
from tqdm.auto import tqdm


def read_oscilloscope_data(
    file_path: str,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Reads oscilloscope ASCII data file to DataFrame with columns ['Time', 'Data'].
    """
    time = []
    channel_1 = []

    # Open the file and process lines
    with open(file_path, 'r', encoding='latin') as file:
        lines = file.readlines()

        for line in lines:
            # Check if the line contains data with a semicolon separator
            if ";" in line:
                # Split the line into parts
                parts = line.strip().split(";")

                time.append(float(parts[0].strip()))
                channel_1.append(float(parts[1].strip()))

    df = pd.DataFrame({'Time': time, 'Data': channel_1})
    if output_path:
        df.to_csv(output_path, index=False)
    return df


def load_data_from_directory(directory):
    """
    Load data from CSV files in a directory.
    Returns a list of data arrays.
    """
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            data.append(df['Amplitude'].values)
    return np.array(data)

def preprocessing(
    metadata_df: pd.DataFrame,
    out_dir: str,
    segment_length: int,
    step: int,
    f_sampling: int,
    cutoff_freq: int,
    apply_window: bool = False,
    db: bool = True,
) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Convert every measurement in *metadata_df* into FFT magnitude windows.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        Measurement-level table produced by `create_metadata_df`.  **Must** be
        indexed by `"measurement_id"` and contain at least the columns:

            ["state", "phase", "load_condition", "experiment"]

    out_dir : str
        Destination directory for the three artefacts written to disk.

    segment_length : int
        Number of time-domain samples per window.

    step : int
        Hop-size (in samples) between consecutive windows.

    f_sampling : int
        Sampling frequency (Hz) used to compute the frequency vector.

    cutoff_freq : int
        Keep FFT bins only up to this frequency (Hz).

    apply_window : bool, optional
        If *True*, applies a Hann window to every slice before the FFT.
        Default = *False* (good enough for equally spaced integer cycles).

    db : bool, optional
        If *True*, returns magnitude in decibels.  Default = *True*.

    Returns
    -------
    segments : np.ndarray
        Dense float32 array of shape *(N_segments, segment_length)* holding the
        pre-processed FFT spectra.

    seg_meta_df : pd.DataFrame
        Segment-level metadata indexed by the **compound** key
        ``["measurement_id", "segment_idx"]`` - row order is **identical** to
        *segments*.

    freqs : np.ndarray
        1-D float64 array (length = *segment_length*) with the frequency bins
        (post cut-off).

    Notes
    -----
    * The function deliberately **does not** sort the DataFrame, preserving the
      original row order so that `segments[i]` <—> `seg_meta_df.iloc[i]`.
    """
    seg_rows: List[dict] = []
    seg_arrays: List[np.ndarray] = []
    freqs: Optional[np.ndarray] = None

    for m_id, meta in tqdm(metadata_df.iterrows(),
                           total=len(metadata_df),
                           desc="pre-processing"):
        if meta['file_path'].split('.')[-1] == 'csv':
            # 1) Load the CSV
            df = load_measurement(m_id, metadata_df)
            current = df["Current"].dropna().values
        elif meta['file_path'].split('.')[-1] == 'txt':
            df = read_oscilloscope_data(meta['file_path'])
            current = df["Data"].dropna().values    

        # 2) Time-domain slicing
        win_arr = segment_signal(
            current,
            segment_length=segment_length,
            step=step,
            apply_window=apply_window,
        )

        # 3) FFT magnitude spectra
        fft_arr, freqs = perform_fft_on_segments(
            win_arr,
            f_sampling=f_sampling,
            cutoff_freq=cutoff_freq,
            db=db,
        )

        # 4) Build a metadata row for every segment
        n_segments = fft_arr.shape[0]
        for s_idx in range(n_segments):
            start_t = df.index[s_idx * step]
            end_t   = df.index[s_idx * step + segment_length - 1]
            seg_rows.append({
                "measurement_id":  m_id,
                'base_id':         meta["base_id"],  
                "segment_idx":     s_idx,
                "start_time":      float(start_t),
                "end_time":        float(end_t),
                "state":           meta["state"],
                "binary_label":    0 if meta["state"] == "normal" else 1,
                "multiclass_label": meta["state"],
                "phase":           int(meta["phase"]),
                "load_condition":  meta["load_condition"],
                "experiment":      meta["experiment"],
            })
        seg_arrays.append(fft_arr.astype(np.float32))

    # 5) Concatenate and persist
    segments = np.vstack(seg_arrays)                              # shape = (N, L)
    seg_meta_df = pd.DataFrame(seg_rows).set_index(
        ["measurement_id", "segment_idx"]
    )

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "segments.npy"), segments)
    np.save(os.path.join(out_dir, "freqs.npy"), freqs)
    seg_meta_df.to_csv(os.path.join(out_dir, "segments_metadata.csv"))

    return segments, seg_meta_df, freqs

def segment_signal(signal, segment_length, step=None, overlap=None, apply_window=True):
    """
    Segments the signal into windows, with options for overlap or fixed step size.

    Parameters:
    - signal: 1D numpy array representing the time-series data.
    - segment_length: Length of each segment in samples.
    - step: Number of samples to shift the window for each iteration (used for non-overlapping windows).
             If None, overlap is used instead.
    - overlap: Fraction of overlap between consecutive windows (0 to 1). Ignored if `step` is provided.
    - apply_window: Whether to apply a window to each segment.

    Returns:
    - numpy array of segmented windows.
    """
    if step is not None:
        # Calculate segments using fixed step size
        segments = [
            signal[i:i + segment_length]
            for i in range(0, len(signal) - segment_length + 1, step)
        ]
    elif overlap is not None:
        # Calculate segments using overlap
        step = int(segment_length * (1 - overlap))
        segments = [
            signal[i:i + segment_length]
            for i in range(0, len(signal) - segment_length + 1, step)
        ]
    else:
        raise ValueError("Either 'step' or 'overlap' must be specified.")

    return np.array(segments)

def time_to_freq_transform(data, f_sampling, db=True, cutoff_freq=None):
    """
    Transforms time-series data to frequency domain using FFT.
    Optionally applies a cutoff frequency.
    
    Parameters:
    - data: NumPy array containing the time series data.
    - f_sampling: Sampling frequency of the data.
    - db: Boolean flag to convert FFT values to decibel scale.
    - cutoff_freq: Optional cutoff frequency to filter the results.
    
    Returns:
    - yf: Transformed frequency domain magnitude values.
    - freqs: Frequency bins.
    """
    if data.ndim != 1:
        raise ValueError(f"Input data must be a 1D NumPy array. Got shape {data.shape}.")
    
    n     = data.shape[0]  # Number of samples
    yf    = np.fft.rfft(data)  # Perform FFT
    freqs = np.fft.rfftfreq(n, d=1/f_sampling)  # Frequency bins

    if cutoff_freq is not None:
        mask  = freqs < cutoff_freq  # Apply cutoff filter
        yf    = yf[mask]
        freqs = freqs[mask]
    
    yf = np.abs(yf)  # Get magnitude
    
    if db:
        eps = np.finfo(float).eps     # Machine epsilon
        yf = 20 * np.log10(yf + eps)  # Convert to dB scale
    
    return yf, freqs


def perform_fft_on_segments(segments, f_sampling, db=True, cutoff_freq=250, remove_dc=False):
    """
    Perform FFT on each segment in the provided array and return the transformed data.
    
    Parameters:
    - segments: 2D NumPy array of segments (each row is a segment).
    - f_sampling: Sampling frequency of the data.
    - db: Boolean flag to convert FFT values to decibel scale.
    - cutoff_freq: Frequency cutoff for filtering FFT results.
    - remove_dc: Boolean flag to remove DC component from each segment.
    
    Returns:
    - fft_segments: 2D NumPy array where each row is the FFT-transformed data of a segment.
    - freqs: Frequency bins (shared across all segments).
    """
    segs = segments.copy()

    if remove_dc:
        segs -= segs.mean(axis=1, keepdims=True)

    num_segments, segment_length = segs.shape
    
    # Determine the length of FFT output
    full_freqs = np.fft.rfftfreq(segment_length, d=1 / f_sampling)
    cutoff_mask = full_freqs < cutoff_freq
    fft_output_length = np.sum(cutoff_mask)
    
    fft_segments = np.zeros((num_segments, fft_output_length))
    
    # Perform FFT for each segment
    for i in range(num_segments):
        fft_segments[i], freqs = time_to_freq_transform(segments[i], f_sampling, db=db, cutoff_freq=cutoff_freq)
    return fft_segments, freqs


def process_time_series(input_data, output_dir, window_length=20000, shift=20, f_sampling=1.0, db=True, cutoff_freq=250, apply_window=True):
    """
    Applies a sliding window over time series data, performs FFT on each window,
    and saves the transformed data.
    
    Parameters:
    - input_data: Path to the CSV file with time-series data or a pandas DataFrame.
    - output_dir: Directory to save the transformed window files.
    - window_length: Number of samples in each window.
    - shift: Number of samples to shift the window for each iteration.
    - f_sampling: Sampling frequency of the data.
    - db: Boolean flag to convert FFT values to decibel scale.
    - cutoff_freq: Frequency cutoff for filtering FFT results.
    """
    # Determine if input_data is a file path or a DataFrame
    if isinstance(input_data, str):
        df = pd.read_csv(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data
    else:
        raise ValueError("input_data must be a file path (str) or a pandas DataFrame.")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Segment the signal
    segments = segment_signal(df['Data'].values, segment_length=window_length, step=shift, apply_window=apply_window)
    
    # Perform FFT on each segment and get the FFT data
    fft_segments, freqs = perform_fft_on_segments(segments, f_sampling, db=db, cutoff_freq=cutoff_freq)
    
    # Save each FFT-transformed segment to a file
    for i, yf in enumerate(fft_segments):
        transformed_df = pd.DataFrame({
            'Frequency (Hz)': freqs,
            'Amplitude': yf
        })
        output_file_path = os.path.join(output_dir, f'Window_{i + 1}.csv')
        transformed_df.to_csv(output_file_path, index=False)
    
    print(f'Results saved to {output_dir}')
    return fft_segments, freqs

def process_file(file_path, label, 
                 f_sampling=10000, cutoff_freq=250, segment_length=10000, step=20, 
                 apply_window=False, db=True,
                 synthetic_fault_fraction=0.0, 
                 fault_types=None, anomaly_injector=None):
    """
    Processes a single file by reading raw data, segmenting, performing FFT,
    and optionally injecting synthetic faults (if label=='normal' and synthetic_fault_fraction>0). 

    Parameters:
    -----------
    file_path : str
        Path to the raw oscilloscope data file.
    label : str
        Label for the file ('normal' or a specific fault type).
    f_sampling : int, optional
        Sampling frequency (default 10000 Hz).
    cutoff_freq : float, optional
        Maximum frequency (Hz) to keep from the FFT (default 250 Hz).
    segment_length : int, optional
        Length (in points) of each segment for FFT extraction (default 6).
    step : int, optional
        Step size for segmenting the signal (default 20).
    apply_window : bool, optional
        If True, a window function is applied to each segment before FFT (default False).
    db : bool, optional
        If True, convert FFT magnitudes to dB scale (default True).
    synthetic_fault_fraction : float, optional
        Fraction (0.0 to 1.0) of normal segments to modify with synthetic faults (default 0.0).
    fault_types : list of str, optional
        List of possible fault types to inject (required if synthetic_fault_fraction>0).
    anomaly_injector: instance of BaseAnomalyInjector (or CompositeAnomalyInjector) to perform injection.

    Returns:
    --------
    tuple: (all_fft_segments, labels, freqs)
        all_fft_segments : list of np.ndarray
            FFT-transformed segments (with or without synthetic injection).
        labels : list of str
            Labels corresponding to each segment.
        freqs : np.ndarray
            Frequency bins from the FFT.
    """
    # 1. Read data from file
    df = read_oscilloscope_data(file_path, output_path=None)

    # 2. Segment the time-series data
    segments = segment_signal(
        df['Data'].values, 
        segment_length=segment_length, 
        step=step, 
        apply_window=apply_window
    )
    
    # 3. Perform FFT on each segment
    fft_segments, freqs = perform_fft_on_segments(
        segments, 
        f_sampling, 
        db=db, 
        cutoff_freq=cutoff_freq
    )
    
    all_fft_segments = []
    all_labels = []
    
    # 4. If the file is labeled 'normal' and we want to inject synthetic faults
    if label == 'normal' and synthetic_fault_fraction > 0 and fault_types is not None:
        n_segments = len(fft_segments)
        n_modify   = int(n_segments * synthetic_fault_fraction)
        modify_indices = random.sample(range(n_segments), n_modify)
        
        for i, seg in enumerate(fft_segments):
            if i in modify_indices:
                # Pick a random fault type
                chosen_fault = random.choice(fault_types)
                
                # Insert anomalies
                mod_seg = anomaly_injector.inject(seg, freqs, chosen_fault)

                all_fft_segments.append(mod_seg)
                all_labels.append(chosen_fault)
            else:
                # Keep it normal
                all_fft_segments.append(seg)
                all_labels.append(label)
    else:
        # 5. No injection, just keep all segments with the given label
        all_fft_segments = fft_segments
        all_labels       = [label] * len(fft_segments)
    
    return all_fft_segments, all_labels, freqs


def count_labels(loader):
    """
    Counts the number of occurrences of each label in a DataLoader.

    Parameters:
    - loader: DataLoader containing data and labels.

    Returns:
    - label_counts: Dictionary with label counts.
    """
    label_counts = {}
    for _, labels in loader:
        for label in labels.numpy():
            label = int(label)
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
    return label_counts

def create_dataset(file_label_map, mode="binary",
                   f_sampling=10000, cutoff_freq=250, segment_length=10000, step=20,
                   apply_window=False, db=True, 
                   fault_types_available=None, include_real_anomalies_in_training=False,
                   test_size=0.3, val_size=0.2, normalization_method='min-max', 
                   seed=42, anomaly_injector=None):
    """
    Creates training and test datasets using process_file.
    
    For files labeled "normal":
      - Process the file without synthetic injection.
      - Split the resulting segments into training and test sets to avoid data leakage.
      - Inject synthetic faults (with randomized amplitude) into a fraction of the training segments using the provided anomaly_injector.
      
    For fault files, synthetic injection is not applied.
    
    Synthetic fault fraction:
      - Binary mode: 0.5 (i.e. 50% of training normal segments are injected)
      - Multiclass mode: k/(k+1), where k is the number of fault types available.
      
    Parameters:
      ...
      anomaly_injector : instance of BaseAnomalyInjector (or CompositeAnomalyInjector), optional.
                           If provided, it will be used to inject synthetic anomalies into training segments.
    
    Returns:
      dict: A dictionary with keys "X_train", "y_train", "X_val", "y_val", "X_test", "y_test", "freqs", and optionally "label_encoder".
    """
    if mode == "binary":
        synthetic_fraction = 0.5
    elif mode == "multiclass":
        if fault_types_available is None or len(fault_types_available) == 0:
            raise ValueError("For multiclass mode, fault_types_available must be provided and non-empty.")
        k = len(fault_types_available)
        synthetic_fraction = k / (k + 1)
    else:
        raise ValueError("Mode must be either 'binary' or 'multiclass'")
    
    train_segments = []
    train_labels = []
    val_segments = []
    val_labels = []
    test_segments = []
    test_labels = []
    common_freqs = None
    
    for file_path, label in file_label_map.items():
        # Process file without injection to obtain baseline segments.
        segs, labels_out, freqs = process_file(
            file_path, label,
            f_sampling=f_sampling, cutoff_freq=cutoff_freq,
            segment_length=segment_length, step=step,
            apply_window=apply_window, db=db,
            synthetic_fault_fraction=0.0,  # No injection in baseline processing.
            fault_types=None,
            anomaly_injector=None
        )
        if common_freqs is None:
            common_freqs = freqs
        
        if label == "normal":
            # Split normal segments into train+validation and test sets.
            segs_train_val, segs_test, labels_train_val, labels_test = train_test_split(
                segs, labels_out, test_size=test_size, random_state=seed
            )
            # Further split train+validation into training and validation sets.
            segs_train, segs_val, labels_train, labels_val = train_test_split(
                segs_train_val, labels_train_val, test_size=val_size, random_state=seed
            )

            n_train = len(segs_train)
            n_modify = int(n_train * synthetic_fraction)
            modify_indices = random.sample(range(n_train), n_modify)
            
            new_train_segs = []
            new_train_labels = []
            for i, seg in enumerate(segs_train):
                if i in modify_indices and anomaly_injector is not None:
                    # Randomly choose a fault type from available list.
                    chosen_fault = random.choice(fault_types_available)
                    # Inject synthetic anomaly using the provided injector.
                    mod_seg = anomaly_injector.inject(seg, freqs, chosen_fault)
                    new_train_segs.append(mod_seg)
                    new_train_labels.append(chosen_fault)
                else:
                    new_train_segs.append(seg)
                    new_train_labels.append(label)
            
            train_segments.extend(new_train_segs)
            train_labels.extend(new_train_labels)
            val_segments.extend(segs_val)
            val_labels.extend(labels_val)
            test_segments.extend(segs_test)
            test_labels.extend(labels_test)
        else:
            # For fault files, split segments into train+validation and test sets.
            segs_train_val, segs_test, labels_train_val, labels_test = train_test_split(
                segs, labels_out, test_size=test_size, random_state=seed
            )
            segs_train, segs_val, labels_train, labels_val = train_test_split(
                segs_train_val, labels_train_val, test_size=val_size, random_state=seed
            )
            if include_real_anomalies_in_training:
                train_segments.extend(segs_train)
                train_labels.extend(labels_train)
            else:
                val_segments.extend(segs_train)
                val_labels.extend(labels_train)
            val_segments.extend(segs_val)
            val_labels.extend(labels_val)
            test_segments.extend(segs_test)
            test_labels.extend(labels_test)
    
    X_train_raw = np.array(train_segments, dtype=np.float32)
    X_val_raw   = np.array(val_segments, dtype=np.float32)
    X_test_raw  = np.array(test_segments, dtype=np.float32)
    
    if normalization_method:
        # TODO: Replace by Normalizer
        X_train = np.array([normalize_segment(seg, method=normalization_method)[0] for seg in X_train_raw])
        X_val   = np.array([normalize_segment(seg, method=normalization_method)[0] for seg in X_val_raw])
        X_test  = np.array([normalize_segment(seg, method=normalization_method)[0] for seg in X_test_raw])
    else:
        X_train = X_train_raw
        X_val   = X_val_raw
        X_test  = X_test_raw
    
    y_train_raw = np.array(train_labels)
    y_val_raw   = np.array(val_labels)
    y_test_raw  = np.array(test_labels)
    
    y_train_bin = np.array([0 if lbl=="normal" else 1 for lbl in y_train_raw])
    y_val_bin   = np.array([0 if lbl=="normal" else 1 for lbl in y_val_raw])
    y_test_bin  = np.array([0 if lbl=="normal" else 1 for lbl in y_test_raw])
    
    le = LabelEncoder()
    y_train_multi = le.fit_transform(y_train_raw)
    y_val_multi   = le.transform(y_val_raw)
    y_test_multi  = le.transform(y_test_raw)
    
    if mode == "binary":
        final_y_train = y_train_bin
        final_y_val   = y_val_bin
        final_y_test  = y_test_bin
    else:
        final_y_train = y_train_multi
        final_y_val   = y_val_multi
        final_y_test  = y_test_multi
    
    return {
        "X_train": X_train,
        "y_train": final_y_train,
        "X_val": X_val,
        "y_val": final_y_val,
        "X_test": X_test,
        "y_test": final_y_test,
        "freqs": common_freqs,
        "label_encoder": le if mode=="multiclass" else None
    }

def make_importance_mask(
        freqs: np.ndarray,
        fault_freqs: Dict[str, np.ndarray],
        delta_hz: float = 3.0,
        smooth: bool = False
    ) -> np.ndarray:
    """
    Returns a 1-D array (len = len(freqs)) with 1.0 inside ±delta_hz of ANY
    characteristic fault frequency and 0.0 elsewhere.

    • Boolean internally → no TypeError.
    • Optional cosine smoothing at the edges.
    """
    mask_bool = np.zeros_like(freqs, dtype=bool)
    for f_list in fault_freqs.values():
        for f0 in f_list:
            mask_bool |= np.abs(freqs - f0) <= delta_hz

    if not smooth:
        return mask_bool.astype(np.float32)

    mask = np.zeros_like(freqs, dtype=np.float32)
    for f_list in fault_freqs.values():
        for f0 in f_list:
            dist = np.abs(freqs - f0)
            ramp = 0.5 * (1 + np.cos(np.pi * dist / (1.5 * delta_hz)))
            mask += np.where(dist <= 1.5 * delta_hz, ramp, 0.0)
    mask = np.clip(mask, 0, 1)
    return mask


def create_metadata_df(base_dir: str, state2name: dict | None = None) -> pd.DataFrame:
    """
    Create a metadata DataFrame for measurement CSV files from the dataset.
    
    This function searches for all CSV files in the following directory structure:
    
        base_dir/experiment_*/current/<load_condition>/<phase>/<file>.csv
    
    For each CSV file, it extracts the following metadata:
        - experiment: The experiment folder name (e.g., 'experiment_1').
        - load_folder: The load condition folder (e.g., '1st_load_80'). From this:
            - state: Derived from the first part (e.g., '1' from '1st').
            - load_condition: The last part (e.g., '80').
        - phase: The motor phase folder (e.g., '1').
        - measurement_id: The CSV filename without its extension + a phase number if provided.
        - num_observations: Number of rows in the CSV (based on the first column).
        - file_path: The full path to the CSV file, allowing for on-the-fly loading.
    
    The function returns a pandas DataFrame with one row per measurement. The
    DataFrame is indexed by measurement_id.
    
    Parameters:
        base_dir (str): Base directory where the experiment folders are located.
        state_type2name (dict, optional): A mapping of states to their names. If provided,
            it will be included in the metadata DataFrame.
        
    Returns:
        pd.DataFrame: A DataFrame containing metadata for each measurement.
    """
    csv_pattern = os.path.join(base_dir, 'experiment_*', 'current', '*', '*', '*.csv')
    csv_files = glob.glob(csv_pattern)

    metadata_list = []

    for file_path in csv_files:
        # Expected structure: 
        # base_dir/experiment_*/current/<load_condition>/<phase>/<filename>.csv
        parts = file_path.split(os.sep)
        try:
            experiment = parts[-5]  # e.g., "experiment_1"
            load_folder = parts[-3]  # e.g., "1st_load_80"
            phase = parts[-2]        # e.g., "1"
            base_id = os.path.splitext(os.path.basename(file_path))[0]
            measurement_id = base_id + f'_{phase}'
        except IndexError as e:
            print(f"File path {file_path} does not match expected structure: {e}")
            continue

        # Parse the load_folder to extract state and load_condition.
        try:
            load_parts = load_folder.split('_')
            if len(load_parts) >= 3:
                state = load_parts[0][0]  # e.g., from "1st" take '1'
                load_condition = load_parts[-1]  # e.g., "80"
            else:
                state = None
                load_condition = load_folder
        except Exception as e:
            print(f"Error parsing load folder {load_folder} in file {file_path}: {e}")
            state, load_condition = None, None

        # Read the CSV file to count the number of observations.
        try:
            df = pd.read_csv(file_path, header=0, index_col=0)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue

        # Drop any columns that are completely NaN (e.g., from trailing delimiters).
        df = df.dropna(axis=1, how='all')

        num_observations = df.shape[0]

        if state2name is not None:
            try:
                state = state2name[int(state)]
            except KeyError:
                print(f"Fault type {state} not found in mapping.")
                state = None

        # Create the metadata entry.
        meta_entry = {
            'measurement_id': measurement_id,
            'base_id': base_id,
            'experiment': experiment,
            'state': state,
            'load_condition': load_condition,
            'phase': phase,
            'num_observations': num_observations,
            'file_path': file_path,
            'engine_cfg_path': f'../dataset/engine_2/engine.yml'
        }
        metadata_list.append(meta_entry)

    if metadata_list:
        metadata_df = pd.DataFrame(metadata_list)
        metadata_df.set_index('measurement_id', inplace=True)
    else:
        metadata_df = pd.DataFrame(columns=[
            'experiment',
            'state',
            'load_condition',
            'phase',
            'num_observations',
            'file_path',
            'engine_cfg_path'
        ])

    return metadata_df

def load_measurement(measurement_id: str, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load a specified measurement from its CSV file based on the metadata DataFrame.
    
    This function retrieves the file path for the given measurement_id from metadata_df,
    then loads the CSV file located at that path. The CSV file is assumed to contain two
    columns representing time and current (without a header). Extra columns (if any) are discarded.
    
    The returned DataFrame has a MultiIndex where:
      - The first level is the measurement_id (repeated for all observations).
      - The second level is a sequential observation index (obs_index).
    
    Parameters:
        measurement_id (str): The unique identifier of the measurement to load.
        metadata_df (pd.DataFrame): A DataFrame (indexed by measurement_id) containing at least
                                    a 'file_path' column with the path to the CSV file.
    
    Returns:
        pd.DataFrame: A DataFrame with columns ['Time', 'Current'] and a MultiIndex (measurement_id, obs_index).
    
    Raises:
        KeyError: If the measurement_id is not found in the metadata DataFrame.
        Exception: If there is an error in loading or processing the CSV file.
    """
    # Verify that the measurement_id exists in the metadata DataFrame.
    if measurement_id not in metadata_df.index:
        raise KeyError(f"Measurement {measurement_id} not found in metadata.")

    # Retrieve the file path for the specified measurement.
    file_path = metadata_df.loc[measurement_id, 'file_path']

    try:
        # Read the CSV file. The file is assumed to have no header.
        df = pd.read_csv(file_path, header=0, index_col=0)
    except Exception as e:
        raise Exception(f"Error reading file {file_path}: {e}")

    # Drop columns that are completely NaN (common if trailing delimiters exist).
    df = df.dropna(axis=1, how='all')

    # Rename columns to 'Time' and 'Current'.
    df.columns = ['Time', 'Current']

    # Set Index
    df.set_index('Time', inplace=True) 
    df.sort_index(inplace=True)
    return df

def filter_segments(
    seg_meta_df: pd.DataFrame,
    segments: np.ndarray,
    training_classes: List[str],
    loads_to_use: List[str],
    phases_to_use: List[str]
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Filter segment metadata and corresponding segments array based on specified criteria.

    Parameters
    ----------
    seg_meta_df : pd.DataFrame
        DataFrame containing segment metadata. Must have columns:
        - 'state'
        - 'load_condition'
        - 'phase'
    segments : np.ndarray
        2D array of segment data where each row corresponds to a row in seg_meta_df.
    training_classes : List[str]
        Allowed values for the 'state' column.
    loads_to_use : List[str]
        Allowed values for the 'load_condition' column.
    phases_to_use : List[str]
        Allowed values for the 'phase' column.

    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray]
        A tuple containing:
        - Filtered DataFrame (sub-set of seg_meta_df)
        - Filtered segments array (rows matching the filtered DataFrame)
    """
    # Build individual masks
    training_mask = seg_meta_df['state'].isin(training_classes)
    loads_mask    = seg_meta_df['load_condition'].isin(loads_to_use)
    phases_mask   = seg_meta_df['phase'].isin(phases_to_use)

    # Combine masks
    final_mask = training_mask & loads_mask & phases_mask

    # Apply mask
    filtered_meta     = seg_meta_df.loc[final_mask].reset_index(drop=True)
    filtered_segments = segments[final_mask]

    return filtered_meta, filtered_segments
