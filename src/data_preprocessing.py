import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt

from scipy.signal import convolve, hann, resample
import glob
import random
import math


def read_oscilloscope_data(file_path, output_path=None):
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


def segment_signal(signal, segment_length, step=None, overlap=None, apply_window=True):
    """
    Segments the signal into windows, with options for overlap or fixed step size.

    Parameters:
    - signal: 1D numpy array representing the time-series data.
    - segment_length: Length of each segment in samples.
    - step: Number of samples to shift the window for each iteration (used for non-overlapping windows).
             If None, overlap is used instead.
    - overlap: Fraction of overlap between consecutive windows (0 to 1). Ignored if `step` is provided.
    - apply_window: Whether to apply a Hann window to each segment.

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

    if apply_window:
        hann_window = hann(segment_length)
        segments = [seg * hann_window for seg in segments]

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
    - yf: Transformed frequency domain values.
    - freqs: Frequency bins.
    """
    if data.ndim != 1:
        raise ValueError(f"Input data must be a 1D NumPy array. Got shape {data.shape}.")
    
    n = data.shape[0]  # Number of samples
    yf = np.fft.rfft(data)  # Perform FFT
    freqs = np.fft.rfftfreq(n, d=1/f_sampling)  # Frequency bins
    
    if db:
        yf = 20 * np.log10(np.abs(yf))  # Convert to dB scale
    
    if cutoff_freq is not None:
        mask = freqs < cutoff_freq  # Apply cutoff filter
        yf = yf[mask]
        freqs = freqs[mask]
    
    return yf, freqs


def perform_fft_on_segments(segments, f_sampling, db=True, cutoff_freq=250):
    """
    Perform FFT on each segment in the provided array and return the transformed data.
    
    Parameters:
    - segments: 2D NumPy array of segments (each row is a segment).
    - f_sampling: Sampling frequency of the data.
    - db: Boolean flag to convert FFT values to decibel scale.
    - cutoff_freq: Frequency cutoff for filtering FFT results.
    
    Returns:
    - fft_segments: 2D NumPy array where each row is the FFT-transformed data of a segment.
    - freqs: Frequency bins (shared across all segments).
    """
    num_segments = segments.shape[0]
    segment_length = segments.shape[1]
    
    # Determine the length of FFT output
    full_freqs = np.fft.rfftfreq(segment_length, d=1 / f_sampling)
    cutoff_mask = full_freqs < cutoff_freq
    fft_output_length = np.sum(cutoff_mask)
    
    fft_segments = np.zeros((num_segments, fft_output_length))
    freqs = full_freqs[cutoff_mask]
    
    # Perform FFT for each segment
    for i in range(num_segments):
        yf, _ = time_to_freq_transform(segments[i], f_sampling, db=db, cutoff_freq=cutoff_freq)
        fft_segments[i, :] = yf
    
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

def extract_segment(fft_data, target_frequency, segment_length):
    """
    Extracts a segment around a target frequency from FFT-transformed data.
    
    Parameters:
    - fft_data: A single FFT-transformed data array.
    - target_frequency: The frequency at which to center the extracted segment.
    - segment_length: The number of data points in each segment.
    
    Returns:
    - segment: Extracted segment array.
    """
    target_index = int(target_frequency * 2 + 2)  # Adjust index based on sampling rate if needed
    start = max(0, target_index - segment_length // 2 - 1)
    end = min(len(fft_data), target_index + segment_length // 2 - 1)
    segment = fft_data[start:end]

    # Pad segment if it's shorter than the specified segment length
    if len(segment) < segment_length:
        segment = np.pad(segment, (0, segment_length - len(segment)), 'constant')

    return segment

def normalize_segment(segment, method='z-score'):
    """
    Normalizes a segment using the specified method.
    
    Parameters:
    - segment: Array representing the segment to normalize.
    - method: Normalization method, either 'z-score' (standardization) or 'min-max'.
    
    Returns:
    - normalized_segment: The normalized segment array.
    - stats: Normalization statistics (mean and std for z-score, min and max for min-max).
    """
    if method == 'z-score':
        mean = np.mean(segment)
        std = np.std(segment)
        normalized_segment = (segment - mean) / (std + 1e-8)  # Avoid division by zero
        stats = (mean, std)
    elif method == 'min-max':
        min_val = np.min(segment)
        max_val = np.max(segment)
        normalized_segment = (segment - min_val) / (max_val - min_val + 1e-8)  # Avoid division by zero
        stats = (min_val, max_val)
    else:
        raise ValueError("Normalization method must be either 'z-score' or 'min-max'")

    return normalized_segment, stats

def extract_and_normalize_peak_segments(fft_data, segment_length=20, target_frequency=50, method='z-score'):
    """
    Extracts and normalizes peak segments from FFT-transformed data based on a target frequency.
    
    Parameters:
    - fft_data: A 2D array of FFT-transformed data (each row is a different sample).
    - segment_length: The number of data points in each segment.
    - target_frequency: The frequency at which to center the extracted segment.
    - method: Normalization method, either 'z-score' (standardization) or 'min-max'.
    
    Returns:
    - peak_segments: Array of normalized peak segments centered around target frequency.
    - stats: Array of tuples containing normalization statistics for each segment
             (mean and std for z-score, min and max for min-max).
    """
    peak_segments = []
    stats = []

    for spec in fft_data:
        segment = extract_segment(spec, target_frequency, segment_length)
        normalized_segment, segment_stats = normalize_segment(segment, method)
        peak_segments.append(normalized_segment)
        stats.append(segment_stats)

    return np.array(peak_segments), np.array(stats)


def insert_synthetic_peaks(normal_data, synthetic_peaks, target_frequency, segment_length):
    """
    Inserts synthetic peaks into normal data samples at the target frequency location.
    
    Parameters:
    - normal_data: Array of normal (unmodified) data samples.
    - synthetic_peaks: Array of synthetic peak segments to insert.
    - target_frequency: The frequency location where peaks should be inserted.
    - segment_length: The length of the peak segment to be inserted.
    
    Returns:
    - augmented_data: Array of augmented data samples with synthetic peaks inserted.
    """
    augmented_data = []
    target_index = int(target_frequency * 2 + 2)

    for data_sample in normal_data:
        data_sample = data_sample.copy()
        for peak in synthetic_peaks:
            start = max(0, target_index - segment_length // 2)
            end = min(len(data_sample), start + segment_length)
            if end - start < segment_length:
                peak = peak[:end - start]
            data_sample[start:end] = peak
        augmented_data.append(data_sample)

    return np.array(augmented_data)

def create_datasets(normal_data_dir, anomalous_data_dir, batch_size, num_classes=2):
    """
    Loads data from directories, assigns labels, shuffles, and creates PyTorch data loaders.
    
    Parameters:
    - normal_data_dir: Directory containing CSV files of normal data samples.
    - anomalous_data_dir: Directory containing CSV files of anomalous data samples.
    - batch_size: Number of samples per batch for the data loader.
    - num_classes: Number of classes for classification (default is 2 for binary).
    
    Returns:
    - train_loader: DataLoader for training.
    - test_loader: DataLoader for testing.
    """
    # Load data from each directory
    normal_data = load_data_from_directory(normal_data_dir)
    anomalous_data = load_data_from_directory(anomalous_data_dir)

    # Assign labels
    normal_labels = np.zeros(len(normal_data), dtype=int)
    anomalous_labels = np.ones(len(anomalous_data), dtype=int)

    # Combine and shuffle data and labels
    data = np.concatenate((normal_data, anomalous_data), axis=0)
    labels = np.concatenate((normal_labels, anomalous_labels), axis=0)
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    # Reshape for PyTorch (samples, channels, length)
    data = data.reshape(data.shape[0], 1, -1)

    # Convert to tensors
    data_tensor = torch.tensor(data, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Create dataset and split into training and testing
    dataset = TensorDataset(data_tensor, labels_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def plot_random_segments(peak_segments, num_segments=10, labels=None, max_points=1000, axis_labels=None, x_values=None):
    """
    Plots random segments from the provided segments.

    Parameters:
    - peak_segments: Array of segments to plot.
    - num_segments: Number of random segments to plot.
    - labels: Optional list of custom labels for the segments.
    - max_points: Maximum number of points to plot per segment (downsamples if necessary).
    - axis_labels: Optional tuple (x_label, y_label) for x and y axis labels.
    - x_values: Optional array of x-values corresponding to the peak segments.
    """
    if len(peak_segments) < num_segments:
        raise ValueError("Not enough segments to plot")

    random_indices = np.random.choice(len(peak_segments), num_segments, replace=False)
    selected_segments = peak_segments[random_indices]

    if labels is not None:
        if len(labels) != len(peak_segments):
            raise ValueError("Length of labels must match the length of peak_segments")
        selected_labels = [labels[i] for i in random_indices]
    else:
        selected_labels = [f"Segment {i}" for i in random_indices]

    # Setup subplot grid
    fig, axes = plt.subplots(num_segments // 2, 2, figsize=(15, 5 * (num_segments // 2)))
    axes = axes.flatten()

    x_label, y_label = axis_labels if axis_labels else ('Frequency (Hz)', 'Power (dB)')

    for i, ax in enumerate(axes):
        segment = selected_segments[i]

        # Downsample if necessary and max_points is not None
        if max_points is not None and len(segment) > max_points:
            segment = resample(segment, max_points)
            if x_values is not None:
                if len(x_values.shape) == 1:
                    x_segment = resample(x_values, max_points)
                else:
                    x_segment = resample(x_values[random_indices[i]], max_points)
            else:
                x_segment = np.linspace(0, len(segment), len(segment))
        else:
            if x_values is not None:
                if len(x_values.shape) == 1:
                    x_segment = x_values[:len(segment)]
                else:
                    x_segment = x_values[random_indices[i]]
            else:
                x_segment = np.linspace(0, len(segment), len(segment))

        ax.plot(x_segment, segment, linewidth=1.5, color='blue')
        ax.set_title(selected_labels[i], fontsize=10, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()



def plot_random_files_from_directory(directory, num_files=8):
    """
    Plots data from a specified number of random .csv files in the directory.
    
    Parameters:
    directory (str): Path to the directory containing .csv files.
    num_files (int): Number of files to randomly select and plot.
    """
    
    # List .csv files in the directory
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    
    # Select random files based on num_files or available files if fewer
    random_files = random.sample(files, min(num_files, len(files)))
    num_plots = len(random_files)

    # Calculate rows and columns needed to display the plots
    cols = min(4, num_plots)  # Set a maximum of 4 columns
    rows = math.ceil(num_plots / cols)  # Determine rows based on the number of plots

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten() if num_plots > 1 else [axes]  # Flatten axes array if more than one plot

    for i, file in enumerate(random_files):
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)
        data = df['Data'].values

        axes[i].plot(data)
        axes[i].set_title(file)
        axes[i].set_xlabel('Index')
        axes[i].set_ylabel('Value')

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

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

def add_smoothed_peak_to_files(input_directory, output_directory, peak_height_hybrid, peak_center_hybrid, peak_base_width, kernel_size):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    for csv_file in glob.glob(input_directory + '/*.csv'):
        data = pd.read_csv(csv_file)

        triangle_peak = np.zeros(len(data))
        half_base_width = peak_base_width // 2
        start_index = peak_center_hybrid - half_base_width
        end_index = start_index + peak_base_width
        triangle_peak[start_index:start_index + half_base_width] = np.linspace(0, peak_height_hybrid, half_base_width)
        triangle_peak[start_index + half_base_width:end_index] = np.linspace(peak_height_hybrid, 0, peak_base_width - half_base_width)

        gaussian_kernel = np.exp(-np.linspace(-2, 2, kernel_size)**2)
        gaussian_kernel /= gaussian_kernel.sum() 

        smoothed_peak = convolve(triangle_peak, gaussian_kernel, mode='same')

        data_with_smoothed_peak = data.iloc[:, 0] + smoothed_peak

        return data_with_smoothed_peak