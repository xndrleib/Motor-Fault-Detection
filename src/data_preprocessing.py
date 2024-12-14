import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt

from scipy.signal import convolve
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

    df = pd.DataFrame({'Time (ms)': time, 'Data': channel_1})
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


def time_to_freq_transform(df, f_sampling, db=True, cutoff_freq=None):
    """
    Transforms time-series data to frequency domain using FFT.
    Optionally applies a cutoff frequency.
    
    Parameters:
    - df: DataFrame containing the time series data in 'Data' column.
    - f_sampling: Sampling frequency of the data.
    - db: Boolean flag to convert FFT values to decibel scale.
    - cutoff_freq: Optional cutoff frequency to filter the results.
    
    Returns:
    - yf: Transformed frequency domain values.
    - freqs: Frequency bins.
    """
    y = df['Data']
    n = len(y)
    yf = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, d=1/f_sampling) # with Nyquist frequency
    if db:
        yf = 20 * np.log10(np.abs(yf))
    if cutoff_freq is not None:
        mask = freqs < cutoff_freq
        yf = yf[mask]
        freqs = freqs[mask]
    return yf, freqs


def process_time_series(file_path, output_dir, window_length=20000, shift=20, f_sampling=1.0, db=True, cutoff_freq=250):
    """
    Applies a sliding window over time series data, performs FFT on each window,
    and saves the transformed data.
    
    Parameters:
    - file_path: Path to the CSV file with time-series data.
    - output_dir: Directory to save the transformed window files.
    - window_length: Number of samples in each window.
    - shift: Number of samples to shift the window for each iteration.
    - f_sampling: Sampling frequency of the data.
    - db: Boolean flag to convert FFT values to decibel scale.
    - cutoff_freq: Frequency cutoff for filtering FFT results.
    """
    df = pd.read_csv(file_path)
    os.makedirs(output_dir, exist_ok=True)

    file_counter = 1
    for start in range(0, len(df) - window_length + 1, shift):
        window_df = df.iloc[start:start + window_length]
        yf, freqs = time_to_freq_transform(window_df, f_sampling, db=db, cutoff_freq=cutoff_freq)
        
        transformed_df = pd.DataFrame({
            'Frequency (Hz)': freqs,
            'Amplitude': yf
        })
        
        output_file_path = os.path.join(output_dir, f'Window_{file_counter}.csv')
        transformed_df.to_csv(output_file_path, index=False)
        file_counter += 1

    print(f'Results saved to {output_dir}')


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
    target_index = int(target_frequency * 2 + 2)  # Adjust index based on sampling rate if needed

    for spec in fft_data:
        # Define the segment around the target index
        start = max(0, target_index - segment_length // 2 - 1)
        end = min(len(spec), target_index + segment_length // 2 - 1)
        segment = spec[start:end]

        # Pad segment if it's shorter than the specified segment length
        if len(segment) < segment_length:
            segment = np.pad(segment, (0, segment_length - len(segment)), 'constant')

        # Normalize based on selected method
        if method == 'z-score':
            mean = np.mean(segment)
            std = np.std(segment)
            normalized_segment = (segment - mean) / (std + 1e-8)  # Avoid division by zero
            stats.append((mean, std))
        elif method == 'min-max':
            min_val = np.min(segment)
            max_val = np.max(segment)
            normalized_segment = (segment - min_val) / (max_val - min_val + 1e-8)  # Avoid division by zero
            stats.append((min_val, max_val))
        else:
            raise ValueError("Normalization method must be either 'z-score' or 'min-max'")

        peak_segments.append(normalized_segment)

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

def plot_random_segments(peak_segments, num_segments=10):
    """
    Plots random segments from the provided peak segments.

    Parameters:
    - peak_segments: Array of peak segments to plot.
    - num_segments: Number of random segments to plot.
    """
    if len(peak_segments) < num_segments:
        raise ValueError("Not enough segments to plot")

    random_indices = np.random.choice(len(peak_segments), num_segments, replace=False)
    selected_segments = peak_segments[random_indices]

    fig, axes = plt.subplots(num_segments // 2, 2, figsize=(15, 5 * (num_segments // 2)))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.plot(selected_segments[i])
        ax.set_title(f'Segment {random_indices[i]}')
        ax.set_xlabel('Frequency Index')
        ax.set_ylabel('Amplitude')

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

        indices = np.arange(len(data))
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