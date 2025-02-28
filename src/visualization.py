
import os
import random
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample


def plot_spectrum(df, freq_col="Frequency (Hz)", amp_col="Amplitude",
                  title="Frequency Spectrum", xlabel="Frequency (Hz)",
                  ylabel="Amplitude (dB)", highlight_freqs=None, method="interpolate",
                  save_path=None):
    """
    Plots the frequency spectrum with optional highlighted points using interpolation or closest point methods.

    Parameters:
    - df: Pandas DataFrame containing frequency and amplitude data.
    - freq_col: Column name for frequency data (default: "Frequency (Hz)").
    - amp_col: Column name for amplitude data (default: "Amplitude").
    - title: Title of the plot (default: "Frequency Spectrum").
    - xlabel: Label for the x-axis (default: "Frequency (Hz)").
    - ylabel: Label for the y-axis (default: "Amplitude (dB)").
    - highlight_freqs: List of frequency values to highlight as points.
    - method: Method to handle frequencies not in DataFrame. Options: "interpolate" or "closest".
              "interpolate" uses linear interpolation between known points.
              "closest" finds the closest existing frequency in df.
    - save_path: Optional file path to save the plot as an image.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort the DataFrame by frequency to ensure correct order for interpolation
    df_sorted = df.sort_values(by=freq_col)
    freqs = df_sorted[freq_col].values
    amps = df_sorted[amp_col].values

    # Plot the base spectrum line
    ax.plot(freqs, amps, label='Spectrum', color='blue')

    # If highlight frequencies are provided, calculate or find their amplitude
    if highlight_freqs is not None and len(highlight_freqs) > 0:
        highlight_amps = []
        for hf in highlight_freqs:
            if hf in freqs:
                # Exact match found
                hf_amp = amps[freqs.tolist().index(hf)]
            else:
                # No exact match, handle according to method
                if method == "interpolate":
                    # Use np.interp to interpolate
                    hf_amp = np.interp(hf, freqs, amps)
                elif method == "closest":
                    # Find closest frequency
                    idx_closest = np.argmin(np.abs(freqs - hf))
                    hf_amp = amps[idx_closest]
                else:
                    raise ValueError("Invalid method. Use 'interpolate' or 'closest'.")

            highlight_amps.append(hf_amp)

        # Plot the highlight points
        ax.scatter(highlight_freqs, highlight_amps, color='red', s=50, marker='o', 
                   label='Highlighted Points')

    # Set titles and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Add grid lines
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Reference line at 0 dB (optional)
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--')

    # Add a legend
    ax.legend(loc='best')

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


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
    return fig, ax


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
