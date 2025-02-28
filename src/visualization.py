import matplotlib.pyplot as plt
import numpy as np


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