import numpy as np
import shutil
import os
import random
import matplotlib.pyplot as plt

def dataloader_to_numpy(dataloader):
    """
    Converts a DataLoader to numpy arrays.

    Parameters:
    - dataloader: DataLoader containing data and labels.

    Returns:
    - all_data: Numpy array of data.
    - all_labels: Numpy array of labels.
    """
    all_data = []
    all_labels = []
    for inputs, labels in dataloader:
        all_data.append(inputs.numpy().reshape(inputs.size(0), -1))
        all_labels.append(labels.numpy())
    return np.concatenate(all_data), np.concatenate(all_labels)


def zip_folder(folder_path, zip_path):
    shutil.make_archive(zip_path, 'zip', folder_path)

def move_random_csv_files(input_dir, output_dir, n_files):
    """
    Randomly selects a specified number of .csv files from the input directory 
    and moves them to the output directory. If the output directory does not exist,
    it will be created automatically.

    Parameters:
    input_dir (str): The path to the directory containing the .csv files to be moved.
    output_dir (str): The path to the destination directory where the selected files will be moved.
    n_files (int): The number of .csv files to randomly select and move.
    
    Returns:
    list: A list of the moved file names.
    """
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # List all files in the input directory that end with .csv
    csv_files = [file for file in os.listdir(input_dir) if file.endswith('.csv')]
    
    # Check if there are enough files to select from
    if n_files > len(csv_files):
        raise ValueError(f"Requested {n_files} files, but only found {len(csv_files)} .csv files in {input_dir}.")

    # Randomly select n_files from the list of .csv files
    selected_files = random.sample(csv_files, n_files)
    
    # Move each selected file to the output directory
    moved_files = []
    for file_name in selected_files:
        src_path = os.path.join(input_dir, file_name)
        dest_path = os.path.join(output_dir, file_name)
        shutil.move(src_path, dest_path)
        moved_files.append(file_name)

    return moved_files

def rename_folder(current_folder_path, new_folder_path):
    """
    Renames a folder by moving it from the current path to a new path with a different name.

    Parameters:
    current_folder_path (str): The path to the existing folder.
    new_folder_path (str): The path with the new folder name.
    """
    try:
        os.rename(current_folder_path, new_folder_path)
        return f"Folder renamed from '{current_folder_path}' to '{new_folder_path}'."
    except FileNotFoundError:
        return f"Error: The folder '{current_folder_path}' does not exist."
    except Exception as e:
        return f"An error occurred: {e}"
    

def plot_spectrum(df, freq_col="Frequency (Hz)", amp_col="Amplitude",
                          title="Frequency Spectrum", xlabel="Frequency (Hz)",
                          ylabel="Amplitude (dB)", save_path=None):
    """
    Plots the frequency spectrum
    
    Parameters:
    - df: Pandas DataFrame containing frequency and amplitude data.
    - freq_col: Column name for frequency data in the DataFrame (default: "Frequency (Hz)").
    - amp_col: Column name for amplitude data in the DataFrame (default: "Amplitude").
    - title: Title of the plot (default: "Frequency Spectrum").
    - xlabel: Label for the x-axis (default: "Frequency (Hz)").
    - ylabel: Label for the y-axis (default: "Amplitude (dB)").
    - save_path: Optional file path to save the plot as an image.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the spectrum
    ax.plot(df[freq_col], df[amp_col], label='Spectrum', color='blue')
    
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