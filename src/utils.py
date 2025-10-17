import numpy as np
import shutil
import os
import random
from pathlib import Path
import torch
import datetime

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def create_experiment_folder(base_dir="../res", exp_name="engine_1"):
    # e.g., res/2024-05-21_15-03-25_engine_1/
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{timestamp}_{exp_name}"
    full_path = Path(base_dir) / folder_name
    full_path.mkdir(parents=True, exist_ok=False)
    print(f"[✓] Experiment results folder: {full_path.resolve()}")
    return full_path


def save_best(model, epoch, metric, mode, out_dir):
    """
    Persist best model.
      • model … the nn.Module
      • metric … the *higher-is-better* score (e.g. val-accuracy)
      • mode … "binary" | "multiclass"
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"best_{mode}.pth"

    torch.save({
        "epoch":  epoch,
        "metric": metric,
        "state":  model.state_dict(),
    }, fname)

    print(f"[✓]   New best ({mode}) saved →  {fname}  (metric={metric:.4f})")

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
        batch_inputs = inputs.detach().cpu().numpy().reshape(inputs.size(0), -1)
        batch_labels = labels.detach().cpu().numpy()
        all_data.append(batch_inputs)
        all_labels.append(batch_labels)
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

def generate_induction_motor_signal(
    signal_length=1000,
    sampling_rate=1000,
    fundamental_freq=50,
    noise_scale=0.2,
    add_harmonics=True,
    add_spikes=True,
    add_noise=True,
    add_modulation=True
):
    """
    Generate a synthetic signal simulating induction motor current with selectable problems.
    
    Parameters:
        signal_length (int): Number of samples in the signal.
        sampling_rate (int): Sampling rate in Hz.
        fundamental_freq (float): Fundamental frequency in Hz.
        noise_scale (float): Standard deviation of Gaussian noise.
        add_harmonics (bool): If True, includes harmonic components.
        add_spikes (bool): If True, introduces transient spikes.
        add_noise (bool): If True, adds Gaussian noise.
        add_modulation (bool): If True, applies amplitude modulation.
    
    Returns:
        tuple: (clean_signal, noisy_signal)
            - clean_signal: The base signal without any disturbances.
            - noisy_signal: The signal with selected disturbances added.
    """
    t = np.linspace(0, signal_length / sampling_rate, signal_length)
    
    # Fundamental signal
    fundamental = np.sin(2 * np.pi * fundamental_freq * t)
    
    # Harmonics
    harmonics = (
        0.1 * np.sin(2 * np.pi * 2 * fundamental_freq * t) +
        0.1 * np.sin(2 * np.pi * 3 * fundamental_freq * t) +
        0.1 * np.sin(2 * np.pi * 4 * fundamental_freq * t)
    ) if add_harmonics else 0
    
    # Amplitude modulation
    modulation = (1 + 0.1 * np.sin(2 * np.pi * 0.5 * t)) if add_modulation else 1
    
    # Combine base signal
    clean_signal = modulation * (fundamental + harmonics)
    
    # Initialize noisy signal
    noisy_signal = clean_signal.copy()
    
    # Gaussian noise
    if add_noise:
        noise = np.random.normal(scale=noise_scale, size=signal_length)
        noisy_signal += noise
    
    # Transient spikes 
    if add_spikes:
        for _ in range(5):  # Introduce 5 random spikes
            spike_index = np.random.randint(0, signal_length)
            noisy_signal[spike_index:spike_index + 10] += np.random.normal(scale=3.0, size=10)
    
    return clean_signal, noisy_signal