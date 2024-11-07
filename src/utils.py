import numpy as np

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