# inference.py
import numpy as np
import torch
from tqdm.auto import tqdm


def inference_model(model, test_loader, device):
    """
    Runs inference on the test loader and returns true and predicted labels.

    Parameters:
      - model: Trained model.
      - test_loader: DataLoader for the test set.
      - device: Device to run inference on.

    Returns:
      - all_labels: np.ndarray of shape (n_samples,)
      - all_predictions: np.ndarray of shape (n_samples,)
    """
    model.eval()
    n_samples = len(test_loader.dataset)
    all_labels = np.empty(n_samples, dtype=int)
    all_predictions = np.empty(n_samples, dtype=int)
    idx = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Inference", unit="batch"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            batch_size = labels.size(0)
            all_labels[idx : idx + batch_size] = labels.cpu().numpy()
            all_predictions[idx : idx + batch_size] = preds.cpu().numpy()
            idx += batch_size

    return all_labels, all_predictions
