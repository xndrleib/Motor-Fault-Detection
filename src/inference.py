import torch
from tqdm.auto import tqdm

def inference_resnet_model(model, test_loader, device):
    """
    Runs inference on the test loader and returns true and predicted labels.
    
    Parameters:
      - model: Trained ResNet model.
      - test_loader: DataLoader for the test set.
      - device: Device to run inference on.
      
    Returns:
      - all_labels: list of true labels.
      - all_predictions: list of predicted labels.
    """
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Inference', unit='batch'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
    return all_labels, all_predictions