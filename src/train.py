import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np
from src.models import vae_loss
from tqdm.auto import tqdm


def train_vae(model, dataloader, optimizer, device='cpu', num_epochs=20):
    """
    Trains the VAE model with a single progress bar spanning all epochs.

    Parameters:
    - model: VAE model instance.
    - dataloader: DataLoader for training data.
    - optimizer: Optimizer instance.
    - device: Device to run the training on ('cpu' or 'cuda').
    - num_epochs: Number of epochs to train.
    """
    model.to(device)

    total_steps = num_epochs * len(dataloader)
    progress_bar = tqdm(total=total_steps, desc='Training', unit='batch')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            batch = batch[0].to(device)
            optimizer.zero_grad()
            x_decoded, mu, logvar = model(batch)
            loss = vae_loss(batch, x_decoded, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            progress_bar.update(1)
            progress_bar.set_postfix({
                'epoch': f'{epoch+1}/{num_epochs}',
                'loss': f'{loss.item():.4f}'
            })

        average_loss = total_loss / len(dataloader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Avg Loss: {average_loss:.5f}')

    progress_bar.close()
    print(f'Final Average Loss: {average_loss:.4f}')


def generate_synthetic_peaks(vae_model, num_samples, latent_dim, segment_mins, segment_maxs):
    """
    Generate synthetic peak segments using the trained VAE.
    """
    vae_model.eval()
    device = next(vae_model.parameters()).device

    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        x_decoded_input = vae_model.decoder_input(z)
        generated = vae_model.decoder(x_decoded_input)
        generated = generated.cpu().numpy()
        generated = generated.squeeze(1)  # Remove channel dimension

    denormalized_peaks = []
    for i, segment in enumerate(generated):
        min_val = segment_mins[i % len(segment_mins)]
        max_val = segment_maxs[i % len(segment_maxs)]
        denormalized_segment = segment * (max_val - min_val + 1e-8) + min_val
        denormalized_peaks.append(denormalized_segment)

    return np.array(denormalized_peaks)


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


def calculate_metrics(true_labels, predictions):
    """
    Calculates and prints confusion matrix and classification metrics.
    
    Parameters:
      - true_labels: list or array of ground truth labels.
      - predictions: list or array of predicted labels.
      
    Returns:
      - cm: Confusion matrix (as a NumPy array).
      - accuracy: Overall accuracy in percent.
      - precision: Weighted precision.
      - recall: Weighted recall.
      - f1_score: Weighted F1 score.
    """
    cm = confusion_matrix(true_labels, predictions)
    accuracy = 100.0 * np.sum(np.array(true_labels) == np.array(predictions)) / len(true_labels)
    precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    return cm, accuracy, precision, recall, f1_score


def train_resnet_model(model, train_loader, val_loader, device, num_epochs=10, initial_lr=0.001):
    """
    Trains the ResNet model with a progress bar and calls validation at the end.
    
    Parameters:
    -----------
    model : nn.Module
        The ResNet model to train.
    train_loader : DataLoader
        DataLoader for the training set.
    device : torch.device
        Device to run the training on ('cpu', 'cuda', etc.).
    num_epochs : int
        Number of epochs to train.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    best_val_acc = 0.0
    best_model_state = None

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        if val_loader is not None:
            # Validation step
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            avg_val_loss = val_loss / len(val_loader.dataset)
            val_acc = correct / total
            print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_acc*100:.2f}%')

            # Step the scheduler based on validation loss
            scheduler.step(avg_val_loss)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()
                print("Best model updated.")

            # Load best model state
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

    return model

def compute_mse(original_segment, denoised_segment):
    """
    Computes Mean Squared Error (MSE) between original and denoised segments.
    """
    return np.mean((original_segment - denoised_segment) ** 2)

def compute_snr(original_segment, denoised_segment):
    """
    Computes the Signal-to-Noise Ratio (SNR) in dB for one segment.
    SNR = 20 * log10(||original|| / ||original - denoised||).
    """
    numerator = np.linalg.norm(original_segment)
    denominator = np.linalg.norm(original_segment - denoised_segment) + 1e-12
    return 20 * np.log10(numerator / denominator)

def compute_metrics(original_segments, denoised_segments):
    """
    Computes the average MSE and average SNR across all segments.
    Returns (avg_mse, avg_snr).
    """
    mses = []
    snrs = []
    for orig_seg, den_seg in zip(original_segments, denoised_segments):
        mses.append(compute_mse(orig_seg, den_seg))
        snrs.append(compute_snr(orig_seg, den_seg))
    return np.mean(mses), np.mean(snrs)