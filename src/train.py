import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support
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

def train_resnet_model(model, train_loader, test_loader, device, num_epochs=10):
    """
    Trains and evaluates the ResNet model.

    Parameters:
    - model: ResNet model instance.
    - train_loader: DataLoader for training data.
    - test_loader: DataLoader for testing data.
    - device: Device to run the training on ('cpu' or 'cuda').
    - num_epochs: Number of epochs to train.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test set: {accuracy:.2f}%')

    precision, recall, f1_score, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted')
    print(f'F1 Score: {f1_score:.4f}')
