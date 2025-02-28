import torch
import torch.nn as nn
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)

    total_steps = len(train_loader) * num_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=initial_lr,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos'
    )

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
                scheduler.step()

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

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()
                print("Best model updated.")

            # Load best model state
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

    return model



def vae_loss(x, x_decoded, mu, logvar, beta=1.0, smoothness_weight=0.0, delta=1.0):
    """
    VAE loss using Huber (Smooth L1) for reconstruction + KL divergence + optional TV smoothing.
    
    Args:
      x (Tensor): Original input of shape [B, 1, L].
      x_decoded (Tensor): Model's reconstruction of shape [B, 1, L].
      mu (Tensor): Mean vector from the encoder.
      logvar (Tensor): Log variance from the encoder.
      beta (float): Weight for the KL term (for beta-VAE).
      smoothness_weight (float): If > 0, apply total variation penalty at that weight.
      delta (float): Huber threshold (nn.SmoothL1Loss).
    """
    # 1) Huber (Smooth L1) reconstruction
    huber_fn = nn.SmoothL1Loss(reduction='sum', beta=delta)
    recon_loss = huber_fn(x_decoded, x) / x.size(0)

    # 2) KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    # 3) Optional total variation for smoothing
    tv_loss = 0.0
    if smoothness_weight > 0:
        tv_loss = total_variation_loss(x_decoded, weight=smoothness_weight)

    return recon_loss + beta * kl_loss + tv_loss

def total_variation_loss(signal, weight=1e-3):
    """
    Encourages smoothness in the 1D output signal.
    signal shape: [B, 1, L]
    """
    diff = signal[:, :, 1:] - signal[:, :, :-1]
    tv = torch.mean(torch.abs(diff))
    return weight * tv
