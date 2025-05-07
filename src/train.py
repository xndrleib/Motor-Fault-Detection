import torch
import torch.nn as nn
from tqdm.auto import tqdm
import copy
from src.utils import save_best

def train_vae(model, dataloader, optimizer, device='cpu', num_epochs=20):
    """
    Trains the VAE model with a single progress bar spanning all epochs.

    Parameters:
    - model: VAE model instance.
    - dataloader: DataLoader for training data.
    - optimizer: Optimizer instance.
    - device: Device to run the training on.
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

def train_resnet_model(model, train_loader, val_loader, device, num_epochs=10, initial_lr=1e-3, patience=15):
    """
    Trains the ResNet model with a progress bar and calls validation at the end.
    - CosineAnnealingWarmRestarts scheduler
    - Early stopping
    
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
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=initial_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2)

    best_val_acc = 0.0
    best_state   = None
    epochs_no_improve = 0

    # Training loop
    for epoch in range(1, num_epochs + 1):
        # ── Training ──
        model.train()
        running_loss = 0.0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{num_epochs}', unit='batch') as pbar:
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
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.4f}')

        # ── Validation ──
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            correct = total = 0
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

            # ── Check for improvement ──
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0

                num_classes = model.fc.out_features
                save_best(model, epoch+1, best_val_acc,
                          mode=("binary" if num_classes == 2 else "multiclass"),
                          out_dir="../res/checkpoints")
                print("→ New best model saved")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"→ Early stopping after {patience} epochs with no improvement.")
                    break

        # ── Scheduler step at end of epoch ──
        scheduler.step()

    # ── Load best model before returning ──
    if best_state is not None:
        model.load_state_dict(best_state)
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


def train_resnet_epoch_cached(
    model: torch.nn.Module,
    cached_dataset,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_epochs: int          = 20,
    batch_size: int          = 512,
    initial_lr: float        = 1e-3,
    patience: int            = 15,
    checkpoint_path: str | None = None,
):
    """
    Train a ResNet on an EpochCachedDataset.

    • `cached_dataset.refresh(epoch)` is called at the start of every epoch to
      redraw which normal windows become synthetic faults, keeping labels
      stable *within* the epoch.
    • Early stopping and CosineAnnealingWarmRestarts are preserved.
    • If `checkpoint_path` is given, the best weights are saved there.

    Returns
    -------
    model  -  with the best-validation weights loaded.
    """
    criterion  = nn.CrossEntropyLoss()
    optimizer  = torch.optim.AdamW(model.parameters(),
                                   lr=initial_lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2)

    best_val_acc     = 0.0
    best_state_dict  = None
    epochs_no_improv = 0

    for epoch in range(1, num_epochs + 1):
        # ── 1.  Refresh synthetic map & build DataLoader ──────────────────
        cached_dataset.refresh(epoch)
        train_loader = torch.utils.data.DataLoader(
            cached_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(epoch),
        )

        # ── 2.  Training phase ───────────────────────────────────────────
        model.train()
        running_loss = 0.0
        running_corr = 0
        total        = 0

        with tqdm(total=len(train_loader),
                  desc=f'Epoch {epoch}/{num_epochs}',
                  unit='batch') as pbar:
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * x.size(0)
                _, preds      = out.max(1)
                running_corr += (preds == y).sum().item()
                total        += x.size(0)

                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

        train_loss = running_loss / total
        train_acc  = running_corr / total
        print(f'Epoch [{epoch}/{num_epochs}] '
              f'train-loss: {train_loss:.4f}  acc: {train_acc:6.2%}')

        # ── 3.  Validation phase ────────────────────────────────────────
        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_corr     = 0
            val_total    = 0
            with torch.no_grad():
                for xv, yv in val_loader:
                    xv, yv = xv.to(device), yv.to(device)
                    outv   = model(xv)
                    val_loss_sum += criterion(outv, yv).item() * xv.size(0)
                    _, pv = outv.max(1)
                    val_corr  += (pv == yv).sum().item()
                    val_total += yv.size(0)

            val_loss = val_loss_sum / val_total
            val_acc  = val_corr / val_total
            print(f'  → val‑loss: {val_loss:.4f}  acc: {val_acc:6.2%}')

            # ── checkpoint / early‑stopping ────────────────────────────
            if val_acc > best_val_acc:
                best_val_acc    = val_acc
                best_state_dict = copy.deepcopy(model.state_dict())
                epochs_no_improv = 0
                if checkpoint_path:
                    torch.save(best_state_dict, checkpoint_path)
                    print(f'  [✓] best model saved → {checkpoint_path}')
                else:
                    num_classes = model.fc.out_features
                    save_best(model, epoch, best_val_acc,
                              mode=('binary' if num_classes == 2 else 'multiclass'),
                              out_dir='../res/checkpoints')
            else:
                epochs_no_improv += 1
                if epochs_no_improv >= patience:
                    print(f'  → Early stopping (no improv ≥ {patience})')
                    break

        # ── scheduler step at end of epoch ──
        scheduler.step()

    # ── load best weights before returning ────────────────────────────────
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    return model
