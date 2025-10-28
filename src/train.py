# train.py
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import copy
from src.utils import save_best
from pathlib import Path
import pandas as pd

import logging
logger = logging.getLogger(__name__)


def train_model(model, train_loader, val_loader, device, num_epochs=10, initial_lr=1e-3, patience=15, checkpoint_path: str | None = None, ):
    """
    Trains a model with a progress bar and calls validation at the end.
    • CosineAnnealingWarmRestarts scheduler
    • Early stopping
    • If `checkpoint_path` is given, the best weights are saved there.
    
    Parameters:
    -----------
    model : nn.Module
        The model to train.
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

    best_val_acc      = 0.0
    best_state_dict   = None
    epochs_no_improve = 0

    # Training loop
    for epoch in range(1, num_epochs + 1):
        # ── Training ──
        model.train()
        running_loss = 0.0

        num_batches = len(train_loader)
        with tqdm(total=num_batches, desc=f'Epoch {epoch}/{num_epochs}', unit='batch') as pbar:
            for batch_idx, (inputs, labels) in enumerate(train_loader):
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
        logger.info(f'Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.4f}')

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
            logger.info(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_acc*100:.2f}%')

            # ── checkpoint / early-stopping ────────────────────────────
            if val_acc > best_val_acc:
                best_val_acc    = val_acc
                best_state_dict = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0

                if checkpoint_path:
                    check_path = checkpoint_path
                else:
                    check_path = '../res/checkpoints'

                num_classes = model.num_classes
                save_best(model, epoch, best_val_acc,
                            mode=('binary' if num_classes == 2 else 'multiclass'),
                            out_dir=check_path)
                logger.info(f'  [✓] best model saved → {check_path}')
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(f'  → Early stopping (no improve ≥ {patience})')
                    break

        # ── Scheduler step at end of epoch ──
        scheduler.step()

    # ── Load best model before returning ──
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    return model

def train_epoch_cached(
    model: torch.nn.Module,
    cached_dataset,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_epochs: int           = 20,
    batch_size: int           = 512,
    initial_lr: float         = 1e-3,
    patience: int             = 15,
    checkpoint_path: str | None      = None,
    history_path: str | Path | None  = None,
):
    """
    Train a model on an EpochCachedDataset.

    • `cached_dataset.refresh(epoch)` is called at the start of every epoch to
      redraw which normal windows become synthetic faults, keeping labels stable *within* the epoch.
    • Early stopping and CosineAnnealingWarmRestarts are preserved.
    • If `checkpoint_path` is given, the best weights are saved there.
    • If `history_path` is provided, a CSV with columns [train_loss, val_loss] is written at the end.

    Returns
    -------
    model  -  with the best-validation weights loaded.
    """
    # Loss and optimizer setup
    criterion  = nn.CrossEntropyLoss()
    optimizer  = torch.optim.AdamW(model.parameters(),
                                   lr=initial_lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2)

    # History containers
    train_losses = []
    val_losses   = []
    epochs       = []

    best_val_acc      = 0.0
    best_state_dict   = None
    epochs_no_improve = 0
    using_cuda = device.type == 'cuda'

    for epoch in range(1, num_epochs + 1):
        # ── 1.  Refresh synthetic map & build DataLoader ─────────────
        cached_dataset.refresh(epoch)
        train_loader = torch.utils.data.DataLoader(
            cached_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(epoch),
            # pin_memory=using_cuda,
            # num_workers=2 if using_cuda else 0
        )

        # ── 2.  Training phase ───────────────────────────────────
        model.train()
        epochs.append(epoch)
        running_loss = 0.0
        running_corr = 0
        total        = 0

        num_batches = len(train_loader)
        with tqdm(total=num_batches,
                  desc=f'Epoch {epoch}/{num_epochs}',
                  unit='batch') as pbar:
            for batch_idx, (x, y) in enumerate(train_loader):
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
        train_losses.append(train_loss)
        logger.info(f'Epoch [{epoch}/{num_epochs}] '
                    f'train-loss: {train_loss:.4f}  acc: {train_acc:6.2%}')

        # ── 3.  Validation phase ─────────────────────────────────
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
            val_losses.append(val_loss)
            logger.info(f'  → val-loss: {val_loss:.4f}  acc: {val_acc:6.2%}')

            # ── checkpoint / early-stopping ──────────────────────
            if val_acc > best_val_acc:
                best_val_acc    = val_acc
                best_state_dict = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0

                if checkpoint_path:
                    check_path = checkpoint_path
                else:
                    check_path = '../res/checkpoints'

                num_classes = model.num_classes
                save_best(model, epoch, best_val_acc,
                          mode=('binary' if num_classes == 2 else 'multiclass'),
                          out_dir=check_path)
                logger.info(f'  [✓] best model saved → {check_path}')
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(f'  → Early stopping (no improve ≥ {patience})')
                    break
        else:
            # No validation loader: record placeholder
            val_losses.append(None)

        # ── scheduler step at end of epoch ─────────────────────
        scheduler.step()

    # ── save loss history if requested ───────────────────────────
    if history_path:
        history_path = Path(history_path)
        df = pd.DataFrame({
            'epoch': epochs,
            'train_loss': train_losses,
            'val_loss':   val_losses,
        })
        df.to_csv(history_path, index=False)
        logger.info(f"Saved training history to {history_path}")

    # ── load best weights before returning ────────────────────────
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    return model
