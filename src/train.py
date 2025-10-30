# src/train.py
"""Training loops (standard and epoch-cached)."""
import copy
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src.utils import save_best

logger = logging.getLogger(__name__)


def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader],
    device: torch.device,
    num_epochs: int = 10,
    initial_lr: float = 1e-3,
    patience: int = 15,
    checkpoint_path: Optional[str | Path] = None,
) -> torch.nn.Module:
    """Standard training loop with validation, early-stopping, and checkpointing.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train. ``forward`` must return logits.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader or None
        Validation data loader. If ``None``, no early stopping or checkpoints
        are performed based on validation metrics.
    device : torch.device
        Target device.
    num_epochs : int, default=10
        Maximum number of epochs.
    initial_lr : float, default=1e-3
        Initial learning rate for AdamW.
    patience : int, default=15
        Early stopping patience based on validation accuracy.
    checkpoint_path : str or None, optional
        Output directory for saving the best model. If ``None``, defaults
        to ``'../res/checkpoints'``.

    Returns
    -------
    torch.nn.Module
        The model with best-validation weights loaded.

    Notes
    -----
    - Uses ``CosineAnnealingWarmRestarts`` scheduler.
    - Best model is determined by **highest validation accuracy**.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    best_val_acc = 0.0
    best_state_dict = None
    epochs_no_improve = 0

    # Training loop
    for epoch in range(1, num_epochs + 1):
        # ── Training ──
        model.train()
        running_loss = 0.0

        num_batches = len(train_loader)
        with tqdm(
            total=num_batches, desc=f"Epoch {epoch}/{num_epochs}", unit="batch"
        ) as pbar:
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
        logger.info(f"Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # ── Validation ──
        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            correct = total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss_sum += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            avg_val_loss = val_loss_sum / len(val_loader.dataset)
            val_acc = correct / total
            logger.info(
                f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_acc*100:.2f}%"
            )

            # ── checkpoint / early-stopping ────────────────────────────
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state_dict = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
                check_path = checkpoint_path or "../res/checkpoints"
                num_classes = model.num_classes
                save_best(
                    model,
                    epoch,
                    best_val_acc,
                    mode=("binary" if num_classes == 2 else "multiclass"),
                    out_dir=check_path,
                )
                logger.info(f"  [✓] best model saved → {check_path}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(f"  → Early stopping (no improve ≥ {patience})")
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
    val_loader: Optional[torch.utils.data.DataLoader],
    device: torch.device,
    num_epochs: int = 20,
    batch_size: int = 512,
    initial_lr: float = 1e-3,
    patience: int = 15,
    checkpoint_path: Optional[str | Path] = None,
    history_path: Optional[str | Path] = None,
) -> torch.nn.Module:
    """Training loop for datasets that refresh synthetic labels each epoch.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train. ``forward`` must return logits.
    cached_dataset : EpochCachedDataset-like
        Dataset that implements ``refresh(epoch: int)`` and can be wrapped by a
        DataLoader each epoch.
    val_loader : DataLoader or None
        Validation data loader.
    device : torch.device
        Target device.
    num_epochs : int, default=20
        Maximum number of epochs.
    batch_size : int, default=512
        Training batch size.
    initial_lr : float, default=1e-3
        Initial learning rate for AdamW.
    patience : int, default=15
        Early stopping patience based on validation accuracy.
    checkpoint_path : str or None, optional
        Directory to save the best model.
    history_path : str or Path or None, optional
        If given, CSV with ``[epoch, train_loss, val_loss]`` will be written.

    Returns
    -------
    torch.nn.Module
        The model with best-validation weights loaded.

    Notes
    -----
    - At the start of each epoch, ``cached_dataset.refresh(epoch)`` is called.
    - Uses ``CosineAnnealingWarmRestarts`` scheduler.
    """
    # Loss and optimizer setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    # History containers
    train_losses = []
    val_losses = []
    epochs = []

    best_val_acc = 0.0
    best_state_dict = None
    epochs_no_improve = 0
    using_cuda = device.type == "cuda"

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
        total = 0

        num_batches = len(train_loader)
        with tqdm(
            total=num_batches, desc=f"Epoch {epoch}/{num_epochs}", unit="batch"
        ) as pbar:
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * x.size(0)
                _, preds = out.max(1)
                running_corr += (preds == y).sum().item()
                total += x.size(0)

                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

        train_loss = running_loss / total
        train_acc = running_corr / total
        train_losses.append(train_loss)
        logger.info(
            f"Epoch [{epoch}/{num_epochs}] train-loss: {train_loss:.4f}  acc: {train_acc:6.2%}"
        )

        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_corr = 0
            val_total = 0
            with torch.no_grad():
                for xv, yv in val_loader:
                    xv, yv = xv.to(device), yv.to(device)
                    outv = model(xv)
                    val_loss_sum += criterion(outv, yv).item() * xv.size(0)
                    _, pv = outv.max(1)
                    val_corr += (pv == yv).sum().item()
                    val_total += yv.size(0)

            val_loss = val_loss_sum / val_total
            val_acc = val_corr / val_total
            val_losses.append(val_loss)
            logger.info(f"  → val-loss: {val_loss:.4f}  acc: {val_acc:6.2%}")

            # ── checkpoint / early-stopping ──────────────────────
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state_dict = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
                check_path = checkpoint_path or "../res/checkpoints"
                num_classes = model.num_classes
                save_best(
                    model,
                    epoch,
                    best_val_acc,
                    mode=("binary" if num_classes == 2 else "multiclass"),
                    out_dir=check_path,
                )
                logger.info(f"  [✓] best model saved → {check_path}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(f"  → Early stopping (no improve ≥ {patience})")
                    break
        else:
            # No validation loader: record placeholder
            val_losses.append(None)

        # ── scheduler step at end of epoch ─────────────────────
        scheduler.step()

    # ── save loss history if requested ───────────────────────────
    if history_path:
        history_path = Path(history_path)
        df = pd.DataFrame(
            {"epoch": epochs, "train_loss": train_losses, "val_loss": val_losses}
        )
        df.to_csv(history_path, index=False)
        logger.info(f"Saved training history to {history_path}")

    # ── load best weights before returning ────────────────────────
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    return model
