# src/inference.py
"""Inference utilities."""
import numpy as np
import torch
from tqdm.auto import tqdm


def inference_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    return_extra: bool = False,
):
    """Run batched inference and optionally return logits/probs/embeddings.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model; must return logits from ``forward``. If it implements
        ``forward_features``, embeddings will also be collected.
    test_loader : torch.utils.data.DataLoader
        DataLoader yielding ``(inputs, labels)``.
    device : torch.device
        Target device for inference.
    return_extra : bool, default=False
        If ``True``, return a dict with labels/preds/logits/probs/embeddings.
        If ``False``, return the legacy tuple ``(labels, preds)``.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        If ``return_extra`` is ``False``: ``(labels, preds)``.
    dict
        If ``return_extra`` is ``True``: a dictionary with:
        - ``labels`` : ``(N,)`` int
        - ``preds`` : ``(N,)`` int
        - ``logits`` : ``(N, C)`` float32
        - ``probs`` : ``(N, C)`` float32 (softmax)
        - ``embeddings`` : ``(N, D)`` float32 or ``None``

    Notes
    -----
    Softmax probabilities are computed with dimension ``dim=1``.
    """
    model.eval()
    n_samples = len(test_loader.dataset)
    all_labels = np.empty(n_samples, dtype=int)
    all_predictions = np.empty(n_samples, dtype=int)

    logits_list = []
    probs_list = []
    embeds_list = []

    idx = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Inference", unit="batch"):
            inputs = inputs.to(device)
            outputs = model(inputs)  # (B, C) logits
            probs = torch.softmax(outputs, dim=1)  # (B, C)

            feats = (
                model.forward_features(inputs)
                if hasattr(model, "forward_features")
                else None
            )
            if feats is not None:
                embeds_list.append(feats.cpu().numpy())

            logits_list.append(outputs.cpu().numpy())
            probs_list.append(probs.cpu().numpy())

            _, preds = torch.max(outputs, 1)
            b = labels.size(0)
            all_labels[idx : idx + b] = labels.cpu().numpy()
            all_predictions[idx : idx + b] = preds.cpu().numpy()
            idx += b

    if not return_extra:
        return all_labels, all_predictions

    logits = np.concatenate(logits_list, axis=0)
    probs = np.concatenate(probs_list, axis=0)
    embeddings = np.concatenate(embeds_list, axis=0) if len(embeds_list) > 0 else None

    return {
        "labels": all_labels,
        "preds": all_predictions,
        "logits": logits,
        "probs": probs,
        "embeddings": embeddings,
    }
