"""1D-CNN that classifies a flow's packet sequence into the application that produced it."""

from __future__ import annotations

import copy
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluation.metrics import evaluate_metrics


class FlowCNN(nn.Module):
    """Maps ``(batch, in_channels, seq_len)`` packet sequences to ``(batch, n_classes)`` logits."""

    def __init__(self, in_channels: int = 2, n_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),  # strongest response of each filter anywhere in the sequence
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))  # raw logits; CrossEntropyLoss applies softmax


def class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Inverse-frequency weights (``total / (n_classes * count)``) so small classes count as much as large ones."""
    counts = np.bincount(y, minlength=n_classes)
    return counts.sum() / (n_classes * np.maximum(counts, 1))


def predict(model: nn.Module, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """Predicted class index for each flow in ``X``."""
    model.eval()
    device = next(model.parameters()).device
    preds = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.as_tensor(X[start:start + batch_size], device=device)
            preds.append(model(batch).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds) if preds else np.empty(0, dtype=np.int64)


def train_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_names: Sequence[str],
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device: str | None = None,
    verbose: bool = True,
) -> tuple[FlowCNN, list[dict]]:
    """Train a ``FlowCNN`` and keep the weights from the epoch with the best validation macro-F1.

    Uses Adam and cross-entropy weighted by ``class_weights`` of the training labels. The
    validation set only chooses the epoch; it is never trained on.

    Returns:
        ``(model, history)``: the model with the best-epoch weights loaded, and one dict per
        epoch with ``epoch``, ``train_loss``, ``val_accuracy`` and ``val_f1_macro``.
    """
    if len(X_val) == 0:
        raise ValueError("The validation set is empty; it is needed to choose the best epoch.")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(class_names)

    torch.manual_seed(seed)
    model = FlowCNN(in_channels=X_train.shape[1], n_classes=n_classes).to(device)
    loader = DataLoader(
        TensorDataset(torch.as_tensor(X_train), torch.as_tensor(y_train, dtype=torch.long)),
        batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed),
    )
    weights = torch.as_tensor(class_weights(y_train, n_classes), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_f1, best_state, history = -1.0, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)

        val = evaluate_metrics(y_true=y_val, y_pred=predict(model, X_val), class_names=class_names)
        history.append({"epoch": epoch, "train_loss": total_loss / len(loader.dataset),
                        "val_accuracy": val["accuracy"], "val_f1_macro": val["f1_macro"]})
        if val["f1_macro"] > best_f1:
            best_f1, best_state = val["f1_macro"], copy.deepcopy(model.state_dict())
        if verbose:
            h = history[-1]
            print(f"Epoch {epoch:2d}/{epochs}  train_loss {h['train_loss']:.4f}  "
                  f"val_accuracy {h['val_accuracy']:.3f}  val_f1_macro {h['val_f1_macro']:.3f}")

    model.load_state_dict(best_state)
    return model, history
