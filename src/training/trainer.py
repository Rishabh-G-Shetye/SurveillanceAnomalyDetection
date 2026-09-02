"""
Generic training loop shared by every BaseAnomalyModel implementation.
"""
from typing import Optional
import torch
from torch.utils.data import DataLoader


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 30,
    lr: float = 1e-3,
    device: Optional[str] = None,
    patience: int = 5,
) -> list:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    history = []
    best_loss, epochs_without_improvement = float("inf"), 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            clip = batch["clip"].to(device)
            optimizer.zero_grad()
            loss = model.compute_loss(clip)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * clip.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        history.append(epoch_loss)
        scheduler.step(epoch_loss)
        print(
            f"Epoch {epoch+1}/{num_epochs} - loss: {epoch_loss:.6f}"
            f" - lr: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if epoch_loss < best_loss - 1e-6:
            best_loss, epochs_without_improvement = epoch_loss, 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping: no improvement for {patience} epochs.")
                break
    return history
