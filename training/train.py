# ============================================================
# MarketSentinel — Training Script
# ============================================================

import sys
from pathlib import Path
from typing import Dict
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


# ---- Ensure project root on PYTHONPATH ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloader.temporal_dataloader import TemporalSequenceDataset
from models.model import MarketSentinelModel

print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


# ============================================================
# Configuration (FROZEN DEFAULTS)
# ============================================================

BATCH_SIZE = 16           # Safe for RTX 3050 (4 GB)
NUM_WORKERS = 0           # Windows-safe
HIDDEN_DIM = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 2
EARLY_STOPPING_PATIENCE = 5

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ============================================================
# CANARY INTEGRITY MODE (RUNTIME ONLY)
# ============================================================

CANARY_RUN = True          # Set False for full training
CANARY_FRACTION = 0.25     # 25–30% recommended
CANARY_MAX_BATCHES = None # Optional hard cap (e.g. 20)


# ============================================================
# Collate
# ============================================================

def temporal_collate_fn(batch):
    """
    Custom collate function for TemporalSequenceDataset.

    - Stacks X and y into tensors
    - Keeps symbol and date as Python lists
    """
    return {
        "X": torch.stack([b["X"] for b in batch], dim=0),
        "y": torch.stack([b["y"] for b in batch], dim=0),
        "symbol": [b["symbol"] for b in batch],
        "date": [b["date"] for b in batch],
    }


# ============================================================
# Utilities
# ============================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Two-head regression loss (MSE).
    """
    if y_hat.shape != y.shape:
        raise RuntimeError("Prediction/target shape mismatch")

    loss_fn = nn.MSELoss()
    return loss_fn(y_hat[:, 0], y[:, 0]) + loss_fn(y_hat[:, 1], y[:, 1])


def save_checkpoint(
    path: Path,
    model: nn.Module,
    epoch: int,
    val_loss: float,
    input_dim: int,
    hidden_dim: int,
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
        },
        path,
    )


# ============================================================
# Training Loop
# ============================================================

def train(
    train_h5: Path,
    val_h5: Path,
    feature_schema_path: Path,
):
    device = get_device()
    print(f"🖥 Using device: {device}")

    # ---- Load feature schema ----
    with open(feature_schema_path, "r") as f:
        feature_schema = [line.rstrip("\n") for line in f.readlines()]
    input_dim = len(feature_schema)

    if input_dim <= 0:
        raise RuntimeError("Feature schema is empty")

    # ---- Datasets ----
    train_dataset = TemporalSequenceDataset(
        h5_path=train_h5,
        feature_schema_path=feature_schema_path,
    )

    # --------------------------------------------------------
    # Canary split (TRAIN ONLY, runtime, no file changes)
    # --------------------------------------------------------
    if CANARY_RUN:
        full_len = len(train_dataset)
        canary_len = int(full_len * CANARY_FRACTION)

        # Preserve chronological order (no randomness)
        canary_indices = list(range(canary_len))
        train_dataset = Subset(train_dataset, canary_indices)

        print(
            f"🧪 CANARY RUN ENABLED — "
            f"Using {canary_len}/{full_len} train samples"
        )


    val_dataset = TemporalSequenceDataset(
        h5_path=val_h5,
        feature_schema_path=feature_schema_path,
    )
    # --------------------------------------------------------
    # Canary split (VAL ONLY, runtime, no file changes)
    # --------------------------------------------------------
    if CANARY_RUN:
        full_val_len = len(val_dataset)
        canary_val_len = int(full_val_len * CANARY_FRACTION)

        canary_val_indices = list(range(canary_val_len))
        val_dataset = Subset(val_dataset, canary_val_indices)

        print(
            f"🧪 CANARY RUN (VAL) — "
            f"Using {canary_val_len}/{full_val_len} samples"
        )


    # ---- DataLoaders ----
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        collate_fn=temporal_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        collate_fn=temporal_collate_fn,
    )


    # ---- Model ----
    model = MarketSentinelModel(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    # ========================================================
    # Epoch Loop
    # ========================================================
    for epoch in range(1, MAX_EPOCHS + 1):
        print(f"\n🚀 Epoch {epoch}/{MAX_EPOCHS}")

        # --------------------
        # Training
        # --------------------
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch} [TRAIN]",
            leave=False,
        )

        for batch in progress:
            X = batch["X"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            optimizer.zero_grad()
            y_hat = model(X)
            loss = compute_loss(y_hat, y)

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

            # ---- Live loss update ----
            avg_loss = train_loss_sum / train_batches
            progress.set_postfix(loss=f"{avg_loss:.6f}")


        train_loss = train_loss_sum / max(train_batches, 1)
        print(f"📉 Train loss: {train_loss:.6f}")

        # --------------------
        # Validation
        # --------------------
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                X = batch["X"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)

                y_hat = model(X)
                loss = compute_loss(y_hat, y)

                val_loss_sum += loss.item()
                val_batches += 1

        val_loss = val_loss_sum / max(val_batches, 1)
        print(f"📊 Val loss:   {val_loss:.6f}")

        # --------------------
        # Early Stopping
        # --------------------
        if val_loss < best_val_loss:
            print("💾 New best model — saving checkpoint")
            best_val_loss = val_loss
            epochs_without_improvement = 0

            save_checkpoint(
                path=CHECKPOINT_DIR / "model_best.pt",
                model=model,
                epoch=epoch,
                val_loss=val_loss,
                input_dim=input_dim,
                hidden_dim=HIDDEN_DIM,
            )

            # ------------------------------------------------
            # Canary integrity: checkpoint reload verification
            # ------------------------------------------------
            if CANARY_RUN:
                ckpt = torch.load(
                    CHECKPOINT_DIR / "model_best.pt",
                    map_location="cpu",
                )

                assert "model_state_dict" in ckpt
                assert ckpt["input_dim"] == input_dim
                assert ckpt["hidden_dim"] == HIDDEN_DIM

                print("✅ Canary checkpoint reload verified")

        else:
            epochs_without_improvement += 1
            print(
                f"⏳ No improvement "
                f"({epochs_without_improvement}/{EARLY_STOPPING_PATIENCE})"
            )

            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print("🛑 Early stopping triggered")
                break

    # ---- Cleanup ----
    # train_dataset.close()
    # val_dataset.close()
        # ---- Cleanup ----
    if hasattr(train_dataset, "dataset"):
        train_dataset.dataset.close()
    else:
        train_dataset.close()

    if hasattr(val_dataset, "dataset"):
        val_dataset.dataset.close()
    else:
        val_dataset.close()


    print("\n🏁 TRAINING COMPLETE")
    print(f"✅ Best val loss: {best_val_loss:.6f}")
    print(f"📦 Best model saved to: {CHECKPOINT_DIR / 'model_best.pt'}")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    DATASET_DIR = PROJECT_ROOT / "datasets"

    train_h5 = DATASET_DIR / "gnn_sequences_train.h5"
    val_h5 = DATASET_DIR / "gnn_sequences_val.h5"
    feature_schema_path = DATASET_DIR / "feature_schema.txt"

    train(
        train_h5=train_h5,
        val_h5=val_h5,
        feature_schema_path=feature_schema_path,
    )
