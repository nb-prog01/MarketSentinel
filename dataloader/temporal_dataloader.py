# ============================================================
# MarketSentinel — TemporalSequenceDataset
# ============================================================
# Contract-enforced DataLoader for temporal GNN sequences
# ============================================================

from pathlib import Path
from typing import Union, Dict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


class TemporalSequenceDataset(Dataset):
    """
    Contract-enforced Dataset for MarketSentinel temporal sequences.

    Reads ONLY:
      - gnn_sequences_*.h5
      - feature_schema.txt

    Performs:
      - shape enforcement
      - dtype enforcement
      - finiteness checks

    Performs NO:
      - normalization
      - feature engineering
      - symbol manipulation
      - shuffling
    """

    SEQ_LEN = 60
    TARGET_DIM = 2

    def __init__(
        self,
        h5_path: Union[str, Path],
        feature_schema_path: Union[str, Path],
    ):
        # ---- Resolve paths ----
        self.h5_path = Path(h5_path)
        self.schema_path = Path(feature_schema_path)

        if not self.h5_path.exists():
            raise RuntimeError(f"HDF5 file not found: {self.h5_path}")

        if not self.schema_path.exists():
            raise RuntimeError(f"Feature schema not found: {self.schema_path}")

        # ---- Load feature schema ----
        with open(self.schema_path, "r") as f:
            self.feature_schema = [line.rstrip("\n") for line in f.readlines()]

        if len(self.feature_schema) == 0:
            raise RuntimeError("Feature schema is empty")

        self.feature_dim = len(self.feature_schema)

        # ---- Open HDF5 lazily ----
        self.h5 = h5py.File(self.h5_path, "r")

        # ---- Required datasets ----
        for key in ["X", "y", "symbol", "date"]:
            if key not in self.h5:
                raise RuntimeError(f"Missing dataset '{key}' in {self.h5_path}")

        self.X = self.h5["X"]
        self.y = self.h5["y"]
        self.symbol = self.h5["symbol"]
        self.date = self.h5["date"]

        # ---- Static contract checks (once) ----
        if self.X.ndim != 3:
            raise RuntimeError("X must be 3D: [N, 60, F]")

        N, T, F = self.X.shape

        if T != self.SEQ_LEN:
            raise RuntimeError(f"Sequence length {T} != {self.SEQ_LEN}")

        if F != self.feature_dim:
            raise RuntimeError(
                f"Feature dim {F} != schema length {self.feature_dim}"
            )

        if self.y.shape != (N, self.TARGET_DIM):
            raise RuntimeError(
                f"y shape {self.y.shape} != (N, {self.TARGET_DIM})"
            )

        if self.X.dtype != np.float32:
            raise RuntimeError(f"X dtype {self.X.dtype} != float32")

        if self.y.dtype != np.float32:
            raise RuntimeError(f"y dtype {self.y.dtype} != float32")

        if self.date.dtype != np.int64:
            raise RuntimeError(f"date dtype {self.date.dtype} != int64")

        if self.symbol.shape != (N,):
            raise RuntimeError("symbol shape mismatch")

        self.N = N

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0 or idx >= self.N:
            raise IndexError(f"Index {idx} out of bounds for dataset of size {self.N}")

        # ---- Read single sample (lazy) ----
        X_np = self.X[idx]          # (60, F)
        y_np = self.y[idx]          # (2,)
        sym = self.symbol[idx]
        date_ns = self.date[idx]

        # ---- Runtime assertions (MANDATORY) ----
        if X_np.shape != (self.SEQ_LEN, self.feature_dim):
            raise RuntimeError("X shape violation at runtime")

        if y_np.shape != (self.TARGET_DIM,):
            raise RuntimeError("y shape violation at runtime")

        if not np.isfinite(X_np).all():
            raise RuntimeError("Non-finite values detected in X")

        if not np.isfinite(y_np).all():
            raise RuntimeError("Non-finite values detected in y")

        # ---- Convert to torch tensors ----
        X = torch.from_numpy(X_np).float()
        y = torch.from_numpy(y_np).float()

        # ---- Symbol decoding ----
        if isinstance(sym, bytes):
            sym = sym.decode("utf-8")

        # ---- Date conversion ----
        date = pd.to_datetime(int(date_ns), unit="ns")

        return {
            "X": X,            # Tensor [60, F]
            "y": y,            # Tensor [2]
            "symbol": sym,     # str
            "date": date,      # datetime64
        }

    def close(self):
        """Explicitly close HDF5 file."""
        if self.h5 is not None:
            self.h5.close()
            self.h5 = None
