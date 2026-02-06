# ============================================================
# MarketSentinel — Fusion Layer (v1)
# ============================================================

import torch
import torch.nn as nn


class FusionLayer(nn.Module):
    """
    Fusion of temporal + graph representations.

    v1 behavior:
      - Identity mapping
      - Explicit fusion boundary
      - Contract enforcement only
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")

        self.hidden_dim = hidden_dim

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        H : torch.Tensor
            Shape: [B, H], dtype=float32

        Returns
        -------
        torch.Tensor
            Shape: [B, H], dtype=float32
        """

        # ---- Runtime contract enforcement ----
        if H.ndim != 2:
            raise RuntimeError(f"FusionLayer expected 2D input, got {H.ndim}D")

        B, Hd = H.shape

        if Hd != self.hidden_dim:
            raise RuntimeError(f"Hidden dim {Hd} != expected {self.hidden_dim}")

        if H.dtype != torch.float32:
            raise RuntimeError(f"H dtype {H.dtype} != torch.float32")

        if not torch.isfinite(H).all():
            raise RuntimeError("Non-finite values detected in FusionLayer input")

        # ---- Identity fusion ----
        return H
