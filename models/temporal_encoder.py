# ============================================================
# MarketSentinel — Temporal Encoder
# ============================================================

import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """
    Temporal encoder for [B, 60, F] → [B, H]

    Uses a GRU and returns the final hidden state.
    """

    SEQ_LEN = 60

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        X : torch.Tensor
            Shape: [B, 60, F], dtype=float32

        Returns
        -------
        torch.Tensor
            Shape: [B, H], dtype=float32
        """

        # ---- Runtime contract enforcement ----
        if X.ndim != 3:
            raise RuntimeError(f"TemporalEncoder expected 3D input, got {X.ndim}D")

        B, T, F = X.shape

        if T != self.SEQ_LEN:
            raise RuntimeError(f"Sequence length {T} != {self.SEQ_LEN}")

        if F != self.input_dim:
            raise RuntimeError(f"Feature dim {F} != expected {self.input_dim}")

        if X.dtype != torch.float32:
            raise RuntimeError(f"X dtype {X.dtype} != torch.float32")

        if not torch.isfinite(X).all():
            raise RuntimeError("Non-finite values detected in TemporalEncoder input")

        # ---- GRU forward ----
        # output: [B, 60, H]
        # h_n:    [1, B, H]
        _, h_n = self.gru(X)

        # Take final layer hidden state
        H = h_n[-1]  # [B, H]

        # ---- Output contract ----
        if H.shape != (B, self.hidden_dim):
            raise RuntimeError("TemporalEncoder output shape violation")

        return H
