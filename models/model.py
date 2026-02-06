# ============================================================
# MarketSentinel — Full Model
# ============================================================

import torch
import torch.nn as nn

from models.temporal_encoder import TemporalEncoder
from models.graph_encoder import GraphEncoder
from models.fusion import FusionLayer


class MarketSentinelModel(nn.Module):
    """
    End-to-end MarketSentinel model.

    Input:
      X : [B, 60, F]

    Output:
      y_hat : [B, 2]
        - y_hat[:, 0] → ret_fwd_1d
        - y_hat[:, 1] → ret_fwd_5d
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 2,
    ):
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")

        if output_dim != 2:
            raise ValueError("output_dim must be exactly 2")

        self.temporal_encoder = TemporalEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
        )

        self.graph_encoder = GraphEncoder(
            hidden_dim=hidden_dim,
        )

        self.fusion = FusionLayer(
            hidden_dim=hidden_dim,
        )

        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        X : torch.Tensor
            Shape: [B, 60, F], dtype=float32

        Returns
        -------
        torch.Tensor
            Shape: [B, 2], dtype=float32
        """

        # ---- Runtime contract enforcement ----
        if X.ndim != 3:
            raise RuntimeError(f"Model expected 3D input, got {X.ndim}D")

        if X.dtype != torch.float32:
            raise RuntimeError(f"X dtype {X.dtype} != torch.float32")

        if not torch.isfinite(X).all():
            raise RuntimeError("Non-finite values detected in model input")

        # ---- Forward path ----
        H_temporal = self.temporal_encoder(X)     # [B, H]
        H_graph = self.graph_encoder(H_temporal)  # [B, H]
        H_fused = self.fusion(H_graph)             # [B, H]

        y_hat = self.head(H_fused)                 # [B, 2]

        # ---- Output contract ----
        if y_hat.shape[1] != 2:
            raise RuntimeError("Model output must have 2 dimensions")

        if not torch.isfinite(y_hat).all():
            raise RuntimeError("Non-finite values detected in model output")

        return y_hat
