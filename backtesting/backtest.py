# ============================================================
# MarketSentinel — Backtesting with IC Analysis
# ============================================================
# Read-only inference + evaluation
# ============================================================

import sys
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from training.train import temporal_collate_fn



# ---- Ensure project root on PYTHONPATH ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloader.temporal_dataloader import TemporalSequenceDataset
from models.model import MarketSentinelModel


# ============================================================
# Configuration (FROZEN)
# ============================================================

BATCH_SIZE = 32          # Inference-only, safe for 4GB VRAM
NUM_WORKERS = 0          # Windows-safe
SEED = 42

TARGET_IC = 0.05         # Industry-standard "good" IC
MIN_SYMBOLS_FOR_IC = 5   # Avoid fake stability

BACKTEST_DIR = PROJECT_ROOT / "backtests"
BACKTEST_DIR.mkdir(exist_ok=True)

# ============================================================
# CANARY BACKTEST MODE (RUNTIME ONLY)
# ============================================================

CANARY_BACKTEST = True      # Flip to False for full backtest
CANARY_FRACTION = 0.05      # 5% is ideal for smoke tests


# ============================================================
# Utilities
# ============================================================

def set_determinism(seed: int = SEED):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return np.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return np.nan
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return pearson_corr(rx, ry)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = float(np.mean((y_pred - y_true) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    corr = pearson_corr(y_pred, y_true)
    return {"mse": mse, "mae": mae, "corr": corr}


def compute_ic_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-sectional Spearman IC per date for 1d and 5d horizons.
    """
    rows = []
    for date, g in df.groupby("date"):
        n = len(g)
        if n < MIN_SYMBOLS_FOR_IC:
            rows.append(
                {
                    "date": date,
                    "ic_1d": np.nan,
                    "ic_5d": np.nan,
                    "n_symbols": n,
                }
            )
            continue

        ic_1d = spearman_corr(g["y_pred_1d"].to_numpy(), g["y_true_1d"].to_numpy())
        ic_5d = spearman_corr(g["y_pred_5d"].to_numpy(), g["y_true_5d"].to_numpy())

        rows.append(
            {
                "date": date,
                "ic_1d": ic_1d,
                "ic_5d": ic_5d,
                "n_symbols": n,
            }
        )

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def summarize_ic(ic_ts: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Aggregate IC stats and compute IC-based signal confidence.
    """
    out: Dict[str, Dict[str, float]] = {}

    for horizon in ["1d", "5d"]:
        col = f"ic_{horizon}"
        vals = ic_ts[col].dropna().to_numpy()

        if vals.size == 0:
            mean_ic = std_ic = icir = pos_ratio = conf = 0.0
        else:
            mean_ic = float(np.mean(vals))
            std_ic = float(np.std(vals))
            icir = float(mean_ic / std_ic) if std_ic > 0 else 0.0
            pos_ratio = float(np.mean(vals > 0))
            conf = float(np.clip(mean_ic / TARGET_IC, 0.0, 1.0))

        out[horizon] = {
            "mean_ic": mean_ic,
            "std_ic": std_ic,
            "icir": icir,
            "positive_ic_ratio": pos_ratio,
            "signal_confidence": conf,
        }

    return out


# ============================================================
# Backtesting
# ============================================================

def backtest(
    test_h5: Path,
    feature_schema_path: Path,
    checkpoint_path: Path,
):
    set_determinism()

    device = get_device()
    print(f"🖥 Using device: {device}")

    # ---- Load schema ----
    with open(feature_schema_path, "r") as f:
        feature_schema = [line.rstrip("\n") for line in f.readlines()]
    input_dim = len(feature_schema)
    if input_dim <= 0:
        raise RuntimeError("Feature schema is empty")

    # ---- Load checkpoint ----
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    hidden_dim = int(ckpt["hidden_dim"])

    model = MarketSentinelModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ---- Dataset & Loader ----
    dataset = TemporalSequenceDataset(
        h5_path=test_h5,
        feature_schema_path=feature_schema_path,
    )
    
    # --------------------------------------------------------
    # Canary split (TEST ONLY, runtime, deterministic)
    # --------------------------------------------------------
    if CANARY_BACKTEST:
        full_len = len(dataset)
        canary_len = max(1, int(full_len * CANARY_FRACTION))

        canary_indices = list(range(canary_len))
        dataset = Subset(dataset, canary_indices)

        print(
            f"🧪 CANARY BACKTEST ENABLED — "
            f"Using {canary_len}/{full_len} test samples"
        )


    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        collate_fn=temporal_collate_fn,
    )

    # ---- Inference ----
    records = []

    with torch.no_grad():
        for batch in loader:
            X = batch["X"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            y_hat = model(X)

            for i in range(X.size(0)):
                records.append(
                    {
                        "symbol": batch["symbol"][i],
                        "date": batch["date"][i],
                        "y_true_1d": float(y[i, 0].item()),
                        "y_true_5d": float(y[i, 1].item()),
                        "y_pred_1d": float(y_hat[i, 0].item()),
                        "y_pred_5d": float(y_hat[i, 1].item()),
                    }
                )

    # dataset.close()
    if hasattr(dataset, "dataset"):
        dataset.dataset.close()
    else:
        dataset.close()


    preds = pd.DataFrame(records).sort_values("date").reset_index(drop=True)

    # ---- Save predictions ----
    preds_path = BACKTEST_DIR / "predictions.parquet"
    preds.to_parquet(preds_path, index=False)
    print(f"📦 Saved predictions → {preds_path}")

    # ---- Regression metrics ----
    m1 = regression_metrics(preds["y_true_1d"].to_numpy(), preds["y_pred_1d"].to_numpy())
    m5 = regression_metrics(preds["y_true_5d"].to_numpy(), preds["y_pred_5d"].to_numpy())

    metrics = {
        "1d": m1,
        "5d": m5,
    }

    metrics_path = BACKTEST_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"📊 Saved metrics → {metrics_path}")

    # ---- IC analysis ----
    ic_ts = compute_ic_timeseries(preds)
    ic_ts_path = BACKTEST_DIR / "ic_timeseries.parquet"
    ic_ts.to_parquet(ic_ts_path, index=False)
    print(f"📈 Saved IC timeseries → {ic_ts_path}")

    ic_summary = summarize_ic(ic_ts)
    ic_conf_path = BACKTEST_DIR / "signal_confidence.json"
    with open(ic_conf_path, "w") as f:
        json.dump(ic_summary, f, indent=2)
    print(f"🧭 Saved signal confidence → {ic_conf_path}")

    print("\n🏁 BACKTEST COMPLETE")
    print(json.dumps(ic_summary, indent=2))


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    DATASET_DIR = PROJECT_ROOT / "datasets"
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

    backtest(
        test_h5=DATASET_DIR / "gnn_sequences_test.h5",
        feature_schema_path=DATASET_DIR / "feature_schema.txt",
        checkpoint_path=CHECKPOINT_DIR / "model_best.pt",
    )
