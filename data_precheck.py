# data_precheck.py

import os
import sys
import json
import h5py
import numpy as np
import pandas as pd
from collections import defaultdict

# ---------------- CONFIG ---------------- #

EXPECTED_WINDOW = 60
EXPECTED_FEATURES = 263
MAX_ABS_VALUE = 10.0   # hard bound to prevent GRU overflow
REQUIRED_H5_KEYS = {"X", "y1", "y5", "dates"}

FILES = {
    "edges": "gnn_data/edges_static.parquet",
    "nodes": "gnn_data/node_features.parquet",
    "train": "gnn_data/gnn_sequences_train.h5",
    "val": "gnn_data/gnn_sequences_val.h5",
    "test": "gnn_data/gnn_sequences_test.h5",
}

# ---------------------------------------- #

errors = []
warnings = []
stats = {}

def fail(msg):
    errors.append(msg)

def warn(msg):
    warnings.append(msg)

def check_file_exists(path):
    if not os.path.exists(path):
        fail(f"Missing file: {path}")

# ---------- FILE EXISTENCE ---------- #

for name, path in FILES.items():
    check_file_exists(path)

if errors:
    raise RuntimeError(f"Precheck failed early:\n{errors}")

# ---------- EDGES CHECK ---------- #

edges = pd.read_parquet(FILES["edges"])

required_cols = {"symbol_i", "symbol_j", "weight"}
if not required_cols.issubset(edges.columns):
    fail(f"edges_static.parquet missing columns: {required_cols - set(edges.columns)}")

if edges.isna().any().any():
    fail("edges_static.parquet contains NaNs")

if not np.isfinite(edges["weight"]).all():
    fail("edges_static.parquet contains non-finite weights")

stats["edges_count"] = len(edges)

# ---------- NODE FEATURES CHECK ---------- #

nodes = pd.read_parquet(FILES["nodes"])

if nodes.isna().any().any():
    fail("node_features.parquet contains NaNs")

numeric_cols = nodes.select_dtypes(include=[np.number]).columns
if len(numeric_cols) != EXPECTED_FEATURES:
    fail(f"Expected {EXPECTED_FEATURES} node features, found {len(numeric_cols)}")

if not np.isfinite(nodes[numeric_cols].values).all():
    fail("node_features.parquet contains non-finite values")

max_node_val = np.abs(nodes[numeric_cols].values).max()
if max_node_val > MAX_ABS_VALUE:
    warn(f"Node features exceed safe range: max={max_node_val:.2e}")

node_symbols = set(nodes["symbol"].unique())
stats["node_symbols"] = len(node_symbols)

# ---------- HDF5 CHECK FUNCTION ---------- #

def check_h5(path, split_name):
    with h5py.File(path, "r") as h5:
        symbols = list(h5.keys())
        stats[f"{split_name}_symbols"] = len(symbols)

        for sym in symbols:
            grp = h5[sym]

            if not REQUIRED_H5_KEYS.issubset(grp.keys()):
                fail(f"{path}:{sym} missing keys {REQUIRED_H5_KEYS - set(grp.keys())}")

            X = grp["X"][:]
            y1 = grp["y1"][:]
            y5 = grp["y5"][:]

            # shape checks
            if X.ndim != 3 or X.shape[1] != EXPECTED_WINDOW or X.shape[2] != EXPECTED_FEATURES:
                fail(f"{path}:{sym} invalid X shape {X.shape}")

            # dtype checks
            if X.dtype != np.float32:
                warn(f"{path}:{sym} X dtype is {X.dtype}, expected float32")

            # finite checks
            if not np.isfinite(X).all():
                fail(f"{path}:{sym} X contains NaN/inf")

            if not np.isfinite(y1).all() or not np.isfinite(y5).all():
                fail(f"{path}:{sym} targets contain NaN/inf")

            # magnitude checks (ROOT CAUSE PREVENTION)
            max_val = np.abs(X).max()
            if max_val > MAX_ABS_VALUE:
                fail(
                    f"{path}:{sym} has unsafe magnitude "
                    f"(max |X| = {max_val:.2e})"
                )

            # window count consistency
            if not (len(y1) == len(y5) == X.shape[0]):
                fail(f"{path}:{sym} window/target length mismatch")

# ---------- RUN SPLIT CHECKS ---------- #

check_h5(FILES["train"], "train")
check_h5(FILES["val"], "val")
check_h5(FILES["test"], "test")

# ---------- SYMBOL CONSISTENCY ---------- #

def extract_symbols(h5_path):
    with h5py.File(h5_path, "r") as h5:
        return set(h5.keys())

train_syms = extract_symbols(FILES["train"])
val_syms = extract_symbols(FILES["val"])
test_syms = extract_symbols(FILES["test"])

if not (train_syms & val_syms & test_syms):
    warn("Some symbols do not appear in all splits (expected but verify intent)")

if not train_syms.issubset(node_symbols):
    fail("Train symbols missing from node_features")

# ---------- FINAL REPORT ---------- #

report = {
    "status": "FAIL" if errors else "PASS",
    "errors": errors,
    "warnings": warnings,
    "stats": stats,
}

print(json.dumps(report, indent=2))

if errors:
    raise RuntimeError("❌ Data precheck FAILED — fix data before training")

print("✅ Data precheck PASSED — safe to train")
