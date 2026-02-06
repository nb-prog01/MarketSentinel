# 📜 MarketSentinel — Data & Training Contracts

This document defines **all generated artifacts**, their **intent**, **format**, and **validation rules** required for safe downstream usage in:

* DataLoader
* Model construction
* Training
* Backtesting

If any contract fails, **training must not start**.

---

## 0️⃣ Design Philosophy (Non-Negotiable)

> Models do not fail first.
> **Interfaces fail first.**

Every artifact here exists to:

* isolate responsibility
* make assumptions explicit
* enable ablation and reproducibility
* prevent silent leakage or drift

---

# 1️⃣ `feature_schema.txt`

### **Intent**

Defines the **single frozen feature interface** used everywhere downstream.

This file is the **feature ABI** (application binary interface).

---

### **Format**

Plain text, one feature per line, ordered.

```
feature_001
feature_002
...
feature_263
```

---

### **Hard Guarantees**

* Order is **fixed**
* Count is **fixed**
* Names are **canonical**
* Used by:

  * temporal HDF5
  * node features
  * DataLoader
  * model input layers

---

### **Validations**

* Must exist
* Must be non-empty
* Feature count > 0
* Feature order must match:

  * `node_features.parquet`
  * `X.shape[-1]` in HDF5

---

### **Downstream Assumptions**

* Model input dimension = `len(feature_schema)`
* Any mismatch → **hard failure**

---

# 2️⃣ `node_features.parquet`

### **Intent**

Defines **node-level state** at each `(symbol, date)`.

Used for:

* graph node initialization
* debugging / inspection
* static node embeddings (if needed)

---

### **Format**

Parquet table.

| column     | type               |
| ---------- | ------------------ |
| symbol     | string (canonical) |
| date       | datetime64[ns]     |
| *features* | float32            |

Total columns:

```
2 + len(feature_schema)
```

---

### **Hard Guarantees**

* Same feature order as `feature_schema.txt`
* dtype = `float32`
* All values finite
* Symbols are canonical (`AAPL`, `BTC_USD`, etc.)

---

### **Validations**

* Column order exactly:

  ```
  ["symbol", "date"] + feature_schema
  ```
* No NaNs / infs in features
* Feature count matches schema
* Symbol namespace matches HDF5 symbols

---

### **Downstream Assumptions**

* Node features can be joined by `(symbol, date)`
* No aggregation required
* No normalization required

---

# 3️⃣ `gnn_sequences_train.h5`

`gnn_sequences_val.h5`
`gnn_sequences_test.h5`

### **Intent**

Primary **model training input**: temporal sequences.

This is the **only** data the model trains on.

---

### **Format**

HDF5 file with datasets:

```
/X       float32 [N, 60, F]
/y       float32 [N, 2]
/symbol  bytes   [N]
/date    int64   [N]   # nanoseconds since epoch
```

Where:

* `F = len(feature_schema)`
* Targets:

  * `y[:, 0] = ret_fwd_1d`
  * `y[:, 1] = ret_fwd_5d`

---

### **Hard Guarantees**

* Chronological split (per symbol)
* No shuffling
* No leakage across splits
* All values finite
* dtype strictly enforced

---

### **Validations**

* `X.shape == (N, 60, F)`
* `y.shape == (N, 2)`
* `X.dtype == float32`
* `y.dtype == float32`
* `date.dtype == int64`
* `np.isfinite(X).all()`
* `np.isfinite(y).all()`
* `date` convertible back to datetime

---

### **Downstream Assumptions**

* DataLoader reads HDF5 lazily
* No preprocessing needed
* Targets aligned to window end
* Temporal encoder expects exactly 60 steps

---

# 4️⃣ `edges_static.parquet`

### **Intent**

Defines **graph structure** for GNN message passing.

Current version is a **baseline identity graph**.

---

### **Format**

Parquet table.

| column   | type    |
| -------- | ------- |
| symbol_i | string  |
| symbol_j | string  |
| weight   | float32 |

---

### **Current Semantics (v1)**

```
symbol_i == symbol_j
weight   == 1.0
```

This represents:

* self-loops only
* no cross-asset inductive bias

---

### **Hard Guarantees**

* Symbols exist in `node_features.parquet`
* Weights are finite
* Graph is static

---

### **Validations**

* Required columns exist
* `symbol_i` ⊆ node symbols
* `symbol_j` ⊆ node symbols
* `np.isfinite(weight).all()`

---

### **Downstream Assumptions**

* GNN code path is active
* Message passing degenerates to identity
* Future edge upgrades do **not** affect loaders/models

---

# 5️⃣ `data_precheck.py` (Compiler Gate)

### **Intent**

Single **authoritative validation gate**.

If this fails:

> **Training must not start.**

---

### **What it validates**

* Feature schema consistency
* Node features correctness
* Temporal HDF5 integrity
* Edge alignment
* dtype, shape, finiteness

---

### **Philosophy**

* Treats dataset like compiled binary
* Prevents undefined behavior
* Makes failures early and loud

---

# 6️⃣ DataLoader Contract

### **Input Assumptions**

* Reads **only** HDF5 + schema
* No feature inference
* No normalization
* No symbol munging

---

### **Output Contract**

Each batch yields:

```
X : float32 [B, 60, F]
y : float32 [B, 2]
node_ids / symbols (optional)
```

---

### **Hard Rules**

* Must respect split boundaries
* Must not reorder time within windows
* Must not drop features

---

# 7️⃣ Model Construction Contract

### **Input Layer**

```
input_dim = len(feature_schema)
sequence_len = 60
```

---

### **Graph Encoder**

* Accepts static edges
* Must handle self-loops correctly

---

### **Temporal Encoder**

* Assumes features are:

  * scaled
  * finite
  * bounded

---

### **Output Heads**

```
Head 1 → ret_fwd_1d
Head 2 → ret_fwd_5d
```

---

# 8️⃣ Training & Backtesting Contract

### **Training**

* Uses **train split only**
* Validation used only for:

  * early stopping
  * hyperparam tuning

---

### **Backtesting**

* Uses **test split only**
* No retraining
* No feature recomputation

---

# 9️⃣ Versioning Rule (Critical)

Any change to:

* feature schema
* NaN policy
* window size
* edge semantics

➡️ **requires a full dataset rebuild**
➡️ old models become invalid

---

## ✅ Final Status (Current Build)

| Artifact              | Status |
| --------------------- | ------ |
| feature_schema.txt    | ✅      |
| node_features.parquet | ✅      |
| gnn_sequences_*.h5    | ✅      |
| edges_static.parquet  | ✅      |
| precheck              | ✅      |

