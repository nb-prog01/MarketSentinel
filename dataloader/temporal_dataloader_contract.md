Good. We proceed **straight down the line**.

---

# 📜 MarketSentinel — **Temporal DataLoader Contract** (Frozen)

This is the **only legal interface** between:

```
Validated HDF5  →  Model
```

Once frozen, **all downstream code must obey this contract**.

---

## 1️⃣ Purpose (Non-Negotiable)

The DataLoader exists to:

* read **only validated HDF5 artifacts**
* expose **exactly what the model consumes**
* enforce shape, dtype, and ordering at runtime
* perform **zero feature engineering**
* perform **zero normalization**
* perform **zero symbol manipulation**

It is a **reader + enforcer**, nothing more.

---

## 2️⃣ File Location (Frozen)

```
dataloader/temporal_dataloader.py
```

No other loader is permitted.

---

## 3️⃣ Constructor Contract

```python
TemporalSequenceDataset(
    h5_path: Path | str,
    feature_schema_path: Path | str,
)
```

### Constructor Guarantees

On construction, it must:

* assert HDF5 file exists
* assert schema file exists
* load schema **once**
* open HDF5 lazily (no tensor loads)
* validate:

  * dataset keys
  * feature dimension == schema length
  * sequence length == 60
  * target dim == 2

❌ If any assertion fails → **raise RuntimeError**

---

## 4️⃣ Dataset Length

```python
__len__() -> int
```

Returns:

```
N = number of temporal windows
```

---

## 5️⃣ `__getitem__` Contract (Critical)

```python
__getitem__(idx: int) -> dict
```

### Output (Single Sample)

```python
{
  "X": Tensor[60, F]   float32,
  "y": Tensor[2]       float32,
  "symbol": str,
  "date": datetime64
}
```

### Hard Rules

* No reshaping
* No squeezing
* No casting
* No padding
* No reordering
* No normalization
* No augmentation

Indexing is **direct**, not shuffled.

---

## 6️⃣ Runtime Assertions (Mandatory)

Every `__getitem__` call must assert:

* `X.shape == (60, F)`
* `y.shape == (2,)`
* `X.dtype == torch.float32`
* `y.dtype == torch.float32`
* all values finite

These assertions stay **forever**.

---

## 7️⃣ Batching Semantics

Batching is delegated to `torch.utils.data.DataLoader`.

The dataset must be:

* order-stable
* deterministic
* reproducible

---

## 8️⃣ Split Discipline

Each split uses **one dataset instance**:

```
train → gnn_sequences_train.h5
val   → gnn_sequences_val.h5
test  → gnn_sequences_test.h5
```

No mixing. Ever.

---

## 9️⃣ Forbidden Behavior (Explicit)

🚫 Any of the following is a contract violation:

* reading node_features.parquet
* reading edges_static.parquet
* inferring feature count
* handling NaNs
* scaling features
* clipping values
* symbol remapping
* time reordering

Those belong **upstream** and are already frozen.

---

## 🔒 Contract Status

**Status:** LOCKED
**Change policy:** Breaking change → full dataset rebuild
