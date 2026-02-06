Perfect. Specs first, **then** code — and we’ll design this **explicitly for your machine** (Windows + RTX 3050 4 GB). No accidental OOMs, no Linux-only assumptions.

---

# 📜 MarketSentinel — **Training Loop Design (Frozen)**

This is the **only legal way** the model is trained.

```
DataLoader → Model → Loss → Optimizer → Checkpoint
```

No shortcuts. No clever tricks.

---

## 0️⃣ Hardware-Aware Constraints (Locked)

Your setup:

* **CPU**: Ryzen 5 7535HS
* **RAM**: 16 GB
* **GPU**: RTX 3050 Laptop, **4 GB VRAM**
* **OS**: Windows

### Consequences

* ❌ No massive batch sizes
* ❌ No multi-GPU logic
* ❌ No DDP / DeepSpeed
* ❌ No mixed precision *by default* (we’ll add it safely later)

✅ We design for:

* batch sizes **8–32**
* single-GPU determinism
* stable memory usage

---

## 1️⃣ Training Script Location (Frozen)

```
training/train.py
```

This script:

* owns the training loop
* owns the optimizer
* owns checkpointing
* does **not** define the model or dataset

---

## 2️⃣ Inputs to Training Script (Contract)

```python
train.py receives:
  - path to train HDF5
  - path to val HDF5
  - feature_schema.txt
```

Nothing else.

🚫 Training script must **not**:

* read node_features.parquet
* read edges_static.parquet
* infer feature counts
* touch test data

---

## 3️⃣ DataLoader Setup (Frozen)

```python
batch_size = 16  # default (safe for 4GB VRAM)
num_workers = 0  # Windows-safe
pin_memory = True (only if CUDA)
shuffle = False  # IMPORTANT: temporal order preserved
```

Why:

* Windows + PyTorch + workers > 0 = pain
* Sequence data does **not** need shuffle
* Determinism > randomness (for now)

---

## 4️⃣ Model Setup (Frozen)

```python
input_dim = len(feature_schema)
hidden_dim = 128
output_dim = 2
```

* Model instantiated **once**
* `.to(device)` exactly once
* `.train()` / `.eval()` toggled explicitly

---

## 5️⃣ Loss Function (Critical)

We use **two-head regression loss**.

### Contract

```python
loss = MSE(y_hat[:, 0], y[:, 0])
     + MSE(y_hat[:, 1], y[:, 1])
```

* No weighting (v1)
* No fancy losses
* Loss defined **outside** model

---

## 6️⃣ Optimizer (Frozen)

```python
optimizer = AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
)
```

Why:

* stable
* well-understood
* forgiving with unnormalized features

---

## 7️⃣ Training Loop Semantics

### One Epoch

```
for batch in train_loader:
  X, y
  y_hat = model(X)
  loss = compute_loss
  loss.backward()
  optimizer.step()
  optimizer.zero_grad()
```

### Validation

* no gradients
* no optimizer
* model.eval()

---

## 8️⃣ Early Stopping (Simple & Safe)

* Monitor **val loss**
* Patience = 5 epochs
* Save **best model only**

---

## 9️⃣ Checkpoint Contract

Saved artifact:

```
checkpoints/
└── model_best.pt
```

Contents:

```python
{
  "model_state_dict": ...,
  "input_dim": F,
  "hidden_dim": H,
  "epoch": int,
  "val_loss": float
}
```

🚫 No optimizer state (v1)
🚫 No training data saved

---

## 🔒 What Is Now Locked

* Training semantics
* Loss definition
* Optimizer choice
* Batch discipline
* Hardware-safe defaults

From here on:

* model changes do NOT affect training code
* data changes require full rebuild
* training bugs are isolated

---