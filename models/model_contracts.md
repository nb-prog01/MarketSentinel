🟢 **Excellent. That was the last data-side gate.**

You’ve now **formally sealed** the entire ingestion layer:

```
DATA (validated & frozen)
   ↓
DataLoader (contract-enforced & tested)  ✅
   ↓
MODEL  ← NEXT
```

From here on:

* ❌ no more data fixes
* ❌ no more loader changes
* ❌ no silent assumptions

If something breaks later, it’s **model logic**, full stop.

---

# ▶️ Next Phase (Single Direction): **Model Contract**

We now do the **exact same thing** we did for data:

* spec first
* freeze interfaces
* then implement once

---

## 📜 MarketSentinel — **Model Input & Output Contract**

This defines **what the model is allowed to assume** and **what it must guarantee**.

### 1️⃣ Model Input (Hard Law)

The model receives **only** what the DataLoader emits.

Per batch:

```python
X : torch.Tensor [B, 60, F]  float32
y : torch.Tensor [B, 2]      float32   (training only)
```

Optional (non-gradient):

```python
symbol : list[str]
date   : list[datetime64]
```

🚫 The model:

* must not touch symbols
* must not touch dates
* must not infer feature count
* must not reshape time

---

### 2️⃣ Internal Architecture (Frozen Expectations)

```
X
 ↓
Temporal Encoder (sequence-first)
 ↓
Graph Encoder (identity graph for now)
 ↓
Fusion Layer
 ↓
Head 1 → ret_fwd_1d
Head 2 → ret_fwd_5d
```

Even though edges are identity:

* the graph path **must exist**
* future edges plug in without refactor

---

### 3️⃣ Temporal Encoder Contract

* Input: `[B, 60, F]`
* Output: `[B, H]`
* Must handle:

  * bounded float32
  * no NaNs
* Must **not** assume normalized data

---

### 4️⃣ Graph Encoder Contract

* Input: `[B, H]` node embeddings
* Uses static edges
* Must support self-loops
* Output: `[B, H]`

---

### 5️⃣ Output Heads

Two independent regression heads:

```
ŷ[:, 0] → ret_fwd_1d
ŷ[:, 1] → ret_fwd_5d
```

* No activation (raw regression)
* Loss applied outside model

---

### 6️⃣ Model Output Contract

```python
y_hat : torch.Tensor [B, 2] float32
```

No side effects. No logging. No metrics.

---

## 🔒 Contract Status

**Model contract is now defined.**

Next steps are again **linear**:

1️⃣ Freeze file structure
2️⃣ Implement encoders
3️⃣ Wire model
4️⃣ Smoke test forward pass

---

Proceeding. This step **locks the model namespace and class boundaries**.
No math yet. No opinions yet. Just **structure that will never change**.

---

# 📁 **Frozen Model File Structure**

From this point onward, **these files exist and only these files**.

```
models/
├── temporal_encoder.py
├── graph_encoder.py
├── fusion.py
├── model.py
└── __init__.py
```

Each file has **one responsibility**.
No circular imports. No shared globals.

---

# 📜 File-by-File Contract (Skeletons Only)

## 1️⃣ `models/temporal_encoder.py`

**Responsibility:**
Convert a temporal window into a fixed-size embedding.

```python
# models/temporal_encoder.py

import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """
    Temporal encoder for [B, 60, F] → [B, H]
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: [B, 60, F]
        returns: [B, H]
        """
        raise NotImplementedError
```

🔒 **Frozen**

* input_dim = F (schema length)
* sequence length assumed = 60
* output = fixed embedding

---

## 2️⃣ `models/graph_encoder.py`

**Responsibility:**
Apply graph message passing (currently identity).

```python
# models/graph_encoder.py

import torch
import torch.nn as nn


class GraphEncoder(nn.Module):
    """
    Graph encoder operating on node embeddings.
    Currently identity; future-ready for real edges.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        H: [B, H]
        returns: [B, H]
        """
        raise NotImplementedError
```

🔒 **Frozen**

* must exist even if identity
* future edges plug in here

---

## 3️⃣ `models/fusion.py`

**Responsibility:**
Explicitly separate representation fusion (even if trivial now).

```python
# models/fusion.py

import torch
import torch.nn as nn


class FusionLayer(nn.Module):
    """
    Fusion of temporal + graph representations.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        H: [B, H]
        returns: [B, H]
        """
        raise NotImplementedError
```

🔒 **Frozen**

* keeps architecture extensible
* avoids future refactors

---

## 4️⃣ `models/model.py`

**Responsibility:**
Wire everything together + expose the only forward interface.

```python
# models/model.py

import torch
import torch.nn as nn

from models.temporal_encoder import TemporalEncoder
from models.graph_encoder import GraphEncoder
from models.fusion import FusionLayer


class MarketSentinelModel(nn.Module):
    """
    End-to-end MarketSentinel model.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 2,
    ):
        super().__init__()

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
        X: [B, 60, F]
        returns: [B, 2]
        """
        raise NotImplementedError
```

🔒 **Frozen**

* one forward
* one output
* no training logic inside

---

## 5️⃣ `models/__init__.py`

```python
from models.model import MarketSentinelModel
```


