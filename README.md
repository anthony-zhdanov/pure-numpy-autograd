# pure-numpy-autograd

A modular neural network training framework and automatic differentiation engine built from scratch in pure Python and NumPy — no ML frameworks used.

> **Built by hand.** Every line of code, every mathematical derivation, and every design decision in this project is original. AI assistance is used minimally and only for explanation, never for generating implementation code.

---

## What This Is

Most engineers who use PyTorch have never seen what sits beneath it. This project is a ground-up reconstruction of the core machinery that makes deep learning work: a dynamic computation graph, reverse-mode automatic differentiation, a full suite of optimizers derived from their original papers, and a live training dashboard that exposes the internal health of a training run.

The goal is not to produce a faster or more capable framework than PyTorch — PyTorch exists. The goal is complete, from-scratch understanding of every component that makes neural network training possible.

---

## Architecture

```
autograd_engine/
├── tensor.py          # Tensor class — computation graph node + autograd engine
├── ops.py             # Standalone operations
├── loss.py            # MSE, CrossEntropy, BCE
├── data.py            # Dataset base class + DataLoader
├── nn/
│   ├── module.py      # Module base class — parameters(), zero_grad(), __call__
│   ├── layers.py      # Dense, Dropout, BatchNorm
│   └── activations.py # ReLU, Sigmoid, Tanh, GELU
├── optim/
│   ├── sgd.py         # SGD + Momentum
│   ├── adam.py        # Adam with bias correction
│   ├── adamw.py       # AdamW — decoupled weight decay
│   └── lion.py        # Lion — sign-based optimizer (research extension)
└── backend/
    ├── numpy_backend.py  # Default: wraps NumPy
    └── cupy_backend.py   # GPU: drop-in swap via get_backend()
```

---

## Core Design: The Autograd Engine

Every arithmetic operation on a `Tensor` returns a new `Tensor` that records:
- `data` — the NumPy array result of the operation
- `grad` — accumulated gradient, initialized to zero
- `_prev` — the set of parent `Tensor` nodes that produced this one
- `_backward` — a closure that knows how to push gradients back to `_prev`

Calling `.backward()` on a loss `Tensor` topologically sorts the computation graph and calls each node's `_backward` in reverse order, accumulating gradients at every leaf. This is reverse-mode automatic differentiation — the same algorithm that underlies PyTorch, JAX, and every modern ML framework.

Gradient accumulation uses `+=` throughout. A `Tensor` can participate in multiple downstream operations; each contributes independently to its gradient, and the total derivative is their sum.

---

## Installation

```bash
git clone https://github.com/anthony-zhdanov/pure-numpy-autograd
cd pure-numpy-autograd
pip install -e .
```

With optional dependencies:

```bash
pip install -e ".[dev]"        # pytest + torch (gradient validation)
pip install -e ".[dashboard]"  # streamlit + plotly
pip install -e ".[gpu]"        # cupy (GPU backend)
```

---

## Current Status

This project is under active development. The table below reflects the current state of each component.

| Component | Status |
|-----------|--------|
| `Tensor` forward pass — all arithmetic, `exp`, `log`, `relu`, `sum` | Complete |
| `_backward` closures — all operations | Complete |
| `backward()` — topological sort + gradient accumulation | In progress |
| Gradient tests against PyTorch | Pending |
| Computation graph visualizer | Pending |
| `nn.Module` base class | Pending |
| Dense, Dropout, BatchNorm layers | Pending |
| Activation functions (ReLU, Sigmoid, Tanh, GELU) | Pending |
| Loss functions (MSE, BCE, CrossEntropy) | Pending |
| DataLoader | Pending |
| SGD + Momentum | Pending |
| Adam, AdamW | Pending |
| Lion optimizer | Pending |
| LR scheduling + gradient clipping | Pending |
| Full training run on MNIST vs PyTorch | Pending |
| Live Streamlit training dashboard | Pending |
| PyTorch benchmark + profiling analysis | Pending |
| CuPy GPU backend | Pending |
| MAML — higher-order gradients | Pending |

---

## Roadmap

### Month 1 — The Foundation

**Week 1: Autograd Engine**
Dynamic computation graph with forward and backward passes across arbitrary graphs. Every operation stores a `_backward` closure that propagates gradients using the chain rule. Validated against PyTorch to 6+ decimal places.

**Week 2: Layers and the Module System**
A `Module` base class mirroring PyTorch's API: `parameters()`, `zero_grad()`, `__call__` → `forward()`. Implements Dense (Linear), Dropout, BatchNorm, and activation functions. The BatchNorm backward pass is derived fully by hand — one of the more mathematically involved derivations in the project.

**Week 3: Optimizers**
Five optimizers implemented from their original papers: SGD, SGD with Momentum, RMSProp, Adam, and AdamW. Each is derived mathematically before implementation. Also includes learning rate scheduling and gradient clipping.

**Week 4: Loss Functions, DataLoader, First Training Run**
MSE, Binary Cross-Entropy, and Categorical Cross-Entropy with numerically stable implementations. A custom `DataLoader` with batching and shuffling. A full training run on MNIST with loss curves matched against PyTorch to 3 decimal places.

### Month 2 — Depth, Extensions, Polish

**Week 5: Live Training Dashboard**
A Streamlit dashboard with five real-time panels: loss curves, gradient norms per layer, weight distribution histograms, learning rate schedule, and activation statistics. Runs training in a background thread and updates live.

**Week 6: PyTorch Benchmark**
Systematic benchmark measuring numerical correctness, training speed, memory usage, convergence behavior, and optimizer sensitivity. Includes profiling with `cProfile` and a written engineering analysis of where the gap with PyTorch comes from and why.

**Week 7: Lion Optimizer + GPU Backend**
Reproduction of the Lion optimizer from Chen et al. (Google Brain, 2023) — a sign-based update rule discovered by an evolutionary algorithm. Validated against the paper's results. CuPy backend swaps NumPy for GPU acceleration with no changes to the core `Tensor` logic.

**Week 8: MAML + Documentation + Deploy**
Model-Agnostic Meta-Learning (Finn et al., 2017) as a Tier 4 extension. Requires the autograd engine to track gradients-of-gradients via `create_graph=True` — demonstrating higher-order differentiation support. Deployed as a live Streamlit demo.

---

## Mathematical Foundations

Full derivations for every component — chain rule, topological sort, all optimizer update rules, BatchNorm backward, and MAML outer loop — are documented in `docs/math.md`.

Papers implemented directly from source:
- *Learning representations by back-propagating errors* — Rumelhart et al., 1986
- *Batch Normalization* — Ioffe & Szegedy, 2015
- *Adam: A Method for Stochastic Optimization* — Kingma & Ba, 2014
- *Decoupled Weight Decay Regularization* — Loshchilov & Hutter, 2017
- *Symbolic Discovery of Optimization Algorithms (Lion)* — Chen et al., 2023
- *Model-Agnostic Meta-Learning* — Finn et al., 2017

---

## Running Tests

```bash
pytest tests/
```

---

## Running the Dashboard

```bash
streamlit run dashboard/app.py
```
