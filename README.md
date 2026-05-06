# IC-POT

This repository contains a small reference implementation of
intent-controlled partial optimal transport (IC-POT), the formulation introduced
in the accompanying anonymous NeurIPS submission. The goal of the package is to
make the mathematical object in the paper easy to reproduce and inspect, not to
provide a large production framework.

IC-POT solves the finite slack problem

```text
min_{P,u,v >= 0} <C,P> + <c_source,u> + <c_target,v>
subject to       P 1 + u = mu,
                 P^T 1 + v = nu.
```

The transport cost `C` defines the geometry of matching. The pointwise costs
`c_source` and `c_target` define how expensive it is to leave mass unmatched on
each side. The two unmatched-cost vectors are the user-facing way to encode the
structured reject policy studied in the paper.

## Install

```bash
pip install -e .
```

The package depends only on NumPy and SciPy for the core solver.

## Minimal 1D example

```python
import numpy as np
from icpot import solve_icpot

x = np.linspace(0.0, 1.0, 50)[:, None]
y = np.linspace(0.0, 1.0, 50)[:, None]
mu = np.exp(-0.5 * ((x[:, 0] - 0.35) / 0.07) ** 2)
nu = np.exp(-0.5 * ((y[:, 0] - 0.38) / 0.07) ** 2)
nu += 0.5 * np.exp(-0.5 * ((y[:, 0] - 0.75) / 0.05) ** 2)

c_source = 0.2
c_target = 0.05 + 0.5 * np.exp(-0.5 * ((y[:, 0] - 0.38) / 0.15) ** 2)

sol = solve_icpot(
    mu,
    nu,
    source_points=x,
    target_points=y,
    c_source=c_source,
    c_target=c_target,
    solver="sparse",
)

print(sol.objective.total)
print(sol.dense_plan())
print(sol.source_unmatched, sol.target_unmatched)
```

The same example is available as:

```bash
PYTHONPATH=src python examples/basic_1d.py
```

## Main API

```python
solve_icpot(
    mu,
    nu,
    C=None,
    source_points=None,
    target_points=None,
    source_axes=None,
    target_axes=None,
    source_spacing=None,
    target_spacing=None,
    metric="sqeuclidean",
    periodic=None,
    axis_weights=None,
    diagonal_tie_break=0.0,
    c_source=1.0,
    c_target=1.0,
    solver="sparse",
    return_dense_plan=True,
    edge_mask=None,
)
```

Inputs:

- `c_source` and `c_target` can be scalars, arrays, or callables
  `f(mu, nu, C)`.
- `C` can be provided directly. If it is omitted, it is built from points or
  regular grid coordinates.
- `source_axes` / `target_axes` define explicit tensor-product grids.
- `periodic=[(axis, period), ...]` can be used for cyclic coordinates such as
  angles.
- `axis_weights=[...]` rescales coordinates in the transport cost, which is
  useful for anisotropic grids.
- `diagonal_tie_break>0` adds an optional APOT-style diagonal bias equal to the
  chosen strength times the smallest positive off-diagonal cost. This is useful
  to break degeneracies between equivalent transport plans and stabilize
  qualitative matches. It is disabled by default because it deliberately
  changes the objective.
- Source and target masses may have different total mass. Feasibility is handled
  by the two unmatched slack variables.

Solvers:

- `solver="highs"` solves the full linear program with SciPy HiGHS.
- `solver="sparse"` applies the exact IC-POT admissibility rule
  `C_ij <= c_source_i + c_target_j` before calling HiGHS on the reduced graph.
  Both solvers optimize the same IC-POT objective.
- `edge_mask` optionally restricts the transport graph. This changes the
  feasible problem and should be used deliberately.

Returned solution:

- `sol.dense_plan()` returns the dense transport plan.
- `sol.source_unmatched` and `sol.target_unmatched` are the slack variables
  `u` and `v`.
- `sol.objective` reports the transport and unmatched objective terms.
- `sol.stats` reports the solver backend, runtime, and sparse edge count.

## Checks

```bash
PYTHONPATH=src python tests/run_smoke_tests.py
PYTHONPATH=src python examples/basic_1d.py
```

The smoke tests check that the sparse solver matches the dense HiGHS objective,
that the IC-POT constraints hold, that unequal total masses are feasible, and
that the grid API works.

Disclaimer : LLM tools have been used to help develop this library
