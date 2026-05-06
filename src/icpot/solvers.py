from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from icpot.problem import ICPOTProblem
from icpot.solution import ICPOTSolution, ObjectiveBreakdown, SolverStats

SolverName = Literal["highs", "dense", "sparse", "sparse-highs"]


@dataclass(frozen=True)
class SolverOptions:
    """Options shared by the exact LP solvers."""

    admissibility_tol: float = 1e-12
    return_dense_plan: bool = True
    highs_options: dict = field(default_factory=dict)


def admissible_edge_mask(problem: ICPOTProblem, *, tol: float = 1e-12) -> np.ndarray:
    """Exact IC-POT admissibility mask.

    If ``C_ij > c_source_i + c_target_j``, transporting along edge ``(i,j)``
    is dominated by leaving the two units unmatched. Removing those edges does
    not change the optimum of the linear slack problem.
    """

    return problem.C <= (problem.c_source[:, None] + problem.c_target[None, :] + float(tol))


def solve_linear_program(
    problem: ICPOTProblem,
    *,
    solver: SolverName = "sparse",
    options: Optional[SolverOptions] = None,
    edge_mask: Optional[np.ndarray] = None,
) -> ICPOTSolution:
    """Solve IC-POT with SciPy HiGHS.

    ``solver='highs'`` or ``'dense'`` keeps all transport edges. ``solver='sparse'``
    or ``'sparse-highs'`` first applies the admissibility rule from the paper,
    then solves the same LP on the reduced edge set.
    """

    opts = options if options is not None else SolverOptions()
    t0 = time.time()

    n = problem.n_source
    m = problem.n_target
    if solver in {"highs", "dense"}:
        mask = np.ones((n, m), dtype=bool)
    elif solver in {"sparse", "sparse-highs"}:
        mask = admissible_edge_mask(problem, tol=opts.admissibility_tol)
    else:
        raise ValueError(f"Unknown solver {solver!r}.")

    if edge_mask is not None:
        edge_mask = np.asarray(edge_mask, dtype=bool)
        if edge_mask.shape != (n, m):
            raise ValueError(f"edge_mask must have shape {(n, m)}, got {edge_mask.shape}.")
        mask = mask & edge_mask

    edge_i, edge_j = np.where(mask)
    n_edges = int(edge_i.size)
    n_vars = n_edges + n + m

    objective = np.concatenate(
        [
            problem.C[edge_i, edge_j],
            problem.c_source,
            problem.c_target,
        ]
    )

    # Equality constraints:
    #   sum_j P_ij + u_i = mu_i
    #   sum_i P_ij + v_j = nu_j
    rows = []
    cols = []
    data = []
    if n_edges:
        rows.extend(edge_i.tolist())
        cols.extend(range(n_edges))
        data.extend([1.0] * n_edges)

        rows.extend((n + edge_j).tolist())
        cols.extend(range(n_edges))
        data.extend([1.0] * n_edges)

    source_slack_cols = np.arange(n_edges, n_edges + n)
    target_slack_cols = np.arange(n_edges + n, n_edges + n + m)

    rows.extend(range(n))
    cols.extend(source_slack_cols.tolist())
    data.extend([1.0] * n)

    rows.extend(range(n, n + m))
    cols.extend(target_slack_cols.tolist())
    data.extend([1.0] * m)

    A_eq = coo_matrix((data, (rows, cols)), shape=(n + m, n_vars)).tocsr()
    b_eq = np.concatenate([problem.mu, problem.nu])

    res = linprog(
        c=objective,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=(0.0, None),
        method="highs",
        options=opts.highs_options,
    )
    runtime = time.time() - t0
    if not res.success:
        raise RuntimeError(f"HiGHS failed with status {res.status}: {res.message}")

    x = np.asarray(res.x, dtype=np.float64)
    edge_mass = x[:n_edges]
    u_source = x[n_edges : n_edges + n]
    u_target = x[n_edges + n :]

    keep = edge_mass > max(1e-14, 10.0 * opts.admissibility_tol)
    nz_i = edge_i[keep]
    nz_j = edge_j[keep]
    nz_mass = edge_mass[keep]

    if opts.return_dense_plan:
        P = np.zeros((n, m), dtype=np.float64)
        if n_edges:
            P[edge_i, edge_j] = edge_mass
    else:
        P = None

    transport_cost = float(np.dot(edge_mass, problem.C[edge_i, edge_j])) if n_edges else 0.0
    unmatched_source_cost = float(np.dot(u_source, problem.c_source))
    unmatched_target_cost = float(np.dot(u_target, problem.c_target))

    stats = SolverStats(
        solver=solver,
        success=True,
        status=str(res.message),
        n_iter=getattr(res, "nit", None),
        runtime_s=runtime,
        n_edges=n_edges,
        n_possible_edges=n * m,
    )

    return ICPOTSolution(
        P=P,
        source_unmatched=u_source,
        target_unmatched=u_target,
        edge_source=nz_i,
        edge_target=nz_j,
        edge_mass=nz_mass,
        source_shape=problem.source_shape,
        target_shape=problem.target_shape,
        objective=ObjectiveBreakdown(
            transport=transport_cost,
            unmatched_source=unmatched_source_cost,
            unmatched_target=unmatched_target_cost,
        ),
        stats=stats,
    )
