from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from icpot.problem import CostInput, prepare_problem
from icpot.solution import ICPOTSolution
from icpot.solvers import SolverName, SolverOptions, solve_linear_program


def solve_icpot(
    mu,
    nu,
    *,
    C: Optional[np.ndarray] = None,
    source_points: Optional[np.ndarray] = None,
    target_points: Optional[np.ndarray] = None,
    source_axes: Optional[Sequence[np.ndarray]] = None,
    target_axes: Optional[Sequence[np.ndarray]] = None,
    source_spacing: Optional[Sequence[float]] = None,
    target_spacing: Optional[Sequence[float]] = None,
    metric: str = "sqeuclidean",
    periodic: Optional[Sequence[tuple[int, float]]] = None,
    c_source: CostInput = 1.0,
    c_target: CostInput = 1.0,
    solver: SolverName = "sparse",
    return_dense_plan: bool = True,
    admissibility_tol: float = 1e-12,
    edge_mask: Optional[np.ndarray] = None,
    highs_options: Optional[dict] = None,
    sanitize: bool = False,
) -> ICPOTSolution:
    """Solve intent-controlled partial optimal transport.

    Parameters
    ----------
    mu, nu:
        Source and target nonnegative masses. They may have different shapes
        and different total mass.
    C:
        Pairwise transport cost matrix. If omitted, it is built from points or
        grid coordinates.
    c_source, c_target:
        Pointwise unmatched costs. Each can be a scalar, an array matching the
        corresponding marginal, or a callable ``f(mu, nu, C)``.
    solver:
        ``"highs"``/``"dense"`` solves the full LP. ``"sparse"``/``"sparse-highs"``
        first removes edges dominated by the pointwise unmatched costs and then
        solves the reduced LP with HiGHS.
    edge_mask:
        Optional user restriction on admissible transport edges. Passing this
        changes the feasible transport graph and should be used deliberately.
    """

    problem = prepare_problem(
        mu,
        nu,
        C=C,
        source_points=source_points,
        target_points=target_points,
        source_axes=source_axes,
        target_axes=target_axes,
        source_spacing=source_spacing,
        target_spacing=target_spacing,
        metric=metric,
        periodic=periodic,
        c_source=c_source,
        c_target=c_target,
        sanitize=sanitize,
    )
    options = SolverOptions(
        admissibility_tol=admissibility_tol,
        return_dense_plan=return_dense_plan,
        highs_options={} if highs_options is None else dict(highs_options),
    )
    return solve_linear_program(problem, solver=solver, options=options, edge_mask=edge_mask)
