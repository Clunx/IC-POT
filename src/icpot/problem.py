from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from icpot.costs import add_diagonal_tie_break, grid_points, pairwise_cost

ArrayLike = np.ndarray | Sequence[float]
CostInput = float | ArrayLike | Callable[[np.ndarray, np.ndarray, np.ndarray], ArrayLike]


@dataclass(frozen=True)
class ICPOTProblem:
    """Prepared finite-dimensional IC-POT problem."""

    mu: np.ndarray
    nu: np.ndarray
    C: np.ndarray
    c_source: np.ndarray
    c_target: np.ndarray
    source_shape: Optional[tuple[int, ...]]
    target_shape: Optional[tuple[int, ...]]

    @property
    def n_source(self) -> int:
        return int(self.mu.size)

    @property
    def n_target(self) -> int:
        return int(self.nu.size)


def _as_nonnegative_vector(x: ArrayLike, *, name: str) -> tuple[np.ndarray, tuple[int, ...]]:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        raise ValueError(f"{name} must contain at least one entry.")
    shape = tuple(arr.shape)
    vec = arr.reshape(-1)
    if not np.all(np.isfinite(vec)):
        raise ValueError(f"{name} contains NaN or Inf.")
    if np.any(vec < 0):
        raise ValueError(f"{name} contains negative entries.")
    return vec, shape


def _resolve_unmatched_cost(
    spec: CostInput,
    *,
    mu: np.ndarray,
    nu: np.ndarray,
    C: np.ndarray,
    side: str,
) -> np.ndarray:
    size = mu.size if side == "source" else nu.size
    if callable(spec):
        out = spec(mu, nu, C)
    else:
        out = spec
    arr = np.asarray(out, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(size, float(arr), dtype=np.float64)
    else:
        arr = arr.reshape(-1)
    if arr.shape != (size,):
        raise ValueError(f"{side} unmatched cost must have shape {(size,)}, got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{side} unmatched cost contains NaN or Inf.")
    if np.any(arr < 0):
        raise ValueError(f"{side} unmatched cost contains negative entries.")
    return arr


def prepare_problem(
    mu: ArrayLike,
    nu: ArrayLike,
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
    axis_weights: Optional[Sequence[float]] = None,
    diagonal_tie_break: float = 0.0,
    c_source: CostInput = 1.0,
    c_target: CostInput = 1.0,
    sanitize: bool = False,
) -> ICPOTProblem:
    """Validate inputs and build an IC-POT problem.

    This function implements the finite problem

    ``min <C,P> + <c_source,u> + <c_target,v>``

    subject to ``P 1 + u = mu``, ``P.T 1 + v = nu`` and nonnegativity.
    """

    mu_arr = np.asarray(mu, dtype=np.float64)
    nu_arr = np.asarray(nu, dtype=np.float64)
    if sanitize:
        mu_arr = np.clip(np.nan_to_num(mu_arr, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
        nu_arr = np.clip(np.nan_to_num(nu_arr, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)

    mu_vec, source_shape = _as_nonnegative_vector(mu_arr, name="mu")
    nu_vec, target_shape = _as_nonnegative_vector(nu_arr, name="nu")

    if C is None:
        if source_points is None:
            source_points = grid_points(source_shape, axes=source_axes, spacing=source_spacing)
        if target_points is None:
            target_points = grid_points(target_shape, axes=target_axes, spacing=target_spacing)
        C_arr = pairwise_cost(
            source_points,
            target_points,
            metric=metric,
            periodic=periodic,
            axis_weights=axis_weights,
        )
    else:
        C_arr = np.asarray(C, dtype=np.float64)

    if C_arr.shape != (mu_vec.size, nu_vec.size):
        raise ValueError(f"C must have shape {(mu_vec.size, nu_vec.size)}, got {C_arr.shape}.")
    if not np.all(np.isfinite(C_arr)):
        raise ValueError("C contains NaN or Inf.")
    if np.any(C_arr < 0):
        raise ValueError("C contains negative entries.")
    C_arr = add_diagonal_tie_break(C_arr, diagonal_tie_break)

    c_s = _resolve_unmatched_cost(c_source, mu=mu_vec, nu=nu_vec, C=C_arr, side="source")
    c_t = _resolve_unmatched_cost(c_target, mu=mu_vec, nu=nu_vec, C=C_arr, side="target")

    return ICPOTProblem(
        mu=mu_vec,
        nu=nu_vec,
        C=C_arr,
        c_source=c_s,
        c_target=c_t,
        source_shape=source_shape,
        target_shape=target_shape,
    )
