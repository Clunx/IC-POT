from __future__ import annotations

from itertools import product
from typing import Iterable, Literal, Optional, Sequence

import numpy as np

Metric = Literal["sqeuclidean", "euclidean"]


def grid_points(
    shape: Sequence[int],
    *,
    axes: Optional[Sequence[np.ndarray]] = None,
    spacing: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Return flattened coordinates for a tensor-product grid.

    Parameters
    ----------
    shape:
        Number of grid cells along each axis.
    axes:
        Optional explicit coordinate values for each axis.
    spacing:
        Optional axis spacing used when ``axes`` is not provided.
    """

    shape = tuple(int(s) for s in shape)
    if any(s <= 0 for s in shape):
        raise ValueError("All grid dimensions must be positive.")

    if axes is not None and spacing is not None:
        raise ValueError("Pass either axes or spacing, not both.")

    if axes is None:
        if spacing is None:
            spacing = tuple(1.0 for _ in shape)
        if len(spacing) != len(shape):
            raise ValueError("spacing must have one value per axis.")
        axes = [np.arange(s, dtype=np.float64) * float(h) for s, h in zip(shape, spacing)]
    else:
        if len(axes) != len(shape):
            raise ValueError("axes must have one array per axis.")
        axes = [np.asarray(a, dtype=np.float64).reshape(-1) for a in axes]
        for size, axis in zip(shape, axes):
            if axis.size != size:
                raise ValueError("Each axis length must match shape.")

    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([x.reshape(-1) for x in mesh], axis=1)


def pairwise_cost(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    metric: Metric = "sqeuclidean",
    periodic: Optional[Iterable[tuple[int, float]]] = None,
    axis_weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Build a pairwise transport cost matrix.

    ``periodic`` may contain ``(axis, period)`` pairs. For those coordinates,
    distances are computed on the circle by taking the shortest wrapped
    displacement. ``axis_weights`` rescales each coordinate before computing
    Euclidean distances, which is useful for anisotropic grids.
    """

    x = np.asarray(source_points, dtype=np.float64)
    y = np.asarray(target_points, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("source_points and target_points must be 2D arrays.")
    if x.shape[1] != y.shape[1]:
        raise ValueError("source_points and target_points must have the same dimension.")
    if axis_weights is None:
        weights = np.ones(x.shape[1], dtype=np.float64)
    else:
        weights = np.asarray(axis_weights, dtype=np.float64).reshape(-1)
        if weights.shape != (x.shape[1],):
            raise ValueError(f"axis_weights must have shape {(x.shape[1],)}, got {weights.shape}.")
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("axis_weights must contain finite positive values.")

    diff = x[:, None, :] - y[None, :, :]
    if periodic is not None:
        for axis, period in periodic:
            axis = int(axis)
            period = float(period)
            if period <= 0:
                raise ValueError("period must be positive.")
            d = np.abs(diff[..., axis])
            diff[..., axis] = np.minimum(d, period - d)

    diff = diff * weights[None, None, :]
    sq = np.sum(diff * diff, axis=-1)
    if metric == "sqeuclidean":
        return sq
    if metric == "euclidean":
        return np.sqrt(sq)
    raise ValueError(f"Unknown metric: {metric!r}.")


def add_diagonal_tie_break(C: np.ndarray, strength: float) -> np.ndarray:
    """Add a small APOT-style diagonal tie-break to a square-like cost matrix.

    The added value is ``strength`` times the smallest positive off-diagonal
    cost and is applied to entries ``C[i, i]`` for ``i < min(n_source,n_target)``.
    This deliberately changes the optimization objective and is therefore
    disabled by default in the high-level API.
    """

    strength = float(strength)
    if strength < 0 or not np.isfinite(strength):
        raise ValueError("diagonal_tie_break must be finite and nonnegative.")
    if strength == 0:
        return np.asarray(C, dtype=np.float64)

    out = np.asarray(C, dtype=np.float64).copy()
    n, m = out.shape
    k = min(n, m)
    if k <= 1:
        return out

    mask = np.ones((n, m), dtype=bool)
    diag = np.arange(k)
    mask[diag, diag] = False
    vals = out[mask]
    vals = vals[vals > 0]
    if vals.size == 0:
        return out
    out[diag, diag] += strength * float(np.min(vals))
    return out


def score_to_unmatched_cost(
    score: np.ndarray,
    *,
    c_min: float,
    c_max: float,
    clip: bool = True,
) -> np.ndarray:
    """Map a protection score in ``[0, 1]`` to an unmatched cost.

    High scores make rejection expensive; low scores make rejection cheap.
    """

    score = np.asarray(score, dtype=np.float64)
    if clip:
        score = np.clip(score, 0.0, 1.0)
    if c_min < 0 or c_max < 0:
        raise ValueError("unmatched costs must be nonnegative.")
    if c_max < c_min:
        raise ValueError("c_max must be greater than or equal to c_min.")
    return float(c_min) + (float(c_max) - float(c_min)) * score


def all_grid_edges(source_shape: Sequence[int], target_shape: Sequence[int]) -> np.ndarray:
    """Return a dense boolean edge mask for two grids."""

    n = int(np.prod(tuple(source_shape)))
    m = int(np.prod(tuple(target_shape)))
    return np.ones((n, m), dtype=bool)
