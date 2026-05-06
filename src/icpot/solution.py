from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ObjectiveBreakdown:
    """Objective terms for the IC-POT slack problem."""

    transport: float
    unmatched_source: float
    unmatched_target: float

    @property
    def unmatched(self) -> float:
        return float(self.unmatched_source + self.unmatched_target)

    @property
    def total(self) -> float:
        return float(self.transport + self.unmatched)


@dataclass(frozen=True)
class SolverStats:
    """Small solver report."""

    solver: str
    success: bool
    status: str
    n_iter: Optional[int]
    runtime_s: float
    n_edges: int
    n_possible_edges: int

    @property
    def sparsity(self) -> float:
        if self.n_possible_edges == 0:
            return 0.0
        return 1.0 - float(self.n_edges) / float(self.n_possible_edges)


@dataclass(frozen=True)
class ICPOTSolution:
    """Solution of the IC-POT problem.

    The solution satisfies
    ``P 1 + u_source = mu`` and ``P.T 1 + u_target = nu`` up to solver
    tolerance. ``P`` is dense when requested; otherwise the edge-list fields
    contain the nonzero candidate variables.
    """

    P: Optional[np.ndarray]
    source_unmatched: np.ndarray
    target_unmatched: np.ndarray
    edge_source: np.ndarray
    edge_target: np.ndarray
    edge_mass: np.ndarray
    source_shape: Optional[Tuple[int, ...]]
    target_shape: Optional[Tuple[int, ...]]
    objective: ObjectiveBreakdown
    stats: SolverStats

    def dense_plan(self) -> np.ndarray:
        """Return the dense transport plan."""

        if self.P is not None:
            return self.P
        n = int(self.source_unmatched.size)
        m = int(self.target_unmatched.size)
        P = np.zeros((n, m), dtype=np.float64)
        if self.edge_mass.size:
            P[self.edge_source, self.edge_target] = self.edge_mass
        return P

    @property
    def matched_source(self) -> np.ndarray:
        return self.dense_plan().sum(axis=1)

    @property
    def matched_target(self) -> np.ndarray:
        return self.dense_plan().sum(axis=0)

    def source_unmatched_grid(self) -> np.ndarray:
        if self.source_shape is None:
            raise ValueError("source_shape is not available.")
        return self.source_unmatched.reshape(self.source_shape)

    def target_unmatched_grid(self) -> np.ndarray:
        if self.target_shape is None:
            raise ValueError("target_shape is not available.")
        return self.target_unmatched.reshape(self.target_shape)
