from __future__ import annotations

import numpy as np

from icpot import pairwise_cost, solve_icpot


def test_sparse_matches_dense_objective() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 2))
    y = rng.normal(size=(4, 2))
    C = pairwise_cost(x, y)
    mu = rng.uniform(0.1, 1.0, size=5)
    nu = rng.uniform(0.1, 1.0, size=4)
    c_source = rng.uniform(0.2, 1.0, size=5)
    c_target = rng.uniform(0.2, 1.0, size=4)

    dense = solve_icpot(mu, nu, C=C, c_source=c_source, c_target=c_target, solver="highs")
    sparse = solve_icpot(mu, nu, C=C, c_source=c_source, c_target=c_target, solver="sparse")

    assert np.isclose(dense.objective.total, sparse.objective.total, atol=1e-8)
    assert np.allclose(dense.matched_source + dense.source_unmatched, mu, atol=1e-8)
    assert np.allclose(dense.matched_target + dense.target_unmatched, nu, atol=1e-8)
    assert np.allclose(sparse.matched_source + sparse.source_unmatched, mu, atol=1e-8)
    assert np.allclose(sparse.matched_target + sparse.target_unmatched, nu, atol=1e-8)


def test_unequal_total_mass_is_feasible() -> None:
    mu = np.array([1.0, 2.0, 0.5])
    nu = np.array([0.4, 0.6])
    C = np.array([[0.0, 1.0], [1.0, 0.0], [0.2, 0.3]])

    sol = solve_icpot(mu, nu, C=C, c_source=0.5, c_target=0.5)

    assert np.allclose(sol.matched_source + sol.source_unmatched, mu, atol=1e-8)
    assert np.allclose(sol.matched_target + sol.target_unmatched, nu, atol=1e-8)
    assert sol.edge_mass.sum() <= min(mu.sum(), nu.sum()) + 1e-8


def test_grid_api_and_periodic_cost() -> None:
    mu = np.ones((3, 4))
    nu = np.ones((3, 4))
    sol = solve_icpot(
        mu,
        nu,
        c_source=1.0,
        c_target=1.0,
        periodic=[(1, 4.0)],
        solver="sparse",
    )

    assert sol.dense_plan().shape == (12, 12)
    assert sol.source_unmatched_grid().shape == (3, 4)
