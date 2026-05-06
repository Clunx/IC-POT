from __future__ import annotations

from test_icpot import (
    test_axis_weights_change_transport_cost,
    test_diagonal_tie_break_is_optional,
    test_grid_api_and_periodic_cost,
    test_sparse_matches_dense_objective,
    test_unequal_total_mass_is_feasible,
)


def main() -> None:
    test_sparse_matches_dense_objective()
    test_unequal_total_mass_is_feasible()
    test_grid_api_and_periodic_cost()
    test_axis_weights_change_transport_cost()
    test_diagonal_tie_break_is_optional()
    print("All IC-POT smoke tests passed.")


if __name__ == "__main__":
    main()
