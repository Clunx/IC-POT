"""Intent-controlled partial optimal transport."""

from icpot.api import solve_icpot
from icpot.costs import add_diagonal_tie_break, grid_points, pairwise_cost, score_to_unmatched_cost
from icpot.problem import ICPOTProblem, prepare_problem
from icpot.solution import ICPOTSolution, ObjectiveBreakdown, SolverStats

__all__ = [
    "ICPOTProblem",
    "ICPOTSolution",
    "ObjectiveBreakdown",
    "SolverStats",
    "add_diagonal_tie_break",
    "grid_points",
    "pairwise_cost",
    "prepare_problem",
    "score_to_unmatched_cost",
    "solve_icpot",
]

__version__ = "0.1.0"
