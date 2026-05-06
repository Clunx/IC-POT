from __future__ import annotations

import numpy as np

from icpot import solve_icpot


def main() -> None:
    x = np.linspace(0.0, 1.0, 40)[:, None]
    y = np.linspace(0.0, 1.0, 40)[:, None]
    mu = np.exp(-0.5 * ((x[:, 0] - 0.35) / 0.07) ** 2)
    nu = np.exp(-0.5 * ((y[:, 0] - 0.38) / 0.07) ** 2)
    nu += 0.6 * np.exp(-0.5 * ((y[:, 0] - 0.75) / 0.05) ** 2)

    # Protect the left target mode and make the right target mode cheap to reject.
    c_source = np.full_like(mu, 0.2)
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
    print(f"objective={sol.objective.total:.6f}")
    print(f"transported mass={sol.edge_mass.sum():.6f}")
    print(f"source unmatched={sol.source_unmatched.sum():.6f}")
    print(f"target unmatched={sol.target_unmatched.sum():.6f}")


if __name__ == "__main__":
    main()
