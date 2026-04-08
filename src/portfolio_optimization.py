import numpy as np
import pandas as pd
from scipy.optimize import minimize


def optimize_target_return(returns: pd.DataFrame, target_return: float):
    """
    Encuentra el portafolio de mínima varianza dado un retorno objetivo.
    """

    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    n = len(mean_returns)

    def portfolio_volatility(weights):
        return np.sqrt(weights.T @ cov_matrix @ weights)

    def portfolio_return(weights):
        return weights @ mean_returns

    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w: portfolio_return(w) - target_return},
    )

    bounds = tuple((0, 1) for _ in range(n))

    initial_weights = np.repeat(1 / n, n)

    result = minimize(
        portfolio_volatility,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        return None

    return {
        "weights": result.x,
        "return": portfolio_return(result.x),
        "volatility": portfolio_volatility(result.x),
    }