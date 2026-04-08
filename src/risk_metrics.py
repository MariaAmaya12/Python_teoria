from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.config import TRADING_DAYS
from scipy.stats import chi2


def annual_to_daily_rf(rf_annual: float, periods: int = TRADING_DAYS) -> float:
    """
    Convierte tasa libre de riesgo anual en decimal a tasa diaria equivalente.
    """
    if rf_annual is None or pd.isna(rf_annual):
        raise ValueError("La tasa libre de riesgo anual no es válida.")
    if rf_annual <= -1:
        raise ValueError("La tasa libre de riesgo anual debe ser mayor que -100%.")
    return (1 + rf_annual) ** (1 / periods) - 1


def validate_confidence_level(alpha: float) -> None:
    """
    Valida el nivel de confianza para VaR/CVaR.
    """
    if alpha is None or pd.isna(alpha):
        raise ValueError("El nivel de confianza no es válido.")
    if not (0 < alpha < 1):
        raise ValueError("El nivel de confianza debe estar entre 0 y 1.")


def validate_returns_series(returns: pd.Series, min_obs: int = 30) -> pd.Series:
    """
    Limpia y valida una serie de rendimientos.
    """
    if returns is None:
        raise ValueError("La serie de rendimientos es None.")

    clean = pd.to_numeric(returns, errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna()

    if len(clean) < min_obs:
        raise ValueError(
            f"La serie de rendimientos tiene {len(clean)} observaciones; "
            f"se requieren al menos {min_obs}."
        )

    if np.isclose(clean.std(ddof=1), 0):
        raise ValueError("La serie de rendimientos tiene varianza nula o casi nula.")

    return clean


def validate_weights(weights: np.ndarray, n_assets: int | None = None) -> np.ndarray:
    """
    Valida pesos de portafolio.
    """
    w = np.asarray(weights, dtype=float)

    if w.ndim != 1:
        raise ValueError("Los pesos deben ser un vector unidimensional.")

    if len(w) == 0:
        raise ValueError("El vector de pesos está vacío.")

    if np.any(np.isnan(w)) or np.any(np.isinf(w)):
        raise ValueError("Los pesos contienen NaN o infinitos.")

    if n_assets is not None and len(w) != n_assets:
        raise ValueError(
            f"El número de pesos ({len(w)}) no coincide con el número de activos ({n_assets})."
        )

    if not np.isclose(w.sum(), 1.0, atol=1e-6):
        raise ValueError("Los pesos del portafolio deben sumar 1.")

    return w


def parametric_var_cvar(returns: pd.Series, alpha: float = 0.95) -> dict:
    """
    VaR y CVaR paramétricos (normal).
    Se reportan como pérdidas positivas.
    """
    validate_confidence_level(alpha)

    try:
        r = validate_returns_series(returns)
    except ValueError:
        return {}

    mu = r.mean()
    sigma = r.std(ddof=1)
    q = 1 - alpha
    z = norm.ppf(q)

    var_daily = max(0.0, -(mu + sigma * z))
    cvar_daily = max(0.0, -(mu - sigma * norm.pdf(z) / q))

    return {
        "VaR_diario": float(var_daily),
        "CVaR_diario": float(cvar_daily),
        "VaR_anualizado": float(var_daily * np.sqrt(TRADING_DAYS)),
        "CVaR_anualizado": float(cvar_daily * np.sqrt(TRADING_DAYS)),
    }


def historical_var_cvar(returns: pd.Series, alpha: float = 0.95) -> dict:
    """
    VaR y CVaR históricos.
    Se reportan como pérdidas positivas.
    """
    validate_confidence_level(alpha)

    try:
        r = validate_returns_series(returns)
    except ValueError:
        return {}

    cutoff = np.quantile(r, 1 - alpha)
    tail = r[r <= cutoff]

    var_daily = max(0.0, -cutoff)
    cvar_daily = max(0.0, -tail.mean()) if len(tail) > 0 else 0.0

    return {
        "VaR_diario": float(var_daily),
        "CVaR_diario": float(cvar_daily),
        "VaR_anualizado": float(var_daily * np.sqrt(TRADING_DAYS)),
        "CVaR_anualizado": float(cvar_daily * np.sqrt(TRADING_DAYS)),
    }


def monte_carlo_var_cvar(
    returns_df: pd.DataFrame,
    weights: np.ndarray,
    alpha: float = 0.95,
    n_sim: int = 10000,
    seed: int = 42,
) -> dict:
    """
    VaR y CVaR de Monte Carlo usando normal multivariada.
    Se reportan como pérdidas positivas.
    """
    validate_confidence_level(alpha)

    if n_sim < 1000:
        raise ValueError("El número de simulaciones debe ser al menos 1000.")

    clean = returns_df.copy()
    clean = clean.apply(pd.to_numeric, errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna()

    if clean.empty or len(clean) < 30:
        return {}

    try:
        w = validate_weights(weights, n_assets=clean.shape[1])
    except ValueError:
        return {}

    mu = clean.mean().values
    cov = clean.cov().values

    if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
        return {}

    rng = np.random.default_rng(seed)
    sims = rng.multivariate_normal(mu, cov, size=n_sim)
    port_sim = sims @ w

    cutoff = np.quantile(port_sim, 1 - alpha)
    tail = port_sim[port_sim <= cutoff]

    var_daily = max(0.0, -cutoff)
    cvar_daily = max(0.0, -tail.mean()) if len(tail) > 0 else 0.0

    return {
        "VaR_diario": float(var_daily),
        "CVaR_diario": float(cvar_daily),
        "VaR_anualizado": float(var_daily * np.sqrt(TRADING_DAYS)),
        "CVaR_anualizado": float(cvar_daily * np.sqrt(TRADING_DAYS)),
        "simulated_returns": port_sim,
    }


def risk_comparison_table(
    portfolio_returns: pd.Series,
    asset_returns_df: pd.DataFrame,
    weights: np.ndarray,
    alpha: float = 0.95,
    n_sim: int = 10000,
) -> pd.DataFrame:
    """
    Tabla comparativa de VaR y CVaR.
    """
    p = parametric_var_cvar(portfolio_returns, alpha=alpha)
    h = historical_var_cvar(portfolio_returns, alpha=alpha)
    m = monte_carlo_var_cvar(
        asset_returns_df,
        weights=weights,
        alpha=alpha,
        n_sim=n_sim,
    )

    rows = []
    for name, res in [("Paramétrico", p), ("Histórico", h), ("Monte Carlo", m)]:
        if res:
            rows.append(
                {
                    "método": name,
                    "VaR_diario": res["VaR_diario"],
                    "CVaR_diario": res["CVaR_diario"],
                    "VaR_anualizado": res["VaR_anualizado"],
                    "CVaR_anualizado": res["CVaR_anualizado"],
                }
            )

    return pd.DataFrame(rows)

def kupiec_test(returns: pd.Series, var: float, alpha: float = 0.95) -> dict:
    """
    Test de Kupiec (Proportion of Failures - POF).

    Evalúa si el número de violaciones del VaR es consistente
    con el nivel de confianza esperado.

    - returns: serie de rendimientos
    - var: VaR como pérdida positiva (ej: 0.03)
    - alpha: nivel de confianza (ej: 0.95)
    """

    validate_confidence_level(alpha)

    try:
        r = validate_returns_series(returns)
    except ValueError:
        return {}

    # 🔥 violaciones: cuando pérdida supera el VaR
    violations = r < -var

    n = len(r)
    x = int(violations.sum())

    if n == 0:
        return {}

    p = 1 - alpha
    p_hat = x / n

    # ⚠️ evitar problemas numéricos extremos
    if p_hat == 0 or p_hat == 1:
        return {
            "violations": x,
            "n_obs": n,
            "expected_fail_rate": float(p),
            "observed_fail_rate": float(p_hat),
            "lr_stat": np.nan,
            "p_value": np.nan,
            "conclusion": "Insuficiente variabilidad para test",
        }

    # 🔥 Likelihood Ratio (Kupiec)
    lr = -2 * (
        (n - x) * np.log((1 - p) / (1 - p_hat)) +
        x * np.log(p / p_hat)
    )

    p_value = 1 - chi2.cdf(lr, df=1)

    conclusion = (
        "No se rechaza el modelo (VaR consistente)"
        if p_value > 0.05
        else "Se rechaza el modelo (VaR inconsistente)"
    )

    return {
        "violations": x,
        "n_obs": n,
        "expected_fail_rate": float(p),
        "observed_fail_rate": float(p_hat),
        "lr_stat": float(lr),
        "p_value": float(p_value),
        "conclusion": conclusion,
    }