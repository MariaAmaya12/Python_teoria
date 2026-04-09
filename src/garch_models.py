from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from arch import arch_model


def fit_garch_models(log_returns: pd.Series) -> dict:
    """
    Ajusta ARCH(1), GARCH(1,1) y EGARCH(1,1) sobre una serie de rendimientos
    ya validada y escalada por 100.

    IMPORTANTE:
    - Esta función asume que la serie de entrada ya fue limpiada/validada.
    - No vuelve a escalar los retornos.
    """
    r = pd.to_numeric(log_returns, errors="coerce").dropna()

    if len(r) < 100:
        return {
            "comparison": pd.DataFrame(),
            "volatility": pd.DataFrame(),
            "forecast": pd.DataFrame(),
            "diagnostics": pd.DataFrame(),
            "best_model_name": None,
            "std_resid": pd.DataFrame(),
            "summary_text": "",
        }

    specs = {
        "ARCH(1)": arch_model(r, mean="Constant", vol="ARCH", p=1, q=0, dist="normal"),
        "GARCH(1,1)": arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="normal"),
        "EGARCH(1,1)": arch_model(r, mean="Constant", vol="EGARCH", p=1, q=1, dist="normal"),
    }

    comparison_rows = []
    vol_dict = {}
    fitted_results = {}
    errores_ajuste = []

    for name, model in specs.items():
        try:
            res = model.fit(disp="off")
            fitted_results[name] = res

            converged_flag = getattr(res, "convergence_flag", None)
            convergio = converged_flag == 0 if converged_flag is not None else True

            params = res.params.to_dict()
            alpha_1 = params.get("alpha[1]", np.nan)
            beta_1 = params.get("beta[1]", np.nan)
            omega = params.get("omega", np.nan)
            mu = params.get("mu", np.nan)

            persistencia = (
                float(alpha_1 + beta_1)
                if pd.notna(alpha_1) and pd.notna(beta_1)
                else np.nan
            )

            comparison_rows.append(
                {
                    "modelo": name,
                    "loglik": float(res.loglikelihood),
                    "AIC": float(res.aic),
                    "BIC": float(res.bic),
                    "convergió": "Sí" if convergio else "No",
                    "mu": float(mu) if pd.notna(mu) else np.nan,
                    "omega": float(omega) if pd.notna(omega) else np.nan,
                    "alpha_1": float(alpha_1) if pd.notna(alpha_1) else np.nan,
                    "beta_1": float(beta_1) if pd.notna(beta_1) else np.nan,
                    "persistencia": persistencia,
                }
            )

            vol = pd.Series(res.conditional_volatility, index=r.index, name=name)
            vol_dict[name] = vol

        except Exception as e:
            errores_ajuste.append({"modelo": name, "error": str(e)})

    if not comparison_rows:
        return {
            "comparison": pd.DataFrame(),
            "volatility": pd.DataFrame(),
            "forecast": pd.DataFrame(),
            "diagnostics": pd.DataFrame(errores_ajuste),
            "best_model_name": None,
            "std_resid": pd.DataFrame(),
            "summary_text": "Ningún modelo GARCH pudo ajustarse correctamente.",
        }

    comparison = (
        pd.DataFrame(comparison_rows)
        .sort_values(["AIC", "BIC"], ascending=True)
        .reset_index(drop=True)
    )

    best_model_name = comparison.iloc[0]["modelo"]
    best_result = fitted_results[best_model_name]

    try:
    forecast = best_result.forecast(horizon=10)
    forecast_var = forecast.variance.iloc[-1]
    except ValueError:
    forecast = best_result.forecast(horizon=10, method="simulation")
    forecast_var = forecast.variance.iloc[-1]
    
    forecast_df = pd.DataFrame(
        {
            "horizonte": list(range(1, len(forecast_var) + 1)),
            "varianza_pronosticada": forecast_var.values,
            "volatilidad_pronosticada": np.sqrt(forecast_var.values),
        }
    )

    std_resid = pd.Series(best_result.std_resid, index=r.index).dropna()
    jb = stats.jarque_bera(std_resid)

    best_row = comparison.iloc[0]
    alpha_best = best_row.get("alpha_1", np.nan)
    beta_best = best_row.get("beta_1", np.nan)
    persistencia_best = best_row.get("persistencia", np.nan)

    if pd.notna(persistencia_best):
        if persistencia_best >= 0.98:
            interpretacion_persistencia = (
                "La persistencia de la volatilidad es muy alta; los choques tardan bastante en disiparse."
            )
        elif persistencia_best >= 0.90:
            interpretacion_persistencia = (
                "La volatilidad muestra persistencia alta, consistente con clustering de volatilidad."
            )
        elif persistencia_best >= 0.70:
            interpretacion_persistencia = (
                "La persistencia de la volatilidad es moderada."
            )
        else:
            interpretacion_persistencia = (
                "La persistencia estimada de la volatilidad es relativamente baja."
            )
    else:
        interpretacion_persistencia = (
            "No fue posible calcular la persistencia del modelo seleccionado."
        )

    if pd.notna(alpha_best):
        if alpha_best >= 0.10:
            interpretacion_alpha = (
                "Los choques recientes tienen un efecto importante sobre la volatilidad actual."
            )
        else:
            interpretacion_alpha = (
                "El impacto inmediato de los choques recientes sobre la volatilidad es moderado o bajo."
            )
    else:
        interpretacion_alpha = (
            "El modelo seleccionado no reporta un parámetro alpha estándar interpretable."
        )

    diagnostics_rows = [
        {"metrica": "modelo_seleccionado", "valor": best_model_name},
        {"metrica": "convergió", "valor": best_row["convergió"]},
        {"metrica": "AIC", "valor": float(best_row["AIC"])},
        {"metrica": "BIC", "valor": float(best_row["BIC"])},
        {
            "metrica": "alpha_1",
            "valor": float(alpha_best) if pd.notna(alpha_best) else np.nan,
        },
        {
            "metrica": "beta_1",
            "valor": float(beta_best) if pd.notna(beta_best) else np.nan,
        },
        {
            "metrica": "persistencia_alpha_mas_beta",
            "valor": float(persistencia_best) if pd.notna(persistencia_best) else np.nan,
        },
        {"metrica": "jb_residuos_stat", "valor": float(jb.statistic)},
        {"metrica": "jb_residuos_pvalue", "valor": float(jb.pvalue)},
        {"metrica": "interpretacion_alpha", "valor": interpretacion_alpha},
        {"metrica": "interpretacion_persistencia", "valor": interpretacion_persistencia},
    ]

    if errores_ajuste:
        for err in errores_ajuste:
            diagnostics_rows.append(
                {
                    "metrica": f"error_{err['modelo']}",
                    "valor": err["error"],
                }
            )

    diagnostics = pd.DataFrame(diagnostics_rows)
    diagnostics["valor"] = diagnostics["valor"].astype(str)

    vol_df = pd.concat(vol_dict.values(), axis=1)

    summary_text = (
        f"El modelo seleccionado fue {best_model_name} por presentar el menor AIC. "
        f"{interpretacion_alpha} {interpretacion_persistencia}"
    )

    return {
        "comparison": comparison,
        "volatility": vol_df,
        "forecast": forecast_df,
        "diagnostics": diagnostics,
        "best_model_name": best_model_name,
        "std_resid": std_resid.to_frame("std_resid"),
        "summary_text": summary_text,
    }