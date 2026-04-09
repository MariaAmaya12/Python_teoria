import streamlit as st
import numpy as np

from src.config import ASSETS, DEFAULT_START_DATE, DEFAULT_END_DATE, ensure_project_dirs
from src.download import load_market_bundle
from src.preprocess import equal_weight_vector, equal_weight_portfolio
from src.risk_metrics import risk_comparison_table, kupiec_test
from src.plots import plot_var_distribution

ensure_project_dirs()
st.title("Módulo 5 - VaR y CVaR")

# ==============================
# Sidebar
# ==============================
with st.sidebar:
    st.header("Parámetros de riesgo")
    start_date = st.date_input("Fecha inicial", value=DEFAULT_START_DATE, key="var_start")
    end_date = st.date_input("Fecha final", value=DEFAULT_END_DATE, key="var_end")
    alpha = st.selectbox("Nivel de confianza", [0.95, 0.99], index=0)
    n_sim = st.slider(
        "Simulaciones Monte Carlo",
        min_value=5000,
        max_value=50000,
        value=10000,
        step=5000,
    )

# ==============================
# Carga y preparación de datos
# ==============================
tickers = [meta["ticker"] for meta in ASSETS.values()]
bundle = load_market_bundle(tickers=tickers, start=str(start_date), end=str(end_date))
returns = bundle["returns"].replace([np.inf, -np.inf], np.nan).dropna()

if returns.empty or len(returns) < 30:
    st.error("No hay suficientes datos para calcular métricas de riesgo.")
    st.stop()

weights = equal_weight_vector(returns.shape[1])
portfolio_returns = equal_weight_portfolio(returns)

table = risk_comparison_table(
    portfolio_returns=portfolio_returns,
    asset_returns_df=returns,
    weights=weights,
    alpha=alpha,
    n_sim=n_sim,
)

if table.empty:
    st.error("No fue posible calcular VaR y CVaR con los datos disponibles.")
    st.stop()

# ==============================
# Explicación metodológica
# ==============================
st.markdown(
    f"""
    En este módulo se comparan medidas de riesgo extremo para un portafolio equiponderado
    con un nivel de confianza del **{int(alpha * 100)}%**.

    **Convención usada en este proyecto:**
    - El **VaR** y el **CVaR** se reportan como **pérdidas positivas**.
    - Por ejemplo, un VaR diario de **0.025** significa una pérdida potencial de **2.5%**.
    """
)

st.info(
    """
    **Interpretación de métodos:**
    - **Paramétrico**: supone normalidad de los rendimientos.
    - **Histórico**: usa la distribución empírica observada.
    - **Monte Carlo**: simula escenarios futuros a partir de media y covarianza.
    """
)

# ==============================
# Portafolio
# ==============================
st.subheader("Portafolio equiponderado")
st.write("Pesos:", dict(zip(returns.columns, np.round(weights, 4))))

# ==============================
# Resultados VaR / CVaR
# ==============================
st.subheader("Comparación VaR / CVaR")
st.dataframe(table, width="stretch")

st.plotly_chart(plot_var_distribution(portfolio_returns, table), width="stretch")

# ==============================
# Interpretación automática
# ==============================
var_hist_row = table.loc[table["método"] == "Histórico"]
var_param_row = table.loc[table["método"] == "Paramétrico"]
var_mc_row = table.loc[table["método"] == "Monte Carlo"]

mensajes = []

if not var_hist_row.empty:
    var_h = float(var_hist_row["VaR_diario"].iloc[0])
    cvar_h = float(var_hist_row["CVaR_diario"].iloc[0])
    mensajes.append(
        f"Con {int(alpha * 100)}% de confianza, el **VaR histórico diario** del portafolio es "
        f"**{var_h:.2%}**, mientras que el **CVaR histórico diario** es **{cvar_h:.2%}**. "
        "Esto implica que, en los peores escenarios observados, la pérdida promedio sería más severa que el umbral VaR."
    )

if not var_param_row.empty and not var_hist_row.empty:
    var_p = float(var_param_row["VaR_diario"].iloc[0])
    var_h = float(var_hist_row["VaR_diario"].iloc[0])

    if var_p < var_h:
        mensajes.append(
            "El VaR paramétrico es menor que el VaR histórico, lo que puede sugerir que el supuesto de normalidad "
            "subestima el riesgo extremo frente a la evidencia empírica."
        )
    elif var_p > var_h:
        mensajes.append(
            "El VaR paramétrico es mayor que el VaR histórico, lo que sugiere una estimación más conservadora "
            "bajo el supuesto normal."
        )
    else:
        mensajes.append(
            "El VaR paramétrico y el VaR histórico son muy similares para esta muestra."
        )

if not var_mc_row.empty:
    var_mc = float(var_mc_row["VaR_diario"].iloc[0])
    mensajes.append(
        f"El **VaR Monte Carlo diario** estimado es **{var_mc:.2%}**, útil para contrastar "
        "la sensibilidad del riesgo ante simulaciones probabilísticas."
    )

for msg in mensajes:
    st.info(msg)

st.warning(
    "El VaR paramétrico depende del supuesto de normalidad. Si la distribución de rendimientos presenta colas pesadas "
    "o asimetría, este método puede subestimar el riesgo extremo."
)

# ==============================
# Backtesting VaR - Test de Kupiec
# ==============================
st.subheader("Backtesting VaR - Test de Kupiec")

if not var_hist_row.empty:
    var_hist = float(var_hist_row["VaR_diario"].iloc[0])

    kupiec = kupiec_test(
        returns=portfolio_returns,
        var=var_hist,
        alpha=alpha,
    )

    if kupiec:
        col1, col2, col3 = st.columns(3)
        col1.metric("Violaciones", kupiec["violations"])
        col2.metric("Observadas (%)", f"{kupiec['observed_fail_rate'] * 100:.2f}%")
        col3.metric("Esperadas (%)", f"{kupiec['expected_fail_rate'] * 100:.2f}%")

        st.write(f"**p-value:** {kupiec['p_value']:.4f}")
        st.write(f"**Conclusión:** {kupiec['conclusion']}")

        if kupiec["p_value"] > 0.05:
            st.success(
                "El VaR histórico es consistente con la frecuencia de pérdidas observadas en la muestra."
            )
        else:
            st.error(
                "El VaR histórico no es consistente con la frecuencia de pérdidas observadas. "
                "Esto sugiere que el modelo puede estar subestimando o sobreestimando el riesgo."
            )

        st.info(
            "El test de Kupiec compara la proporción esperada de violaciones del VaR con la proporción observada. "
            "Es una forma de evaluar si el modelo de riesgo está calibrado de manera razonable."
        )
    else:
        st.warning("No fue posible ejecutar el test de Kupiec.")
else:
    st.warning("No hay VaR histórico disponible para ejecutar el test de Kupiec.")