import streamlit as st
import numpy as np

from src.config import ASSETS, DEFAULT_START_DATE, DEFAULT_END_DATE, ensure_project_dirs
from src.download import load_market_bundle
from src.preprocess import equal_weight_vector, equal_weight_portfolio
from src.risk_metrics import risk_comparison_table
from src.plots import plot_var_distribution
from src.risk_metrics import kupiec_test


ensure_project_dirs()
st.title("Módulo 5 - VaR y CVaR")

with st.sidebar:
    st.header("Parámetros de riesgo")
    start_date = st.date_input("Fecha inicial", value=DEFAULT_START_DATE, key="var_start")
    end_date = st.date_input("Fecha final", value=DEFAULT_END_DATE, key="var_end")
    alpha = st.selectbox("Nivel de confianza", [0.95, 0.99], index=0)
    n_sim = st.slider("Simulaciones Monte Carlo", min_value=5000, max_value=50000, value=10000, step=5000)

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

st.subheader("Portafolio equiponderado")
st.write("Pesos:", dict(zip(returns.columns, np.round(weights, 4))))

st.subheader("Comparación VaR / CVaR")
st.dataframe(table, width="stretch")

st.plotly_chart(plot_var_distribution(portfolio_returns, table), width="stretch")

st.info(
    f"""
    Con un nivel de confianza del {int(alpha*100)}%:
    
    - El **VaR** representa la pérdida máxima esperada en condiciones normales de mercado.
    - El **CVaR** (o Expected Shortfall) mide la pérdida promedio en los peores escenarios.
    
    Diferencias entre métodos:
    - **Paramétrico**: asume normalidad.
    - **Histórico**: usa datos reales.
    - **Monte Carlo**: simula escenarios futuros.
    """
)

st.subheader("Backtesting VaR - Test de Kupiec")

if not table.empty and "Histórico" in table["método"].values:

    var_hist_series = table.loc[table["método"] == "Histórico", "VaR_diario"]

    if not var_hist_series.empty:
        var_hist = float(var_hist_series.iloc[0])

        kupiec = kupiec_test(
            returns=portfolio_returns,
            var=var_hist,
            alpha=alpha
        )

    else:
        kupiec = {}

    if kupiec:
        col1, col2, col3 = st.columns(3)

        col1.metric("Violaciones", kupiec["violations"])
        col2.metric("Observadas (%)", f"{kupiec['observed_fail_rate']*100:.2f}%")
        col3.metric("Esperadas (%)", f"{kupiec['expected_fail_rate']*100:.2f}%")

        st.write(f"**p-value:** {kupiec['p_value']:.4f}")
        st.write(f"**Conclusión:** {kupiec['conclusion']}")

        # 🔥 AQUÍ VA LA MEJORA
        if kupiec["p_value"] > 0.05:
            st.success("El modelo VaR es consistente con los datos.")
        else:
            st.error("El modelo VaR NO es consistente (subestima o sobreestima riesgo).")

    else:
        st.warning("No fue posible ejecutar el test de Kupiec.")

else:
    st.warning("No hay VaR histórico disponible para ejecutar Kupiec.")