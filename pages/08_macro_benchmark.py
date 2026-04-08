import streamlit as st

from src.config import (
    ASSETS,
    GLOBAL_BENCHMARK,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    ensure_project_dirs,
)
from src.download import load_market_bundle
from src.preprocess import equal_weight_portfolio
from src.api.macro import macro_snapshot
from src.benchmark import benchmark_summary
from src.plots import plot_benchmark_base100

ensure_project_dirs()
st.title("Módulo 8 - Contexto macro y benchmark")

with st.sidebar:
    st.header("Parámetros")
    start_date = st.date_input("Fecha inicial", value=DEFAULT_START_DATE, key="bm_start")
    end_date = st.date_input("Fecha final", value=DEFAULT_END_DATE, key="bm_end")

tickers = [meta["ticker"] for meta in ASSETS.values()] + [GLOBAL_BENCHMARK]
bundle = load_market_bundle(tickers=tickers, start=str(start_date), end=str(end_date))
returns = bundle["returns"].dropna()

if returns.empty or GLOBAL_BENCHMARK not in returns.columns:
    st.error("No fue posible construir benchmark global.")
    st.stop()

portfolio_returns = equal_weight_portfolio(returns[[c for c in returns.columns if c != GLOBAL_BENCHMARK]])
benchmark_returns = returns[GLOBAL_BENCHMARK]

macro = macro_snapshot()
rf_annual = macro["risk_free_rate_pct"] / 100 if macro["risk_free_rate_pct"] == macro["risk_free_rate_pct"] else 0.03

if macro["inflation_yoy"] != macro["inflation_yoy"]:
    st.warning("No se pudo obtener inflación desde API. Usando fallback o valor no disponible.")

if macro["usdcop_market"] != macro["usdcop_market"]:
    st.warning("No se pudo obtener USD/COP spot desde API. Usando fallback o valor no disponible.")

if macro["cop_per_usd"] != macro["cop_per_usd"]:
    st.warning("No se pudo obtener USD/COP promedio anual desde API. Usando fallback o valor no disponible.")

col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Tasa libre de riesgo (%)",
    f"{macro['risk_free_rate_pct']:.2f}" if macro["risk_free_rate_pct"] == macro["risk_free_rate_pct"] else "N/D"
)
col2.metric(
    "Inflación interanual",
    f"{macro['inflation_yoy']:.2%}" if macro["inflation_yoy"] == macro["inflation_yoy"] else "N/D"
)
col3.metric(
    "USD/COP (spot)",
    f"{macro['usdcop_market']:.2f}" if macro["usdcop_market"] == macro["usdcop_market"] else "N/D"
)
col4.metric(
    "USD/COP (promedio anual)",
    f"{macro['cop_per_usd']:.2f}" if macro["cop_per_usd"] == macro["cop_per_usd"] else "N/D"
)

summary_df, extras_df, cum_port, cum_bench = benchmark_summary(
    portfolio_returns=portfolio_returns,
    benchmark_returns=benchmark_returns,
    rf_annual=rf_annual,
)

st.plotly_chart(plot_benchmark_base100(cum_port, cum_bench), width="stretch")

st.subheader("Desempeño: portafolio vs benchmark")
st.dataframe(summary_df, width="stretch")

st.subheader("Métricas adicionales")
st.dataframe(extras_df, width="stretch")

st.info(
    "Este módulo compara el portafolio con un benchmark global y resume alpha, tracking error e information ratio."
)
