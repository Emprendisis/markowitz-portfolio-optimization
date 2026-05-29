import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from dateutil.relativedelta import relativedelta
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier


# =========================
# Configuración general
# =========================
st.set_page_config(page_title="Markowitz Portfolio Pro", layout="wide")
st.title("Optimización de cartera: Modelo Markowitz")
st.caption("Descarga precios de Yahoo Finance, calcula rendimiento esperado, matriz de covarianza y pesos óptimos de cartera.")


@dataclass(frozen=True)
class FrequencyConfig:
    label: str
    yf_interval: str
    annualization: int


FREQUENCIES = {
    "Diaria": FrequencyConfig("Diaria", "1d", 252),
    "Semanal": FrequencyConfig("Semanal", "1wk", 52),
    "Mensual": FrequencyConfig("Mensual", "1mo", 12),
}

HORIZONS = {
    "1 día": relativedelta(days=1),
    "1 mes": relativedelta(months=1),
    "3 meses": relativedelta(months=3),
    "6 meses": relativedelta(months=6),
    "12 meses": relativedelta(months=12),
    "3 años": relativedelta(years=3),
    "5 años": relativedelta(years=5),
}

MIN_OBSERVATIONS = 3
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NVDA", "BTC-USD"]


# =========================
# Funciones auxiliares
# =========================
def normalize_prices(raw_data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Convierte la descarga de yfinance en una tabla limpia de precios."""
    if raw_data.empty:
        return pd.DataFrame()

    if isinstance(raw_data.columns, pd.MultiIndex):
        if "Adj Close" in raw_data.columns.get_level_values(0):
            prices = raw_data["Adj Close"].copy()
        elif "Close" in raw_data.columns.get_level_values(0):
            prices = raw_data["Close"].copy()
        else:
            return pd.DataFrame()
    else:
        price_col = "Adj Close" if "Adj Close" in raw_data.columns else "Close"
        if price_col not in raw_data.columns:
            return pd.DataFrame()
        col_name = tickers[0] if len(tickers) == 1 else "Precio"
        prices = raw_data[[price_col]].rename(columns={price_col: col_name})

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    prices = prices.dropna(axis=1, how="all")
    prices = prices.ffill().dropna(how="any")
    prices = prices.loc[:, ~prices.columns.duplicated()]
    return prices


@st.cache_data(show_spinner=False)
def load_data(tickers: list[str], start: dt.date, end: dt.date, interval: str) -> pd.DataFrame:
    """Descarga precios desde Yahoo Finance."""
    end_for_yf = end + dt.timedelta(days=1)  # yfinance trata end como fecha excluyente
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end_for_yf,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    return normalize_prices(raw, tickers)


def build_random_portfolios(mu: pd.Series, cov: pd.DataFrame, n_samples: int = 1500) -> pd.DataFrame:
    """Genera carteras aleatorias para visualizar riesgo-rendimiento."""
    weights = np.random.dirichlet(np.ones(len(mu)), n_samples)
    returns = weights @ mu.values
    variances = np.einsum("ij,jk,ik->i", weights, cov.values, weights)
    volatility = np.sqrt(np.maximum(variances, 0))
    sharpe = np.divide(returns, volatility, out=np.zeros_like(returns), where=volatility != 0)
    return pd.DataFrame({"Rendimiento": returns, "Volatilidad": volatility, "Sharpe": sharpe})


def validate_inputs(tickers: list[str], start_date: dt.date, end_date: dt.date) -> list[str]:
    errors = []
    if len(tickers) < 2:
        errors.append("Ingresa al menos dos tickers válidos.")
    if start_date >= end_date:
        errors.append("La fecha inicial debe ser anterior a la fecha final.")
    return errors


# =========================
# Sidebar
# =========================
st.sidebar.header("1. Activos")
num_assets = st.sidebar.number_input("Número de activos", min_value=2, max_value=20, value=4, step=1)

tickers: list[str] = []
for i in range(num_assets):
    default_value = DEFAULT_TICKERS[i] if i < len(DEFAULT_TICKERS) else ""
    ticker = st.sidebar.text_input(f"Ticker {i + 1}", value=default_value, key=f"ticker_{i}")
    ticker = ticker.strip().upper()
    if ticker:
        tickers.append(ticker)

tickers = list(dict.fromkeys(tickers))  # elimina duplicados conservando orden

st.sidebar.header("2. Base de datos")
frequency_label = st.sidebar.selectbox("Periodicidad de precios", list(FREQUENCIES.keys()), index=0)
horizon_label = st.sidebar.selectbox("Plazo de análisis", list(HORIZONS.keys()), index=4)

end_date = st.sidebar.date_input("Fecha final", value=dt.date.today(), max_value=dt.date.today())
start_date = end_date - HORIZONS[horizon_label]
freq = FREQUENCIES[frequency_label]

st.sidebar.info(
    f"Rango calculado: {start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}\n\n"
    f"Periodicidad: {frequency_label} | Factor anual: {freq.annualization}"
)

st.sidebar.header("3. Parámetros de optimización")
opt_type = st.sidebar.selectbox(
    "Objetivo",
    ["Máximo Ratio Sharpe", "Rendimiento Objetivo", "Volatilidad Objetivo"],
)

risk_free_rate = st.sidebar.number_input(
    "Tasa libre de riesgo anual (%)",
    min_value=-10.0,
    max_value=50.0,
    value=0.0,
    step=0.25,
) / 100

target_val = None
if opt_type == "Rendimiento Objetivo":
    target_val = st.sidebar.slider("Rendimiento anualizado objetivo (%)", 1.0, 100.0, 15.0, 0.5) / 100
elif opt_type == "Volatilidad Objetivo":
    target_val = st.sidebar.slider("Volatilidad anualizada objetivo (%)", 1.0, 100.0, 20.0, 0.5) / 100

run_model = st.sidebar.button("Ejecutar modelo", type="primary")


# =========================
# Ejecución
# =========================
input_errors = validate_inputs(tickers, start_date, end_date)

if input_errors:
    for err in input_errors:
        st.error(err)
    st.stop()

if not run_model:
    st.info("Configura los activos, periodicidad y plazo. Después presiona 'Ejecutar modelo'.")
    st.stop()

with st.spinner("Descargando y procesando precios..."):
    try:
        prices = load_data(tickers, start_date, end_date, freq.yf_interval)
    except Exception as exc:
        st.error(f"No fue posible descargar información desde Yahoo Finance: {exc}")
        st.stop()

if prices.empty or prices.shape[1] < 2:
    st.error("No hay datos suficientes para construir una cartera. Revisa tickers, periodicidad o plazo.")
    st.stop()

if len(prices) < MIN_OBSERVATIONS:
    st.error(
        f"La combinación seleccionada generó solo {len(prices)} observaciones. "
        "Selecciona un plazo mayor o una periodicidad más frecuente."
    )
    st.stop()

missing_tickers = sorted(set(tickers) - set(prices.columns.astype(str)))
if missing_tickers:
    st.warning(f"Yahoo Finance no devolvió información suficiente para: {', '.join(missing_tickers)}")

returns = prices.pct_change().dropna()
if returns.empty:
    st.error("No hay suficientes observaciones para calcular rendimientos.")
    st.stop()

try:
    mu = expected_returns.mean_historical_return(prices, frequency=freq.annualization)
    cov = risk_models.sample_cov(prices, frequency=freq.annualization)

    valid_assets = mu.replace([np.inf, -np.inf], np.nan).dropna().index
    mu = mu.loc[valid_assets]
    cov = cov.loc[valid_assets, valid_assets]
    prices = prices.loc[:, valid_assets]

    if len(mu) < 2:
        st.error("Después de limpiar datos, quedan menos de dos activos válidos.")
        st.stop()

    ef = EfficientFrontier(mu, cov, weight_bounds=(0, 1))

    if opt_type == "Máximo Ratio Sharpe":
        ef.max_sharpe(risk_free_rate=risk_free_rate)
    elif opt_type == "Rendimiento Objetivo":
        ef.efficient_return(target_return=target_val)
    else:
        ef.efficient_risk(target_risk=target_val)

    cleaned_weights = ef.clean_weights()
    portfolio_return, portfolio_volatility, portfolio_sharpe = ef.portfolio_performance(
        risk_free_rate=risk_free_rate
    )
except Exception as exc:
    st.error(f"Error en la optimización: {exc}")
    st.info("Ajusta el objetivo, aumenta el plazo o cambia la combinación de activos.")
    st.stop()

# =========================
# Resultados
# =========================
st.subheader("Resumen de configuración")
config_df = pd.DataFrame(
    {
        "Parámetro": ["Periodicidad", "Plazo", "Fecha inicial", "Fecha final", "Observaciones", "Activos válidos"],
        "Valor": [
            frequency_label,
            horizon_label,
            start_date.strftime("%d/%m/%Y"),
            end_date.strftime("%d/%m/%Y"),
            len(prices),
            ", ".join(prices.columns.astype(str)),
        ],
    }
)
st.dataframe(config_df, use_container_width=True, hide_index=True)

col1, col2, col3 = st.columns(3)
col1.metric("Rendimiento esperado anual", f"{portfolio_return:.2%}")
col2.metric("Volatilidad anual", f"{portfolio_volatility:.2%}")
col3.metric("Ratio Sharpe", f"{portfolio_sharpe:.2f}")

weights_df = (
    pd.DataFrame.from_dict(cleaned_weights, orient="index", columns=["Peso"])
    .query("Peso > 0")
    .sort_values("Peso", ascending=False)
)

left, right = st.columns([1, 1])
with left:
    st.subheader("Pesos óptimos")
    st.dataframe(weights_df.style.format({"Peso": "{:.2%}"}), use_container_width=True)

with right:
    st.subheader("Distribución de la cartera")
    fig_pie = px.pie(
        weights_df,
        names=weights_df.index,
        values="Peso",
        hole=0.4,
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)

st.subheader("Precios históricos")
normalized_prices = prices / prices.iloc[0] * 100
fig_prices = px.line(normalized_prices, labels={"value": "Índice base 100", "index": "Fecha", "variable": "Activo"})
st.plotly_chart(fig_prices, use_container_width=True)

st.subheader("Frontera eficiente aproximada")
random_portfolios = build_random_portfolios(mu, cov)
fig_frontier = go.Figure()
fig_frontier.add_trace(
    go.Scatter(
        x=random_portfolios["Volatilidad"],
        y=random_portfolios["Rendimiento"],
        mode="markers",
        marker=dict(size=5, opacity=0.35, color=random_portfolios["Sharpe"], colorscale="Viridis", showscale=True),
        name="Carteras simuladas",
    )
)
fig_frontier.add_trace(
    go.Scatter(
        x=[portfolio_volatility],
        y=[portfolio_return],
        mode="markers",
        marker=dict(size=16, symbol="star", color="red", line=dict(width=1, color="black")),
        name="Cartera óptima",
    )
)
fig_frontier.update_layout(
    xaxis_title="Volatilidad anualizada",
    yaxis_title="Rendimiento anualizado",
    height=600,
)
st.plotly_chart(fig_frontier, use_container_width=True)

st.subheader("Matriz de covarianza anualizada")
st.dataframe(cov.style.format("{:.6f}"), use_container_width=True)

st.subheader("Matriz de correlación")
corr = returns.loc[:, prices.columns].corr()
st.dataframe(corr.style.format("{:.4f}"), use_container_width=True)

st.download_button(
    label="Descargar precios utilizados en CSV",
    data=prices.to_csv().encode("utf-8"),
    file_name="precios_markowitz.csv",
    mime="text/csv",
)
