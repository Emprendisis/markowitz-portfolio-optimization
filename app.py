import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
import datetime

# Configuración de la página
st.set_page_config(page_title="Markowitz Portfolio Pro", layout="wide")

st.title("📈 Optimización de Cartera: Modelo Markowitz")

# --- SIDEBAR: Configuración ---
st.sidebar.header("1. Selección de Activos")

# Paso 1: Número de activos
num_assets = st.sidebar.number_input("¿Cuántos activos deseas incluir?", min_value=2, max_value=20, value=4)

# Paso 2: Nombres de los activos (Tickers)
st.sidebar.subheader("Ingresa los Tickers")
default_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NVDA", "BTC-USD"]
tickers = []

for i in range(num_assets):
    # Toma un ticker por defecto de la lista si existe, si no, vacío
    default_val = default_tickers[i] if i < len(default_tickers) else ""
    t = st.sidebar.text_input(f"Activo #{i+1}", value=default_val, key=f"ticker_{i}")
    if t:
        tickers.append(t.strip().upper())

# Paso 3: Rango de fechas
st.sidebar.header("2. Rango de Fechas")
start_date = st.sidebar.date_input("Fecha de inicio", datetime.date(2021, 1, 1))
end_date = st.sidebar.date_input("Fecha de fin", datetime.date.today())

if end_date > datetime.date.today():
    st.sidebar.warning("La fecha de fin se ajustó a hoy.")
    end_date = datetime.date.today()

# Paso 4: Objetivo
st.sidebar.header("3. Objetivo de Optimización")
opt_type = st.sidebar.selectbox(
    "Selecciona tu meta:",
    ["Máximo Ratio Sharpe", "Rendimiento Objetivo", "Volatilidad Objetivo"]
)

target_val = 0.0
if opt_type == "Rendimiento Objetivo":
    target_val = st.sidebar.slider("Rendimiento Anualizado (%)", 5.0, 100.0, 25.0) / 100
elif opt_type == "Volatilidad Objetivo":
    target_val = st.sidebar.slider("Volatilidad Máxima Permitida (%)", 5.0, 50.0, 20.0) / 100

# --- PROCESAMIENTO ---

@st.cache_data
def load_data(tickers, start, end):
    try:
        data = yf.download(tickers, start=start, end=end, progress=False)
        if 'Adj Close' in data:
            df = data['Adj Close']
        else:
            # Si solo es un ticker, yfinance a veces no devuelve MultiIndex
            df = data[['Adj Close']]
        
        # Si hay varios activos pero fallan algunos, yf descarga lo que puede.
        # Limpiamos columnas totalmente vacías
        df = df.dropna(how='all', axis=1)
        # Limpiamos filas con NAs (días feriados, etc)
        df = df.dropna()
        return df
    except Exception as e:
        return pd.DataFrame()

if len(tickers) == num_assets:
    with st.spinner('Descargando datos de Yahoo Finance...'):
        df = load_data(tickers, start_date, end_date)

    if df.empty or df.shape[1] < 2:
        st.error("Error: No se pudieron obtener datos suficientes para los tickers ingresados. Revisa los nombres o las fechas.")
    else:
        try:
            # Cálculos de Markowitz
            mu = expected_returns.mean_historical_return(df)
            S = risk_models.sample_cov(df)

            # Optimización
            ef = EfficientFrontier(mu, S)
            
            if opt_type == "Máximo Ratio Sharpe":
                weights = ef.max_sharpe()
            elif opt_type == "Rendimiento Objetivo":
                weights = ef.efficient_return(target_return=target_val)
            else: # Volatilidad
                weights = ef.efficient_risk(target_risk=target_val)

            cleaned_weights = ef.clean_weights()
            ret, vol, sharpe = ef.portfolio_performance()

            # --- VISUALIZACIÓN ---
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("🎯 Resultados de la Cartera")
                st.metric("Rendimiento Esperado", f"{ret:.2%}")
                st.metric("Volatilidad (Riesgo)", f"{vol:.2%}")
                st.metric("Ratio Sharpe", f"{sharpe:.2f}")
                
                st.markdown("**Pesos Óptimos:**")
                pesos_data = pd.DataFrame.from_dict(cleaned_weights, orient='index', columns=['Proporción'])
                pesos_filtered = pesos_data[pesos_data['Proporción'] > 0]
                st.dataframe(pesos_filtered.style.format("{:.2%}"))

            with col2:
                st.subheader("🥧 Distribución de Activos")
                fig_pie = px.pie(
                    names=list(cleaned_weights.keys()),
                    values=list(cleaned_weights.values()),
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.T10
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            # --- GRÁFICA DE FRONTERA EFICIENTE ---
            st.subheader("📈 Frontera Eficiente y Ubicación de tu Cartera")
            
            # Simulación de carteras aleatorias
            n_samples = 1000
            w_rand = np.random.dirichlet(np.ones(len(mu)), n_samples)
            rets_rand = w_rand @ mu.values
            vols_rand = np.sqrt(np.diag(w_rand @ S.values @ w_rand.T))

            fig_frontier = go.Figure()

            # Nube de carteras
            fig_frontier.add_trace(go.Scatter(
                x=vols_rand, y=rets_rand,
                mode='markers',
                marker=dict(color='rgba(100, 100, 100, 0.3)', size=5),
                name='Carteras Aleatorias'
            ))

            # Punto de nuestra cartera optimizada
            fig_frontier.add_trace(go.Scatter(
                x=[vol], y=[ret],
                mode='markers',
                marker=dict(color='red', size=15, symbol='star', line=dict(width=2, color='white')),
                name='Tu Cartera Óptima'
            ))

            fig_frontier.update_layout(
                xaxis_title="Volatilidad Anualizada (Riesgo)",
                yaxis_title="Rendimiento Anualizado",
                template="plotly_white",
                height=600
            )
            st.plotly_chart(fig_frontier, use_container_width=True)

        except Exception as e:
            st.error(f"Error en la optimización: {e}")
            st.info("Sugerencia: El rendimiento objetivo puede ser demasiado alto para los activos seleccionados.")
else:
    st.info("Completa los nombres de los tickers en el panel de la izquierda para comenzar.")