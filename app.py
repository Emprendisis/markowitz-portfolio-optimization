import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns, plotting

# Configuración de la página
st.set_page_config(page_title="Optimizador de Cartera Markowitz", layout="wide")

st.title("📈 Modelo de Frontera Eficiente de Markowitz")
st.markdown("""
Esta aplicación optimiza una cartera de inversión utilizando la Teoría Moderna de Cartera.
Selecciona tus activos, elige un objetivo y visualiza la distribución óptima.
""")

# --- SIDEBAR: Parámetros de entrada ---
st.sidebar.header("Configuración de la Cartera")

# Selección de activos
tickers_input = st.sidebar.text_input(
    "Ingresa los Tickers de Yahoo Finance (separados por coma):",
    "AAPL, MSFT, GOOGL, AMZN, TSLA, GOLD"
)
tickers = [t.strip().upper() for t in tickers_input.split(",")]

# Rango de fechas
start_date = st.sidebar.date_input("Fecha de inicio", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("Fecha de fin", pd.to_datetime("today"))

# Selección de Objetivo
st.sidebar.subheader("Objetivo de Optimización")
opt_type = st.sidebar.radio(
    "Selecciona tu objetivo:",
    ("Rendimiento Objetivo", "Volatilidad Objetivo", "Máximo Ratio Sharpe")
)

target_val = 0.0
if opt_type == "Rendimiento Objetivo":
    target_val = st.sidebar.slider("Rendimiento Anualizado Objetivo (%)", 5.0, 100.0, 20.0) / 100
elif opt_type == "Volatilidad Objetivo":
    target_val = st.sidebar.slider("Volatilidad Anualizada Objetivo (%)", 5.0, 50.0, 15.0) / 100

# --- PROCESAMIENTO DE DATOS ---
@st.cache_data
def get_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    return data

try:
    df = get_data(tickers, start_date, end_date)

    if df.empty or len(tickers) < 2:
        st.error("Por favor ingresa al menos 2 tickers válidos.")
    else:
        # Calcular rendimientos esperados y matriz de covarianza
        mu = expected_returns.mean_historical_return(df)
        S = risk_models.sample_cov(df)

        # Optimización
        ef = EfficientFrontier(mu, S)
        
        try:
            if opt_type == "Rendimiento Objetivo":
                weights = ef.efficient_return(target_return=target_val)
            elif opt_type == "Volatilidad Objetivo":
                weights = ef.efficient_risk(target_risk=target_val)
            else:
                weights = ef.max_sharpe()
            
            cleaned_weights = ef.clean_weights()
            performance = ef.portfolio_performance(verbose=False)

            # --- RESULTADOS EN COLUMNAS ---
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Resultados de la Optimización")
                perf_df = pd.DataFrame({
                    "Métrica": ["Rendimiento Esperado", "Volatilidad Anual", "Ratio Sharpe"],
                    "Valor": [f"{performance[0]:.2%}", f"{performance[1]:.2%}", f"{performance[2]:.2f}"]
                })
                st.table(perf_df)

                st.subheader("Distribución de Activos")
                weights_df = pd.DataFrame.from_dict(cleaned_weights, orient='index', columns=['Peso'])
                weights_df = weights_df[weights_df['Peso'] > 0] # Mostrar solo los que tienen peso
                weights_df['Peso (%)'] = (weights_df['Peso'] * 100).round(2)
                st.dataframe(weights_df[['Peso (%)']])

            with col2:
                st.subheader("🥧 Composición de la Cartera")
                fig_pie = px.pie(
                    weights_df, 
                    values='Peso', 
                    names=weights_df.index, 
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig_pie)

            # --- GRÁFICA DE FRONTERA EFICIENTE ---
            st.subheader("📈 Frontera Eficiente")
            
            # Generar puntos de la frontera
            n_samples = 100
            risks = []
            returns = []
            
            # Re-instanciar ef para graficar
            ef_plot = EfficientFrontier(mu, S)
            
            # Puntos aleatorios para visualización
            w_random = np.random.dirichlet(np.ones(len(tickers)), 1000)
            r_random = (w_random @ mu.values)
            v_random = np.sqrt(np.diag(w_random @ S.values @ w_random.T))

            fig_frontier = go.Figure()

            # Nube de puntos (carteras aleatorias)
            fig_frontier.add_trace(go.Scatter(
                x=v_random, y=r_random,
                mode='markers', name='Carteras Aleatorias',
                marker=dict(color='lightgrey', size=4, opacity=0.5)
            ))

            # El punto optimizado
            fig_frontier.add_trace(go.Scatter(
                x=[performance[1]], y=[performance[0]],
                mode='markers', name='Cartera Optimizada',
                marker=dict(color='red', size=12, symbol='star')
            ))

            fig_frontier.update_layout(
                xaxis_title="Volatilidad (Riesgo)",
                yaxis_title="Rendimiento Esperado",
                template="plotly_white",
                height=500
            )
            st.plotly_chart(fig_frontier, use_container_width=True)

        except Exception as e:
            st.error(f"Error en la optimización: {e}")
            st.warning("Prueba con un objetivo de rendimiento más bajo o revisa los activos.")

except Exception as e:
    st.error(f"Error al obtener datos: {e}")

st.info("Nota: Este modelo asume posiciones largas únicamente (no ventas en corto).")