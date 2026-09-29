import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Configuration de la page
st.set_page_config(page_title="Prédiction Spot Électricité", page_icon="⚡", layout="wide")
st.title("⚡ Tableau de bord : Prix Spot de l'Électricité (France)")
st.markdown("Comparaison entre le prix réel Day-Ahead et la prédiction du modèle Machine Learning (XGBoost).")

# 1. Chargement du modèle
@st.cache_resource
def load_model():
    return joblib.load('model_prix_spot.pkl')

data_dict = load_model()
model = data_dict['model']
features = data_dict['features']

# 2. Fonction pour récupérer et préparer les données récentes (30 derniers jours)
@st.cache_data(ttl=3600) # Mise en cache d'une heure
def get_recent_data():
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=30)
    
    # Prix Spot (Energy-Charts)
    url_price = f"https://api.energy-charts.info/price?bzn=FR&start={start_date}&end={end_date}"
    res_price = requests.get(url_price).json()
    df_price = pd.DataFrame({
        'time': pd.to_datetime(res_price['unix_seconds'], unit='s', utc=True),
        'price_eur_mwh': res_price['price']
    }).set_index('time')

    # Météo (Open-Meteo Forecast API avec past_days)
    url_weather = "https://api.open-meteo.com/v1/forecast"
    params_weather = {
        "latitude": 46.22, "longitude": 2.21,
        "past_days": 30, "forecast_days": 1,
        "hourly": "temperature_2m,wind_speed_10m,direct_radiation"
    }
    res_weather = requests.get(url_weather, params=params_weather).json()
    df_weather = pd.DataFrame(res_weather['hourly'])
    df_weather['time'] = pd.to_datetime(df_weather['time'], utc=True)
    df_weather.set_index('time', inplace=True)

    # Fusion
    df = df_price.join(df_weather, how='inner')
    
    # Feature Engineering
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['price_lag_24h'] = df['price_eur_mwh'].shift(24)
    df['price_lag_168h'] = df['price_eur_mwh'].shift(168)
    df['price_rolling_7d'] = df['price_lag_24h'].rolling(window=168).mean()
    
    return df.dropna()

# 3. Exécution et Affichage
with st.spinner("Téléchargement des données récentes et calcul des prédictions..."):
    df = get_recent_data()
    
    # Prédictions
    X = df[features]
    df['prediction'] = model.predict(X)

    # Calcul des métriques sur cette période
    mae = np.mean(np.abs(df['price_eur_mwh'] - df['prediction']))
    
    # Affichage des KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Dernier Prix Réel", f"{df['price_eur_mwh'].iloc[-1]:.2f} €/MWh")
    col2.metric("Dernière Prédiction", f"{df['prediction'].iloc[-1]:.2f} €/MWh")
    col3.metric("Erreur Moyenne (MAE sur 30j)", f"{mae:.2f} €/MWh")

    # Graphique interactif Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['price_eur_mwh'], mode='lines', name='Prix Réel', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['prediction'], mode='lines', name='Prédiction XGBoost', line=dict(color='red', dash='dash')))
    
    fig.update_layout(
        title="Évolution du Prix Spot vs Prédiction (Derniers jours)",
        xaxis_title="Date",
        yaxis_title="Prix (€/MWh)",
        hovermode="x unified",
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Tableau de données brut (optionnel)
    with st.expander("Voir les données brutes"):
        st.dataframe(df[['price_eur_mwh', 'prediction'] + features].tail(24))