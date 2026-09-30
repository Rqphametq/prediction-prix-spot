import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import holidays
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pytz
from entsoe import EntsoePandasClient
import yfinance as yf

st.set_page_config(page_title="Prédiction Spot Électricité", page_icon="⚡", layout="wide")
st.title("⚡ Tableau de bord : Prix Spot de l'Électricité (France)")

with st.sidebar:
    st.header("⚙️ Configuration API")
    entsoe_token = st.text_input("Token ENTSO-E", type="password")
    if not entsoe_token:
        st.warning("Veuillez entrer votre Token ENTSO-E dans le menu de gauche.")
        st.stop()

@st.cache_resource
def load_model():
    return joblib.load('model_prix_spot_v10.pkl')

try:
    data_dict = load_model()
    model = data_dict['model']
    features = data_dict['features']
except FileNotFoundError:
    st.error("Le fichier 'model_prix_spot_v10.pkl' est introuvable.")
    st.stop()

@st.cache_data(ttl=3600)
def get_dashboard_data(token):
    tz = pytz.timezone('Europe/Paris')
    today = datetime.now(tz).date()
    start_date = today - timedelta(days=40)
    tomorrow = today + timedelta(days=1)
    
    url_price = f"https://api.energy-charts.info/price?bzn=FR&start={start_date}&end={tomorrow}"
    res_price = requests.get(url_price).json()
    df_price = pd.DataFrame({
        'time': pd.to_datetime(res_price['unix_seconds'], unit='s', utc=True),
        'price_eur_mwh': res_price['price']
    }).set_index('time')
    df_price = df_price.resample('1h').mean()

    clean_token = token.strip()
    client = EntsoePandasClient(api_key=clean_token)
    start_ts = pd.Timestamp(start_date, tz='Europe/Paris')
    end_ts = pd.Timestamp(tomorrow, tz='Europe/Paris') + pd.Timedelta(days=1)
    
    load_data = client.query_load_forecast('FR', start=start_ts, end=end_ts)
    if isinstance(load_data, pd.Series):
        df_load = load_data.to_frame(name='load_forecast_mwh')
    else:
        df_load = load_data.copy()
        df_load.columns = ['load_forecast_mwh']
        
    df_ws = client.query_wind_and_solar_forecast('FR', start=start_ts, end=end_ts)
    df_ws['wind_forecast_mwh'] = df_ws.filter(like='Wind').sum(axis=1)
    df_ws['solar_forecast_mwh'] = df_ws.filter(like='Solar').sum(axis=1)
    df_ws = df_ws[['wind_forecast_mwh', 'solar_forecast_mwh']]
    
    df_entsoe = df_load.join(df_ws, how='inner').resample('1h').mean()
    df_entsoe.index = df_entsoe.index.tz_convert('UTC')

    df = df_price.join(df_entsoe, how='right')

    gas_df = yf.Ticker("TTF=F").history(start=start_date.strftime('%Y-%m-%d'))
    if not gas_df.empty:
        gas = gas_df[['Close']].copy()
        gas.index = pd.to_datetime(gas.index, utc=True)
        gas = gas.resample('1h').ffill()
        gas.columns = ['gas_price']
        df = df.join(gas, how='left')
        df['gas_price'] = df['gas_price'].ffill().bfill()
    else:
        df['gas_price'] = 40.0
    
    df['net_demand_mwh'] = df['load_forecast_mwh'] - df['wind_forecast_mwh'] - df['solar_forecast_mwh']
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    fr_holidays = holidays.France()
    df['is_holiday'] = df.index.map(lambda x: int(x.date() in fr_holidays))
    df['non_working_day'] = np.maximum(df['is_weekend'], df['is_holiday'])

    df['hour_sin'] = np.sin(df['hour'] * (2. * np.pi / 24))
    df['hour_cos'] = np.cos(df['hour'] * (2. * np.pi / 24))
    df['day_sin'] = np.sin(df['dayofweek'] * (2. * np.pi / 7))
    df['day_cos'] = np.cos(df['dayofweek'] * (2. * np.pi / 7))

    df['price_lag_24h'] = df['price_eur_mwh'].shift(24)
    df['price_lag_168h'] = df['price_eur_mwh'].shift(168)
    df['price_rolling_7d'] = df['price_lag_24h'].rolling(window=168).mean()
    
    df.dropna(subset=features, inplace=True)
    return df

with st.spinner("Connexion aux serveurs..."):
    try:
        df = get_dashboard_data(entsoe_token)
    except Exception as e:
        st.error(f"Erreur : {e}")
        st.stop()

    X = df[features]
    
    # LA SÉCURITÉ EST ICI : Prédiction directe, avec un plancher strict à 0€
    raw_predictions = model.predict(X)
    df['prediction'] = np.maximum(raw_predictions, 0)
    
    df_eval = df.dropna(subset=['price_eur_mwh']).copy()
    df_future = df[df['price_eur_mwh'].isna()].copy()
    
    df_eval['erreur_absolue'] = np.abs(df_eval['price_eur_mwh'] - df_eval['prediction'])
    df_eval['erreur_relative'] = df_eval['prediction'] - df_eval['price_eur_mwh']

    display_start = pd.Timestamp.utcnow() - pd.Timedelta(days=30)
    df_eval = df_eval[df_eval.index >= display_start]
    
    mae = np.mean(df_eval['erreur_absolue']) if not df_eval.empty else 0

    tab1, tab2 = st.tabs(["📈 Projections & Suivi", "🧠 Analyse du Modèle"])

    with tab1:
        col1, col2, col3 = st.columns(3)
        col1.metric("Dernier Prix Réel", f"{df_eval['price_eur_mwh'].iloc[-1]:.2f} €/MWh" if not df_eval.empty else "N/A")
        col2.metric("Prédiction (Inconnue)", f"{df_future['prediction'].mean():.2f} €/MWh" if not df_future.empty else "En attente...")
        col3.metric("Erreur Moyenne (MAE 30j)", f"{mae:.2f} €/MWh")

        fig1 = go.Figure()
        if not df_eval.empty:
            fig1.add_trace(go.Scatter(x=df_eval.index, y=df_eval['price_eur_mwh'], mode='lines', name='Prix Réel', line=dict(color='#1f77b4')))
            fig1.add_trace(go.Scatter(x=df_eval.index, y=df_eval['prediction'], mode='lines', name='Prédiction', line=dict(color='#ff7f0e', dash='dash')))
        
        if not df_future.empty:
            future_plot = pd.concat([df_eval.iloc[[-1]] if not df_eval.empty else pd.DataFrame(), df_future])
            fig1.add_trace(go.Scatter(x=future_plot.index, y=future_plot['prediction'], mode='lines', name='Projection (J+1)', line=dict(color='#2ca02c', width=3)))

        fig1.add_vline(x=pd.Timestamp.now(tz='Europe/Paris'), line_width=2, line_dash="dot", line_color="gray", annotation_text="Maintenant")
        fig1.update_layout(title="Évolution du Prix Spot (V10 - Recalibré 6 mois)", hovermode="x unified", template="plotly_white")
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        df_importance = pd.DataFrame({'Variable': features, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=True)
        fig2 = px.bar(df_importance, x='Importance', y='Variable', orientation='h', title="Poids des Fondamentaux")
        st.plotly_chart(fig2, use_container_width=True)

        if not df_eval.empty:
            fig3 = px.bar(df_eval, x=df_eval.index, y='erreur_relative', color='erreur_relative', color_continuous_scale='RdBu_r', color_continuous_midpoint=0, title="Analyse des Résidus")
            st.plotly_chart(fig3, use_container_width=True)