"""
================================================================================
PAGINA 3: 🔮 PREVISIONI
================================================================================
Form interattivo per generare previsioni con il modello XGBoost.
Include confronto con ground truth se la data è nel dataset.

REQUISITO TRACCIA:
"integrare nella dashboard il modello costruito nel Project Work #2
in modo che l'utente possa fornire alcuni input tramite un semplice form"
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import json
from datetime import datetime, timedelta

st.set_page_config(page_title="Previsioni", page_icon="🔮", layout="wide")

# =============================================================================
# CARICAMENTO RISORSE
# =============================================================================
@st.cache_resource
def load_model():
    """Carica il modello XGBoost"""
    try:
        model = joblib.load('models/xgboost_model_final.pkl')
        return model
    except:
        st.warning("⚠️ Modello non trovato. Usando stima semplificata.")
        return None

@st.cache_data
def load_data():
    """Carica il dataset processato"""
    try:
        df = pd.read_csv('data/traffic_processed.csv', parse_dates=['date_time'])
        return df
    except:
        return None

@st.cache_data
def load_feature_columns():
    """Carica la lista delle feature"""
    try:
        with open('data/feature_columns.json', 'r') as f:
            return json.load(f)
    except:
        return None

# =============================================================================
# FUNZIONI HELPER
# =============================================================================
def get_historical_value(df, target_datetime):
    """Recupera il valore reale (ground truth) dal dataset"""
    if df is None:
        return None
    
    # Cerca corrispondenza esatta o entro 30 minuti
    mask = abs(df['date_time'] - target_datetime) < pd.Timedelta(minutes=30)
    if mask.any():
        return df.loc[mask, 'traffic_volume'].iloc[0]
    return None

def get_historical_lag(df, target_datetime, hours_back):
    """Recupera valore storico a N ore prima"""
    if df is None:
        return None
    
    lag_datetime = target_datetime - timedelta(hours=hours_back)
    mask = abs(df['date_time'] - lag_datetime) < pd.Timedelta(minutes=30)
    if mask.any():
        return df.loc[mask, 'traffic_volume'].iloc[0]
    return None

def create_features(dt, temp_celsius, weather_main, lag_1, lag_24, lag_168, 
                    rolling_mean_24h=None, rolling_std_24h=None):
    """Crea il DataFrame delle feature per la previsione"""
    
    hour = dt.hour
    dow = dt.dayofweek
    month = dt.month
    year = dt.year
    day_of_month = dt.day
    week_of_year = dt.isocalendar()[1]
    
    # Feature binarie
    is_weekend = 1 if dow >= 5 else 0
    is_holiday = 0  # Semplificazione
    is_workday = 1 if (dow < 5 and is_holiday == 0) else 0
    
    # Encoding ciclico
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    # Rush hour
    is_rush_hour_am = 1 if hour in [6, 7, 8, 9] else 0
    is_rush_hour_pm = 1 if hour in [16, 17, 18] else 0
    is_rush_hour = 1 if (is_rush_hour_am or is_rush_hour_pm) else 0
    is_night = 1 if hour in [22, 23, 0, 1, 2, 3, 4, 5] else 0
    is_work_rush = 1 if (is_rush_hour and is_workday) else 0
    
    # Lag features
    lag_2 = lag_1 * 0.98 if lag_1 else 3000
    lag_3 = lag_1 * 0.96 if lag_1 else 3000
    lag_48 = lag_24 if lag_24 else 3000
    
    # Differenze
    diff_lag_1_2 = (lag_1 - lag_2) if (lag_1 and lag_2) else 0
    diff_lag_1_24 = (lag_1 - lag_24) if (lag_1 and lag_24) else 0
    
    # Rolling (stima se non disponibile)
    if rolling_mean_24h is None:
        rolling_mean_24h = lag_1 if lag_1 else 3000
    if rolling_std_24h is None:
        rolling_std_24h = 500
    rolling_mean_168h = rolling_mean_24h
    rolling_min_24h = rolling_mean_24h * 0.5
    rolling_max_24h = rolling_mean_24h * 1.5
    rolling_range_24h = rolling_max_24h - rolling_min_24h
    
    # Meteo
    weather_types = ['Clear', 'Clouds', 'Drizzle', 'Fog', 'Haze', 'Mist', 
                     'Rain', 'Smoke', 'Snow', 'Squall', 'Thunderstorm']
    weather_features = {f'weather_{w}': (1 if weather_main == w else 0) for w in weather_types}
    
    is_bad_weather = 1 if weather_main in ['Snow', 'Thunderstorm', 'Fog'] else 0
    is_precipitation = 1 if weather_main in ['Rain', 'Drizzle', 'Snow'] else 0
    
    # Costruzione dizionario feature
    features = {
        'hour': hour,
        'day_of_week': dow,
        'month': month,
        'year': year,
        'day_of_month': day_of_month,
        'week_of_year': week_of_year,
        'is_weekend': is_weekend,
        'is_holiday': is_holiday,
        'is_workday': is_workday,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'dow_sin': dow_sin,
        'dow_cos': dow_cos,
        'month_sin': month_sin,
        'month_cos': month_cos,
        'is_rush_hour_am': is_rush_hour_am,
        'is_rush_hour_pm': is_rush_hour_pm,
        'is_rush_hour': is_rush_hour,
        'is_night': is_night,
        'is_work_rush': is_work_rush,
        'temp_celsius': temp_celsius,
        'rain_1h_capped': 0,
        'is_bad_weather': is_bad_weather,
        'is_precipitation': is_precipitation,
        'lag_1': lag_1 if lag_1 else 3000,
        'lag_2': lag_2,
        'lag_3': lag_3,
        'lag_24': lag_24 if lag_24 else 3000,
        'lag_48': lag_48,
        'lag_168': lag_168 if lag_168 else 3000,
        'diff_lag_1_2': diff_lag_1_2,
        'diff_lag_1_24': diff_lag_1_24,
        'rolling_mean_24h': rolling_mean_24h,
        'rolling_std_24h': rolling_std_24h,
        'rolling_mean_168h': rolling_mean_168h,
        'rolling_min_24h': rolling_min_24h,
        'rolling_max_24h': rolling_max_24h,
        'rolling_range_24h': rolling_range_24h,
        'time_diff_hours': 1,
        'is_after_gap': 0,
    }
    
    features.update(weather_features)
    
    return pd.DataFrame([features])

def estimate_traffic_simple(hour, is_weekend, temp):
    """Stima semplificata se il modello non è disponibile"""
    base = 3000
    
    # Pattern orario
    if 7 <= hour <= 9:
        base += 2000
    elif 16 <= hour <= 18:
        base += 2500
    elif 0 <= hour <= 5:
        base -= 2000
    
    # Weekend
    if is_weekend:
        base -= 500
    
    # Temperatura
    if temp < -10:
        base -= 300
    elif temp > 25:
        base += 200
    
    return max(0, base + np.random.normal(0, 200))

# =============================================================================
# INTERFACCIA
# =============================================================================
st.title("🔮 Previsioni Traffico")
st.markdown("Genera previsioni utilizzando il modello XGBoost addestrato")

# Caricamento risorse
model = load_model()
df = load_data()
feature_columns = load_feature_columns()

# Verifica disponibilità dati
if df is not None:
    min_date = df['date_time'].min().date()
    max_date = df['date_time'].max().date()
    st.info(f"📅 Dati disponibili: {min_date} → {max_date}. Seleziona una data in questo range per vedere il confronto con il valore reale.")
else:
    min_date = datetime(2012, 1, 1).date()
    max_date = datetime(2018, 12, 31).date()

st.markdown("---")

# =============================================================================
# FORM INPUT
# =============================================================================
st.subheader("📝 Parametri di Input")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📅 Data e Ora**")
    selected_date = st.date_input(
        "Data",
        value=datetime(2018, 6, 15).date(),
        min_value=min_date,
        max_value=max_date
    )
    selected_hour = st.selectbox(
        "Ora",
        options=list(range(24)),
        index=12,
        format_func=lambda x: f"{x:02d}:00"
    )
    is_holiday = st.checkbox("Giorno festivo", value=False)

with col2:
    st.markdown("**🌤️ Condizioni Meteo**")
    weather_options = ['Clear', 'Clouds', 'Rain', 'Snow', 'Fog', 'Mist', 'Drizzle', 'Thunderstorm']
    weather = st.selectbox("Condizione meteo", options=weather_options, index=0)
    temperature = st.slider("Temperatura (°C)", min_value=-30, max_value=40, value=20)

with col3:
    st.markdown("**📊 Valori Storici**")
    use_auto_lag = st.checkbox("Usa valori automatici dal dataset", value=True)
    
    if not use_auto_lag:
        manual_lag_1 = st.number_input("Traffico 1 ora fa", min_value=0, max_value=8000, value=3000)
        manual_lag_24 = st.number_input("Traffico 24 ore fa", min_value=0, max_value=8000, value=3000)
    else:
        manual_lag_1 = None
        manual_lag_24 = None

# =============================================================================
# CALCOLO PREVISIONE
# =============================================================================
if st.button("🚀 Genera Previsione", type="primary", use_container_width=True):
    
    # Costruisci datetime
    target_dt = pd.Timestamp(datetime.combine(selected_date, datetime.min.time()) + timedelta(hours=selected_hour))
    
    # Recupera valori storici
    if use_auto_lag and df is not None:
        lag_1 = get_historical_lag(df, target_dt, 1)
        lag_24 = get_historical_lag(df, target_dt, 24)
        lag_168 = get_historical_lag(df, target_dt, 168)
        
        if lag_1 is None:
            st.warning("⚠️ Dati storici non disponibili per questa data. Uso valori di default.")
            lag_1, lag_24, lag_168 = 3000, 3000, 3000
    else:
        lag_1 = manual_lag_1 if manual_lag_1 else 3000
        lag_24 = manual_lag_24 if manual_lag_24 else 3000
        lag_168 = 3000
    
    # Genera previsione
    if model is not None and feature_columns is not None:
        # Crea feature
        X = create_features(
            dt=target_dt,
            temp_celsius=temperature,
            weather_main=weather,
            lag_1=lag_1,
            lag_24=lag_24,
            lag_168=lag_168
        )
        
        # Allinea colonne
        for col in feature_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_columns]
        
        # Previsione
        prediction = model.predict(X)[0]
        prediction = max(0, prediction)
    else:
        # Stima semplificata
        is_weekend = 1 if target_dt.dayofweek >= 5 else 0
        prediction = estimate_traffic_simple(selected_hour, is_weekend, temperature)
    
    # Recupera ground truth (se disponibile)
    ground_truth = get_historical_value(df, target_dt)
    
    st.markdown("---")
    
    # =============================================================================
    # VISUALIZZAZIONE RISULTATI
    # =============================================================================
    st.subheader("📊 Risultato Previsione")
    
    # Determina livello traffico
    if prediction < 1500:
        level = "🟢 BASSO"
        level_color = "green"
    elif prediction < 3500:
        level = "🟡 MEDIO"
        level_color = "orange"
    elif prediction < 5500:
        level = "🟠 ALTO"
        level_color = "darkorange"
    else:
        level = "🔴 MOLTO ALTO"
        level_color = "red"
    
    # Layout risultati
    if ground_truth is not None:
        # CON CONFRONTO GROUND TRUTH
        col1, col2, col3, col4 = st.columns(4)
        
        error = prediction - ground_truth
        error_pct = abs(error) / ground_truth * 100 if ground_truth > 0 else 0
        
        with col1:
            st.metric(
                label="🔮 Previsione",
                value=f"{prediction:,.0f}",
                help="Veicoli/ora previsti dal modello"
            )
        
        with col2:
            st.metric(
                label="✅ Valore Reale",
                value=f"{ground_truth:,.0f}",
                help="Ground truth dal dataset"
            )
        
        with col3:
            st.metric(
                label="📏 Errore",
                value=f"{error:+,.0f}",
                delta=f"{error_pct:.1f}%",
                delta_color="inverse"
            )
        
        with col4:
            st.metric(
                label="🚦 Livello",
                value=level.split()[1],
                help="Livello di traffico"
            )
        
        # Valutazione errore
        st.markdown("---")
        if error_pct < 5:
            st.success(f"✅ **Previsione eccellente!** Errore del {error_pct:.1f}% (< 5%)")
        elif error_pct < 10:
            st.success(f"✅ **Buona previsione!** Errore del {error_pct:.1f}% (< 10%)")
        elif error_pct < 20:
            st.warning(f"⚠️ **Previsione accettabile.** Errore del {error_pct:.1f}%")
        else:
            st.error(f"❌ **Previsione imprecisa.** Errore del {error_pct:.1f}% - Possibili cause: evento speciale, dati anomali")
    
    else:
        # SENZA GROUND TRUTH (data futura o non disponibile)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="🔮 Previsione",
                value=f"{prediction:,.0f}",
                help="Veicoli/ora previsti"
            )
        
        with col2:
            # Calcola valore tipico per quell'ora
            if df is not None:
                typical = df[df['date_time'].dt.hour == selected_hour]['traffic_volume'].mean()
            else:
                typical = 3300
            
            delta = prediction - typical
            st.metric(
                label="📈 Tipico per quest'ora",
                value=f"{typical:,.0f}",
                delta=f"{delta:+,.0f} vs previsione",
                delta_color="off"
            )
        
        with col3:
            st.metric(
                label="🚦 Livello Traffico",
                value=level.split()[1],
                help="Livello di traffico previsto"
            )
        
        st.info("ℹ️ **Nota:** Per questa data non è disponibile il valore reale nel dataset. La previsione non può essere verificata.")
    
    # =============================================================================
    # GRAFICO CONTESTUALE
    # =============================================================================
    st.markdown("---")
    st.subheader("📈 Contesto: Pattern Giornaliero")
    
    if df is not None:
        # Pattern medio per ora
        is_weekend_selected = 1 if target_dt.dayofweek >= 5 else 0
        
        if is_weekend_selected:
            hourly_pattern = df[df['date_time'].dt.dayofweek >= 5].groupby(
                df['date_time'].dt.hour
            )['traffic_volume'].mean()
            pattern_label = "Weekend"
        else:
            hourly_pattern = df[df['date_time'].dt.dayofweek < 5].groupby(
                df['date_time'].dt.hour
            )['traffic_volume'].mean()
            pattern_label = "Feriale"
        
        fig = go.Figure()
        
        # Pattern tipico
        fig.add_trace(go.Scatter(
            x=list(range(24)),
            y=hourly_pattern.values,
            mode='lines+markers',
            name=f'Pattern tipico ({pattern_label})',
            line=dict(color='steelblue', width=2),
            fill='tozeroy',
            fillcolor='rgba(70, 130, 180, 0.2)'
        ))
        
        # Punto previsione
        fig.add_trace(go.Scatter(
            x=[selected_hour],
            y=[prediction],
            mode='markers',
            name='Previsione',
            marker=dict(color='red', size=15, symbol='star')
        ))
        
        # Punto ground truth (se disponibile)
        if ground_truth is not None:
            fig.add_trace(go.Scatter(
                x=[selected_hour],
                y=[ground_truth],
                mode='markers',
                name='Valore Reale',
                marker=dict(color='green', size=12, symbol='circle')
            ))
        
        fig.update_layout(
            xaxis_title='Ora del giorno',
            yaxis_title='Traffico (veicoli/ora)',
            hovermode='x unified',
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # =============================================================================
    # DETTAGLI TECNICI
    # =============================================================================
    with st.expander("🔧 Dettagli Tecnici"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Input utilizzati:**")
            st.write(f"- Data/Ora: {target_dt.strftime('%Y-%m-%d %H:%M')}")
            st.write(f"- Meteo: {weather}")
            st.write(f"- Temperatura: {temperature}°C")
            st.write(f"- Festivo: {'Sì' if is_holiday else 'No'}")
        
        with col2:
            st.markdown("**Valori lag:**")
            st.write(f"- lag_1 (1h fa): {lag_1:,.0f}" if lag_1 else "- lag_1: N/A")
            st.write(f"- lag_24 (24h fa): {lag_24:,.0f}" if lag_24 else "- lag_24: N/A")
            st.write(f"- lag_168 (1 sett. fa): {lag_168:,.0f}" if lag_168 else "- lag_168: N/A")
        
        if ground_truth is not None:
            st.markdown("---")
            st.markdown("**Valutazione:**")
            st.write(f"- Previsione: {prediction:,.0f}")
            st.write(f"- Ground Truth: {ground_truth:,.0f}")
            st.write(f"- Errore Assoluto: {abs(error):,.0f}")
            st.write(f"- Errore Percentuale: {error_pct:.2f}%")

# =============================================================================
# INFO SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("### ℹ️ Informazioni")
    st.markdown("""
    **Come funziona:**
    1. Seleziona data e ora
    2. Imposta condizioni meteo
    3. Clicca "Genera Previsione"
    
    **Confronto Ground Truth:**
    - Se la data è nel dataset (2012-2018), viene mostrato anche il valore reale
    - Questo permette di verificare l'accuratezza del modello
    
    **Feature principali:**
    - `lag_1`: traffico 1 ora prima
    - `hour_cos`: encoding ciclico ora
    - `is_weekend`: flag weekend
    """)
    
    if df is not None:
        st.markdown("---")
        st.markdown("### 📊 Dataset")
        st.write(f"- Osservazioni: {len(df):,}")
        st.write(f"- Periodo: {df['date_time'].min().date()} → {df['date_time'].max().date()}")