"""
================================================================================
PAGINA 3: 🔮 PREVISIONI - VERSIONE COMPLETA
================================================================================
Form interattivo per generare previsioni con il modello XGBoost.

FUNZIONALITÀ:
- Modalità STORICA: usa lag reali dal dataset (2012-2018) + confronto ground truth
- Modalità FUTURA: usa rolling forecast per date oltre il dataset (>2018)

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
from pathlib import Path

st.set_page_config(page_title="Previsioni", page_icon="🔮", layout="wide")

# =============================================================================
# CARICAMENTO RISORSE
# =============================================================================
@st.cache_resource
def load_model():
    """Carica il modello XGBoost addestrato"""
    try:
        model_path = Path(__file__).parent.parent / 'models' / 'xgboost_model_final.pkl'
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.warning(f"⚠️ Modello non trovato: {e}")
        return None

@st.cache_data
def load_data():
    """Carica il dataset processato con tutte le feature"""
    try:
        data_path = Path(__file__).parent.parent / 'data' / 'traffic_processed.csv'
        df = pd.read_csv(data_path, parse_dates=['date_time'])
        return df
    except Exception as e:
        st.error(f"⚠️ Impossibile caricare i dati: {e}")
        return None

@st.cache_data
def load_feature_columns():
    """Carica la lista delle feature utilizzate dal modello"""
    try:
        json_path = Path(__file__).parent.parent / 'data' / 'feature_columns.json'
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"⚠️ Feature columns non trovate: {e}")
        return None

# =============================================================================
# FUNZIONI HELPER - RECUPERO DATI DAL DATASET
# =============================================================================
def get_historical_value(df, target_datetime):
    """
    Recupera il valore reale (ground truth) dal dataset.
    Cerca corrispondenze entro 30 minuti dal timestamp target.
    """
    if df is None:
        return None
    
    mask = abs(df['date_time'] - target_datetime) < pd.Timedelta(minutes=30)
    if mask.any():
        return df.loc[mask, 'traffic_volume'].iloc[0]
    return None

def get_all_lags_from_dataset(df, target_datetime):
    """
    Recupera TUTTI i valori lag dal dataset (se esistono).
    
    Returns:
        dict: {'lag_1': value, 'lag_2': value, ...} con None se non trovati
    """
    if df is None:
        return {}
    
    lag_configs = {
        'lag_1': 1,      # 1 ora fa
        'lag_2': 2,      # 2 ore fa
        'lag_3': 3,      # 3 ore fa
        'lag_24': 24,    # 24 ore fa (ieri stessa ora)
        'lag_48': 48,    # 48 ore fa (2 giorni fa)
        'lag_168': 168   # 168 ore fa (1 settimana fa)
    }
    
    lags = {}
    for lag_name, hours_back in lag_configs.items():
        lag_datetime = target_datetime - timedelta(hours=hours_back)
        mask = abs(df['date_time'] - lag_datetime) < pd.Timedelta(minutes=30)
        
        if mask.any():
            lags[lag_name] = df.loc[mask, 'traffic_volume'].iloc[0]
        else:
            lags[lag_name] = None
    
    return lags

def get_rolling_features(df, target_datetime):
    """
    Recupera le rolling features dal dataset (se disponibili).
    Queste sono pre-calcolate nel CSV e includono medie/std mobili.
    """
    if df is None:
        return {}
    
    mask = abs(df['date_time'] - target_datetime) < pd.Timedelta(minutes=30)
    if not mask.any():
        return {}
    
    row = df.loc[mask].iloc[0]
    
    return {
        'rolling_mean_24h': row.get('rolling_mean_24h'),
        'rolling_std_24h': row.get('rolling_std_24h'),
        'rolling_mean_168h': row.get('rolling_mean_168h'),
        'rolling_min_24h': row.get('rolling_min_24h'),
        'rolling_max_24h': row.get('rolling_max_24h'),
        'rolling_range_24h': row.get('rolling_range_24h')
    }

# =============================================================================
# ROLLING FORECAST - PER DATE FUTURE
# =============================================================================
def predict_with_rolling_forecast(model, feature_columns, df, target_datetime, 
                                   temperature, weather_main):
    """
    PREDIZIONE ITERATIVA per date oltre il dataset.
    
    Strategia:
    1. Parte dall'ultimo timestamp disponibile nel dataset
    2. Predice ora per ora fino alla data target
    3. Usa le predizioni precedenti come lag per le successive
    
    Questo è il metodo standard per forecasting multi-step in produzione
    quando non hai ancora i valori reali.
    
    Args:
        model: modello XGBoost addestrato
        feature_columns: lista delle colonne feature
        df: dataset con dati storici
        target_datetime: data/ora da predire (futura)
        temperature: temperatura prevista per il target
        weather_main: condizione meteo prevista per il target
    
    Returns:
        tuple: (prediction_finale, lags_utilizzati, num_iterazioni)
    """
    
    # 1. Verifica se è davvero una data futura
    last_datetime = df['date_time'].max()
    
    if target_datetime <= last_datetime:
        # Non è una data futura, ritorna None
        return None, None, 0
    
    # 2. Calcola quante ore dobbiamo predire
    hours_to_predict = int((target_datetime - last_datetime).total_seconds() / 3600)
    
    st.info(f"🔄 **Rolling Forecast attivo**: predirò {hours_to_predict} ore dal {last_datetime.strftime('%Y-%m-%d %H:%M')} al target")
    
    # 3. Costruisci storia iniziale: ultimi 168 valori (1 settimana) dal dataset
    history = []
    for hours_back in range(168, 0, -1):
        past_dt = last_datetime - timedelta(hours=hours_back)
        val = get_historical_value(df, past_dt)
        history.append(val if val is not None else 3000)  # default se mancante
    
    # 4. Predici iterativamente ora per ora
    current_datetime = last_datetime
    predictions_made = []
    
    # Mostra progress bar
    with st.expander(f"📊 Dettagli Rolling Forecast (mostrando prime 20 iterazioni)", expanded=False):
        progress_bar = st.progress(0)
        detail_text = st.empty()
        
        for i in range(hours_to_predict):
            current_datetime += timedelta(hours=1)
            
            # Costruisci lag dalla storia accumulata
            lags = {
                'lag_1': history[-1],
                'lag_2': history[-2] if len(history) >= 2 else history[-1],
                'lag_3': history[-3] if len(history) >= 3 else history[-1],
                'lag_24': history[-24] if len(history) >= 24 else 3000,
                'lag_48': history[-48] if len(history) >= 48 else 3000,
                'lag_168': history[-168] if len(history) >= 168 else 3000
            }
            
            # Determina meteo: usa input utente solo per l'ultima ora, medio per le altre
            is_last_hour = (i == hours_to_predict - 1)
            temp_current = temperature if is_last_hour else 15
            weather_current = weather_main if is_last_hour else 'Clear'
            
            # Crea features
            X = create_features(
                dt=current_datetime,
                temp_celsius=temp_current,
                weather_main=weather_current,
                lags=lags,
                rolling=None  # le rolling vengono stimate dalla funzione
            )
            
            # Allinea colonne al modello
            for col in feature_columns:
                if col not in X.columns:
                    X[col] = 0
            X = X[feature_columns]
            
            # Esegui predizione
            pred = model.predict(X)[0]
            pred = max(0, pred)  # no valori negativi
            
            # Aggiungi alla storia
            history.append(pred)
            predictions_made.append(pred)
            
            # Mostra dettagli solo per le prime 20 iterazioni
            if i < 20:
                detail_text.text(f"Iterazione {i+1}/{hours_to_predict}: {current_datetime.strftime('%Y-%m-%d %H:%M')} → Predizione: {pred:,.0f} veicoli/h (lag_1={lags['lag_1']:,.0f})")
            elif i == 20:
                detail_text.text(f"... (altre {hours_to_predict - 20} iterazioni in corso) ...")
            
            # Aggiorna progress
            progress_bar.progress((i + 1) / hours_to_predict)
    
    # 5. Ritorna il risultato finale
    final_prediction = predictions_made[-1]
    
    final_lags = {
        'lag_1': history[-1],
        'lag_2': history[-2],
        'lag_3': history[-3],
        'lag_24': history[-24],
        'lag_48': history[-48],
        'lag_168': history[-168]
    }
    
    return final_prediction, final_lags, hours_to_predict

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def create_features(dt, temp_celsius, weather_main, lags=None, rolling=None):
    """
    Crea il DataFrame delle feature per la predizione.
    
    Questa funzione replica esattamente la feature engineering usata in training.
    
    Args:
        dt: datetime target
        temp_celsius: temperatura in Celsius
        weather_main: condizione meteo principale
        lags: dict con valori lag (se None, usa default)
        rolling: dict con rolling features (se None, stima da lag)
    
    Returns:
        pd.DataFrame: una riga con tutte le feature
    """
    
    # ========== FEATURE TEMPORALI BASE ==========
    hour = dt.hour
    dow = dt.dayofweek  # 0=Monday, 6=Sunday
    month = dt.month
    year = dt.year
    day_of_month = dt.day
    week_of_year = dt.isocalendar()[1]
    
    # ========== FEATURE BINARIE ==========
    is_weekend = 1 if dow >= 5 else 0
    is_holiday = 0  # Semplificazione (in produzione si userebbe calendario festività)
    is_workday = 1 if (dow < 5 and is_holiday == 0) else 0
    
    # ========== ENCODING CICLICO (per catturare periodicità) ==========
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    # ========== RUSH HOUR FEATURES ==========
    is_rush_hour_am = 1 if hour in [6, 7, 8, 9] else 0
    is_rush_hour_pm = 1 if hour in [16, 17, 18] else 0
    is_rush_hour = 1 if (is_rush_hour_am or is_rush_hour_pm) else 0
    is_night = 1 if hour in [22, 23, 0, 1, 2, 3, 4, 5] else 0
    is_work_rush = 1 if (is_rush_hour and is_workday) else 0
    
    # ========== LAG FEATURES ==========
    if lags is None:
        lags = {}
    
    # Usa valori dal dict o default 3000
    lag_1 = lags.get('lag_1', 3000)
    lag_2 = lags.get('lag_2', 3000)
    lag_3 = lags.get('lag_3', 3000)
    lag_24 = lags.get('lag_24', 3000)
    lag_48 = lags.get('lag_48', 3000)
    lag_168 = lags.get('lag_168', 3000)
    
    # Differenze tra lag (catturano trend)
    diff_lag_1_2 = (lag_1 - lag_2) if (lag_1 and lag_2) else 0
    diff_lag_1_24 = (lag_1 - lag_24) if (lag_1 and lag_24) else 0
    
    # ========== ROLLING FEATURES ==========
    if rolling is None:
        rolling = {}
    
    # Usa valori reali se disponibili, altrimenti stima da lag
    rolling_mean_24h = rolling.get('rolling_mean_24h', lag_1 if lag_1 else 3000)
    rolling_std_24h = rolling.get('rolling_std_24h', 500)
    rolling_mean_168h = rolling.get('rolling_mean_168h', rolling_mean_24h)
    rolling_min_24h = rolling.get('rolling_min_24h', rolling_mean_24h * 0.5)
    rolling_max_24h = rolling.get('rolling_max_24h', rolling_mean_24h * 1.5)
    rolling_range_24h = rolling.get('rolling_range_24h', rolling_max_24h - rolling_min_24h)
    
    # ========== WEATHER FEATURES (one-hot encoding) ==========
    weather_types = ['Clear', 'Clouds', 'Drizzle', 'Fog', 'Haze', 'Mist', 
                     'Rain', 'Smoke', 'Snow', 'Squall', 'Thunderstorm']
    weather_features = {
        f'weather_{w}': (1 if weather_main == w else 0) for w in weather_types
    }
    
    # Meteo aggregato
    is_bad_weather = 1 if weather_main in ['Snow', 'Thunderstorm', 'Fog'] else 0
    is_precipitation = 1 if weather_main in ['Rain', 'Drizzle', 'Snow'] else 0
    
    # ========== COSTRUZIONE DIZIONARIO FINALE ==========
    features = {
        # Temporali base
        'hour': hour,
        'day_of_week': dow,
        'month': month,
        'year': year,
        'day_of_month': day_of_month,
        'week_of_year': week_of_year,
        
        # Binarie
        'is_weekend': is_weekend,
        'is_holiday': is_holiday,
        'is_workday': is_workday,
        
        # Cicliche
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'dow_sin': dow_sin,
        'dow_cos': dow_cos,
        'month_sin': month_sin,
        'month_cos': month_cos,
        
        # Rush hour
        'is_rush_hour_am': is_rush_hour_am,
        'is_rush_hour_pm': is_rush_hour_pm,
        'is_rush_hour': is_rush_hour,
        'is_night': is_night,
        'is_work_rush': is_work_rush,
        
        # Meteo
        'temp_celsius': temp_celsius,
        'rain_1h_capped': 0,  # Non usato in input utente
        'is_bad_weather': is_bad_weather,
        'is_precipitation': is_precipitation,
        
        # Lag
        'lag_1': lag_1,
        'lag_2': lag_2,
        'lag_3': lag_3,
        'lag_24': lag_24,
        'lag_48': lag_48,
        'lag_168': lag_168,
        'diff_lag_1_2': diff_lag_1_2,
        'diff_lag_1_24': diff_lag_1_24,
        
        # Rolling
        'rolling_mean_24h': rolling_mean_24h,
        'rolling_std_24h': rolling_std_24h,
        'rolling_mean_168h': rolling_mean_168h,
        'rolling_min_24h': rolling_min_24h,
        'rolling_max_24h': rolling_max_24h,
        'rolling_range_24h': rolling_range_24h,
        
        # Altre
        'time_diff_hours': 1,
        'is_after_gap': 0,
    }
    
    # Aggiungi one-hot weather
    features.update(weather_features)
    
    return pd.DataFrame([features])

# =============================================================================
# INTERFACCIA UTENTE
# =============================================================================
st.title("🔮 Previsioni Traffico")
st.markdown("""
Genera previsioni utilizzando il modello XGBoost addestrato.
Supporta sia **predizioni storiche** (con confronto ground truth) che **predizioni future** (con rolling forecast).
""")

# Caricamento risorse
model = load_model()
df = load_data()
feature_columns = load_feature_columns()

# Info dataset
if df is not None:
    min_date = df['date_time'].min().date()
    max_date = df['date_time'].max().date()
    st.success(f"✅ **Dataset caricato**: {len(df):,} osservazioni dal {min_date} al {max_date}")
else:
    st.error("❌ **Dataset non disponibile**")
    min_date = datetime(2012, 1, 1).date()
    max_date = datetime(2025, 12, 31).date()

if model is None:
    st.error("❌ **Modello non disponibile**")
    st.stop()

st.markdown("---")

# =============================================================================
# FORM INPUT
# =============================================================================
st.subheader("📝 Parametri di Input")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**📅 Data e Ora**")
    
    # Scelta modalità
    mode = st.radio(
        "Modalità predizione",
        options=["📊 Storica (2012-2018)", "🚀 Futura (>2018)"],
        help="Storica: usa lag reali dal dataset. Futura: usa rolling forecast."
    )
    
    # Input data in base alla modalità
    if "Storica" in mode:
        selected_date = st.date_input(
            "Data",
            value=datetime(2018, 6, 15).date(),
            min_value=min_date,
            max_value=max_date,
            help="Seleziona una data nel range del dataset"
        )
    else:
        selected_date = st.date_input(
            "Data",
            value=datetime(2020, 6, 15).date(),
            min_value=max_date + timedelta(days=1),
            max_value=datetime(2025, 12, 31).date(),
            help="Seleziona una data futura (oltre il dataset)"
        )
    
    selected_hour = st.selectbox(
        "Ora",
        options=list(range(24)),
        index=12,
        format_func=lambda x: f"{x:02d}:00"
    )

with col2:
    st.markdown("**🌤️ Condizioni Meteo**")
    
    weather_options = ['Clear', 'Clouds', 'Rain', 'Snow', 'Fog', 'Mist', 
                       'Drizzle', 'Thunderstorm']
    weather = st.selectbox(
        "Condizione meteo",
        options=weather_options,
        index=0,
        help="Condizione meteo prevista"
    )
    
    temperature = st.slider(
        "Temperatura (°C)",
        min_value=-30,
        max_value=40,
        value=20,
        help="Temperatura in gradi Celsius"
    )

# =============================================================================
# CALCOLO PREVISIONE
# =============================================================================
if st.button("🚀 Genera Previsione", type="primary", use_container_width=True):
    
    # Costruisci datetime target
    target_dt = pd.Timestamp(
        datetime.combine(selected_date, datetime.min.time()) + timedelta(hours=selected_hour)
    )
    
    st.markdown("---")
    
    # Determina modalità: storica o futura
    is_historical = (df is not None) and (target_dt <= df['date_time'].max())
    
    # ========== MODALITÀ STORICA ==========
    if is_historical:
        st.info("📊 **Modalità Storica Attiva**: Utilizzo lag reali dal dataset")
        
        # Recupera lag e rolling dal dataset
        lags = get_all_lags_from_dataset(df, target_dt)
        rolling = get_rolling_features(df, target_dt)
        
        # Mostra lag recuperati
        with st.expander("🔍 Valori Lag Recuperati dal Dataset", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                for lag_name in ['lag_1', 'lag_2', 'lag_3']:
                    val = lags.get(lag_name)
                    if val is not None:
                        st.success(f"✅ **{lag_name}**: {val:,.0f} veicoli/h")
                    else:
                        st.warning(f"⚠️ **{lag_name}**: Non disponibile (uso default)")
            
            with col2:
                for lag_name in ['lag_24', 'lag_48', 'lag_168']:
                    val = lags.get(lag_name)
                    if val is not None:
                        st.success(f"✅ **{lag_name}**: {val:,.0f} veicoli/h")
                    else:
                        st.warning(f"⚠️ **{lag_name}**: Non disponibile (uso default)")
        
        # Crea features
        X = create_features(
            dt=target_dt,
            temp_celsius=temperature,
            weather_main=weather,
            lags=lags,
            rolling=rolling
        )
        
        # Allinea colonne
        for col in feature_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_columns]
        
        # Predizione
        prediction = model.predict(X)[0]
        prediction = max(0, prediction)
        
        # Recupera ground truth
        ground_truth = get_historical_value(df, target_dt)
        num_iterations = 0
    
    # ========== MODALITÀ FUTURA ==========
    else:
        st.warning("🚀 **Modalità Futura Attiva**: Utilizzo rolling forecast (predizioni iterative)")
        st.markdown("""
        ℹ️ Il modello predirà ora per ora dall'ultimo valore disponibile (2018) fino al target.
        Questo processo può richiedere alcuni secondi per date molto lontane.
        """)
        
        # Rolling forecast
        prediction, lags, num_iterations = predict_with_rolling_forecast(
            model=model,
            feature_columns=feature_columns,
            df=df,
            target_datetime=target_dt,
            temperature=temperature,
            weather_main=weather
        )
        
        ground_truth = None  # Non disponibile per date future
    
    st.markdown("---")
    
    # =============================================================================
    # VISUALIZZAZIONE RISULTATI
    # =============================================================================
    st.subheader("📊 Risultato Previsione")
    
    # Determina livello traffico
    if prediction < 1500:
        level = "🟢 BASSO"
        level_desc = "Traffico scorrevole"
    elif prediction < 3500:
        level = "🟡 MEDIO"
        level_desc = "Traffico moderato"
    elif prediction < 5500:
        level = "🟠 ALTO"
        level_desc = "Traffico intenso"
    else:
        level = "🔴 MOLTO ALTO"
        level_desc = "Traffico congestionato"
    
    # Layout risultati in base a disponibilità ground truth
    if ground_truth is not None:
        # ===== CON GROUND TRUTH (modalità storica) =====
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
                label="📉 Errore",
                value=f"{error:+,.0f}",
                delta=f"{error_pct:.1f}%",
                delta_color="inverse",
                help="Differenza tra previsione e realtà"
            )
        
        with col4:
            st.metric(
                label="🚦 Livello",
                value=level.split()[1],
                help=level_desc
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
            st.error(f"❌ **Previsione imprecisa.** Errore del {error_pct:.1f}% - Possibili cause: evento speciale, anomalia nei dati")
    
    else:
        # ===== SENZA GROUND TRUTH (modalità futura) =====
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="🔮 Previsione",
                value=f"{prediction:,.0f}",
                help="Veicoli/ora previsti"
            )
        
        with col2:
            st.metric(
                label="🔄 Iterazioni",
                value=f"{num_iterations}",
                help="Numero di predizioni intermedie effettuate"
            )
        
        with col3:
            st.metric(
                label="🚦 Livello",
                value=level.split()[1],
                help=level_desc
            )
        
        st.info("""
        ℹ️ **Nota**: Per questa data non è disponibile il valore reale nel dataset.
        La previsione è stata generata tramite rolling forecast e non può essere verificata.
        """)
    
    # =============================================================================
    # GRAFICO CONTESTUALE
    # =============================================================================
    if df is not None:
        st.markdown("---")
        st.subheader("📈 Contesto: Pattern Giornaliero Tipico")
        
        # Calcola pattern medio per il tipo di giorno (weekend/feriale)
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
            pattern_label = "Giorni Feriali"
        
        # Crea grafico
        fig = go.Figure()
        
        # Pattern tipico
        fig.add_trace(go.Scatter(
            x=list(range(24)),
            y=hourly_pattern.values,
            mode='lines+markers',
            name=f'Pattern medio ({pattern_label})',
            line=dict(color='steelblue', width=2),
            fill='tozeroy',
            fillcolor='rgba(70, 130, 180, 0.2)',
            hovertemplate='Ora: %{x}:00<br>Traffico medio: %{y:,.0f}<extra></extra>'
        ))
        
        # Punto previsione
        fig.add_trace(go.Scatter(
            x=[selected_hour],
            y=[prediction],
            mode='markers',
            name='Previsione',
            marker=dict(color='red', size=15, symbol='star'),
            hovertemplate='Ora: %{x}:00<br>Previsione: %{y:,.0f}<extra></extra>'
        ))
        
        # Punto ground truth (se disponibile)
        if ground_truth is not None:
            fig.add_trace(go.Scatter(
                x=[selected_hour],
                y=[ground_truth],
                mode='markers',
                name='Valore Reale',
                marker=dict(color='green', size=12, symbol='circle'),
                hovertemplate='Ora: %{x}:00<br>Valore reale: %{y:,.0f}<extra></extra>'
            ))
        
        fig.update_layout(
            xaxis_title='Ora del giorno',
            yaxis_title='Traffico (veicoli/ora)',
            hovermode='x unified',
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # =============================================================================
    # DETTAGLI TECNICI
    # =============================================================================
    with st.expander("🔧 Dettagli Tecnici della Predizione"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📥 Input Forniti:**")
            st.write(f"- Data/Ora: `{target_dt.strftime('%Y-%m-%d %H:%M')}`")
            st.write(f"- Meteo: `{weather}`")
            st.write(f"- Temperatura: `{temperature}°C`")
            st.write(f"- Tipo giorno: `{'Weekend' if target_dt.dayofweek >= 5 else 'Feriale'}`")
            st.write(f"- Rush hour: `{'Sì' if selected_hour in [6,7,8,9,16,17,18] else 'No'}`")
        
        with col2:
            st.markdown("**🔢 Valori Lag Utilizzati:**")
            if 'lags' in locals() and lags:
                for lag_name in ['lag_1', 'lag_24', 'lag_168']:
                    val = lags.get(lag_name)
                    st.write(f"- `{lag_name}`: {val:,.0f} veicoli/h" if val else f"- `{lag_name}`: default")
            else:
                st.write("- Generati da rolling forecast")
        
        if ground_truth is not None:
            st.markdown("---")
            st.markdown("**📊 Valutazione Accuratezza:**")
            st.write(f"- Previsione: `{prediction:,.0f}` veicoli/h")
            st.write(f"- Ground Truth: `{ground_truth:,.0f}` veicoli/h")
            st.write(f"- Errore Assoluto: `{abs(error):,.0f}` veicoli/h")
            st.write(f"- Errore Percentuale: `{error_pct:.2f}%`")
            st.write(f"- MAE tipico del modello: ~300 veicoli/h")

# =============================================================================
# SIDEBAR INFORMAZIONI
# =============================================================================
with st.sidebar:
    st.markdown("### 📖 Guida all'Uso")
    
    st.markdown("**📊 Modalità Storica (2012-2018)**")
    st.info("""
    - Usa **lag reali** dal dataset
    - Mostra **ground truth** per confronto
    - Valuta **accuratezza** del modello
    - ⚡ Veloce (1 predizione)
    """)
    
    st.markdown("**🚀 Modalità Futura (>2018)**")
    st.warning("""
    - Usa **rolling forecast**
    - Predice ora per ora
    - Usa predizioni come lag
    - ⏱️ Più lento (N predizioni)
    - ⚠️ Errore si accumula nel tempo
    """)
    
    st.markdown("---")
    st.markdown("### ⚙️ Come Funziona")
    
    with st.expander("Rolling Forecast"):
        st.code("""
# Ultimo valore reale: 31 dic 2018
ultimo = 4500

# Target: 15 giu 2020
# Ore da predire: ~12,000

for ora in range(12000):
    lag_1 = predizione_precedente
    pred = model.predict(lag_1, ...)
    predizione_precedente = pred

risultato = pred  # ultima predizione
        """, language="python")
    
    st.markdown("---")
    st.markdown("### 🎯 Feature Principali")
    st.markdown("""
    **Temporali:**
    - Ora, giorno settimana, mese
    - Encoding ciclico (sin/cos)
    - Rush hour, weekend
    
    **Lag:**
    - lag_1, lag_24, lag_168
    - Differenze tra lag
    
    **Meteo:**
    - Temperatura
    - Condizione (Clear, Rain, Snow...)
    
    **Rolling:**
    - Media/std mobile 24h e 168h
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Best Practice")
    st.success("""
    **Per predizioni accurate:**
    
    1. Usa modalità storica per valutare il modello
    2. Per date future prossime (1-2 giorni) il rolling forecast è affidabile
    3. Per date molto lontane (mesi/anni) l'errore aumenta
    4. In produzione: aggiorna il dataset regolarmente con dati reali
    """)
    
    st.markdown("---")
    st.markdown("### ℹ️ Info Modello")
    st.markdown(f"""
    - **Algoritmo**: XGBoost
    - **Dataset**: {len(df):,} osservazioni
    - **Periodo**: 2012-2018
    - **Feature**: {len(feature_columns) if feature_columns else 'N/A'}
    - **Target**: traffic_volume (veicoli/ora)
    """)
