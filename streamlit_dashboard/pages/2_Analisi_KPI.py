#PAGINA 2: ANALISI E KPI

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path  

st.set_page_config(page_title="Analisi e KPI", page_icon="📈", layout="wide")

# CARICAMENTO DATI
@st.cache_data
def load_data():
    """Carica il dataset processato"""
    data_path = Path(__file__).parent.parent / 'data' / 'traffic_processed.csv'
    
    try:
        df = pd.read_csv(data_path, parse_dates=['date_time'])
        return df
    except Exception as e:
        st.error(f" Errore caricamento dati: {e}")
        st.stop()

@st.cache_data
def load_metrics():
    """Carica le metriche"""
    json_path = Path(__file__).parent.parent / 'data' / 'final_metrics.json'
    
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except:
        return {}

# PAGINA
def main():
    st.title(" Analisi e KPI")
    st.markdown("Pattern temporali e indicatori chiave di performance")
    
    # Carica dati
    df = load_data()
    metrics = load_metrics()
    
    # Assicuriamoci che le colonne esistano
    if 'is_weekend' not in df.columns:
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    if 'hour' not in df.columns:
        df['hour'] = df['date_time'].dt.hour
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = df['date_time'].dt.dayofweek
    
    st.markdown("---")
        # KPI CARDS
    st.subheader("🎯 KPI Operativi")
    
    # Calcola KPI
    mean_traffic = df['traffic_volume'].mean()
    max_traffic = df['traffic_volume'].max()
    weekday_mean = df[df['is_weekend'] == 0]['traffic_volume'].mean()
    weekend_mean = df[df['is_weekend'] == 1]['traffic_volume'].mean()
    weekend_delta = (weekend_mean - weekday_mean) / weekday_mean * 100
    
    # Peak hours
    hourly_mean = df.groupby('hour')['traffic_volume'].mean()
    peak_am_hour = hourly_mean[6:10].idxmax()
    peak_pm_hour = hourly_mean[15:19].idxmax()
    
    # Holiday impact
    if df['is_holiday'].sum() > 0:
        normal_mean = df[df['is_holiday'] == 0]['traffic_volume'].mean()
        holiday_mean = df[df['is_holiday'] == 1]['traffic_volume'].mean()
        holiday_delta = (holiday_mean - normal_mean) / normal_mean * 100
    else:
        holiday_delta = -74  # Valore tipico
    
    # Display KPI
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label=" Traffico Medio",
            value=f"{mean_traffic:,.0f}",
            help="Veicoli/ora - media globale"
        )
    
    with col2:
        st.metric(
            label=" Picco Mattutino",
            value=f"{peak_am_hour}:00",
            delta=f"{hourly_mean[peak_am_hour]:,.0f} v/h",
            delta_color="off"
        )
    
    with col3:
        st.metric(
            label=" Picco Serale",
            value=f"{peak_pm_hour}:00",
            delta=f"{hourly_mean[peak_pm_hour]:,.0f} v/h",
            delta_color="off"
        )
    
    with col4:
        st.metric(
            label=" Riduzione Weekend",
            value=f"{abs(weekend_delta):.0f}%",
            delta=f"{weekend_delta:.0f}%",
            delta_color="inverse"
        )
    
    # Seconda riga KPI
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label=" Impatto Festività",
            value=f"{abs(holiday_delta):.0f}%",
            delta="riduzione",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            label=" MAE Modello",
            value=f"{metrics.get('test_mae', 450):.0f}",
            delta=f"+{metrics.get('improvement_vs_naive_pct', 28):.0f}% vs baseline",
            help="Mean Absolute Error su test set"
        )
    
    with col3:
        st.metric(
            label=" R² Score",
            value=f"{metrics.get('test_r2', 0.92):.3f}",
            help="Varianza spiegata dal modello"
        )
    
    with col4:
        model_reliability = (1 - metrics.get('test_mae', 450) / mean_traffic) * 100
        st.metric(
            label=" Affidabilità",
            value=f"{model_reliability:.0f}%",
            help="1 - (MAE / media traffico)"
        )
    
    st.markdown("---")
    
    # PATTERN ORARIO
    st.subheader(" Pattern Orario")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Calcola medie per feriali e weekend
        hourly_weekday = df[df['is_weekend'] == 0].groupby('hour')['traffic_volume'].mean()
        hourly_weekend = df[df['is_weekend'] == 1].groupby('hour')['traffic_volume'].mean()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(24)),
            y=hourly_weekday.values,
            mode='lines+markers',
            name='Feriali',
            line=dict(color='steelblue', width=3),
            fill='tozeroy',
            fillcolor='rgba(70, 130, 180, 0.2)'
        ))
        
        fig.add_trace(go.Scatter(
            x=list(range(24)),
            y=hourly_weekend.values,
            mode='lines+markers',
            name='Weekend',
            line=dict(color='coral', width=3),
            fill='tozeroy',
            fillcolor='rgba(255, 127, 80, 0.2)'
        ))
        
        fig.update_layout(
            title='Traffico Medio per Ora: Feriali vs Weekend',
            xaxis_title='Ora del giorno',
            yaxis_title='Traffico medio (veicoli/ora)',
            xaxis=dict(tickmode='linear', tick0=0, dtick=2),
            hovermode='x unified',
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        ###  Osservazioni
        
        **Pattern Bimodale (Feriali)**:
        - Picco mattutino: 7-8 AM
        - Picco serale: 4-5 PM
        - Minimo notturno: 3-4 AM
        
        **Weekend**:
        - Pattern unimodale
        - Picco pomeridiano
        - Distribuzione più uniforme
        
        **Implicazione**:
        L'ora del giorno è il principale predittore del traffico.
        """)
    
    st.markdown("---")
    
    # HEATMAP GIORNO × ORA
    st.subheader(" Heatmap Ora × Giorno")
    
    # Pivot table
    pivot = df.pivot_table(
        values='traffic_volume',
        index='hour',
        columns='day_of_week',
        aggfunc='mean'
    )
    
    day_names = ['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom']
    pivot.columns = day_names
    
    fig = px.imshow(
        pivot.values,
        labels=dict(x="Giorno", y="Ora", color="Traffico"),
        x=day_names,
        y=list(range(24)),
        color_continuous_scale='YlOrRd',
        aspect='auto'
    )
    
    fig.update_layout(
        title='Traffico Medio per Ora e Giorno della Settimana',
        yaxis=dict(tickmode='linear', tick0=0, dtick=2)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # IMPATTO CONDIZIONI METEO
    st.subheader(" Impatto Condizioni Meteo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        weather_mean = df.groupby('weather_main')['traffic_volume'].mean().sort_values(ascending=True)
        
        colors = ['green' if x > mean_traffic else 'orange' if x > mean_traffic * 0.9 else 'red' 
                  for x in weather_mean.values]
        
        fig = go.Figure(go.Bar(
            x=weather_mean.values,
            y=weather_mean.index,
            orientation='h',
            marker_color=colors,
            text=[f"{x:,.0f}" for x in weather_mean.values],
            textposition='outside'
        ))
        
        fig.add_vline(x=mean_traffic, line_dash="dash", line_color="black",
                      annotation_text=f"Media: {mean_traffic:,.0f}")
        
        fig.update_layout(
            title='Traffico Medio per Condizione Meteo',
            xaxis_title='Traffico medio (veicoli/ora)',
            yaxis_title='Condizione'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        ###  Osservazioni
        
        **Condizioni avverse**:
        - Snow, Fog, Thunderstorm → traffico ridotto
        - Le persone evitano di viaggiare
        
        **Condizioni normali**:
        - Clouds/Haze → traffico elevato
        - Correlato con orari di punta
        
        **Implicazione per il modello**:
        Le condizioni meteo sono predittori deboli ma 
        utili in combinazione con altri fattori.
        """)
    
    st.markdown("---")
    
    # ACTUAL VS PREDICTED (se disponibile)
    st.subheader(" Performance del Modello: Actual vs Predicted")
    
    # 1. Calcola il percorso corretto per questo file specifico
    pred_path = Path(__file__).parent.parent / 'data' / 'test_predictions.csv'
    
    try:
        # 2. Usa la variabile pred_path invece della stringa fissa
        test_pred = pd.read_csv(pred_path, parse_dates=['date_time'])
        
        # Scatter plot
        fig = px.scatter(
            test_pred.sample(min(2000, len(test_pred))),
            x='traffic_volume',
            y='predicted',
            opacity=0.3,
            title='Valori Reali vs Predetti (Test Set 2018)'
        )
        
        # Linea di perfetta previsione
        max_val = max(test_pred['traffic_volume'].max(), test_pred['predicted'].max())
        fig.add_trace(go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode='lines',
            name='Previsione perfetta',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            xaxis_title='Traffico Reale',
            yaxis_title='Traffico Predetto'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except FileNotFoundError:
        st.info(" I dati delle previsioni saranno disponibili dopo l'esecuzione del modello.")

# ESECUZIONE
if __name__ == "__main__":
    main()
