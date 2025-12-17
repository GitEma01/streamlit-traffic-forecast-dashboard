#PAGINA 1: ESPLORAZIONE DATI
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path

st.set_page_config(page_title="Esplorazione Dati", page_icon="📊", layout="wide")

# CARICAMENTO DATI
@st.cache_data
def load_data():
    """Carica il dataset processato"""
    # 1. Trova il percorso del file corrente
    current_file_path = Path(__file__)
    
    # 2. Risale di DUE livelli per arrivare alla root del progetto
    data_path = current_file_path.parent.parent / 'data' / 'traffic_processed.csv'
    
    try:
        df = pd.read_csv(data_path, parse_dates=['date_time'])
        return df
    except FileNotFoundError:
        # Dati di esempio se il file non esiste
        st.warning(f" File non trovato in: {data_path}. Mostrando dati di esempio.")
        dates = pd.date_range('2017-01-01', periods=1000, freq='H')
        df = pd.DataFrame({
            'date_time': dates,
            'traffic_volume': np.random.randint(500, 6000, 1000),
            'temp_celsius': np.random.uniform(-10, 35, 1000),
            'weather_main': np.random.choice(['Clear', 'Clouds', 'Rain'], 1000),
            'hour': dates.hour,
            'day_of_week': dates.dayofweek,
            'is_holiday': np.random.choice([0, 1], 1000, p=[0.97, 0.03])
        })
        return df
        
# PAGINA
def main():
    st.title(" Esplorazione Dati")
    st.markdown("Visualizza e filtra il dataset Metro Interstate Traffic Volume")
    
    # Carica dati
    df = load_data()
    
    st.markdown("---")
    
    # SIDEBAR - FILTRI
    st.sidebar.header("🔍 Filtri")
    
    # Filtro anno
    years = sorted(df['date_time'].dt.year.unique())
    selected_years = st.sidebar.multiselect(
        "Anno",
        options=years,
        default=years,
        help="Seleziona uno o più anni"
    )
    
    # Filtro mese
    months = list(range(1, 13))
    month_names = ['Gen', 'Feb', 'Mar', 'Apr', 'Mag', 'Giu', 
                   'Lug', 'Ago', 'Set', 'Ott', 'Nov', 'Dic']
    selected_months = st.sidebar.multiselect(
        "Mese",
        options=months,
        format_func=lambda x: month_names[x-1],
        default=months
    )
    
    # Filtro giorno settimana
    days = list(range(7))
    day_names = ['Lunedì', 'Martedì', 'Mercoledì', 'Giovedì', 
                 'Venerdì', 'Sabato', 'Domenica']
    selected_days = st.sidebar.multiselect(
        "Giorno della Settimana",
        options=days,
        format_func=lambda x: day_names[x],
        default=days
    )
    
    # Applicazione filtri
    mask = (
        (df['date_time'].dt.year.isin(selected_years)) &
        (df['date_time'].dt.month.isin(selected_months)) &
        (df['date_time'].dt.dayofweek.isin(selected_days))
    )
    df_filtered = df[mask]
    
    # STATISTICHE RIASSUNTIVE
    st.subheader(" Statistiche Riassuntive")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Osservazioni", f"{len(df_filtered):,}")
    
    with col2:
        st.metric("Media Traffico", f"{df_filtered['traffic_volume'].mean():,.0f}")
    
    with col3:
        st.metric("Max Traffico", f"{df_filtered['traffic_volume'].max():,}")
    
    with col4:
        st.metric("Min Traffico", f"{df_filtered['traffic_volume'].min():,}")
    
    st.markdown("---")
    
    # SERIE TEMPORALE
    st.subheader(" Serie Temporale")
    
    # Opzioni di aggregazione
    agg_options = {
        'Orario (raw)': 'H',
        'Giornaliero': 'D',
        'Settimanale': 'W',
        'Mensile': 'M'
    }
    
    col1, col2 = st.columns([1, 3])
    with col1:
        agg_choice = st.selectbox("Aggregazione", list(agg_options.keys()))
    
    # Prepara dati per il grafico
    if agg_choice == 'Orario (raw)':
        # Limita a ultimi 30 giorni per performance
        df_plot = df_filtered.set_index('date_time')['traffic_volume'].tail(720)
    else:
        df_plot = df_filtered.set_index('date_time')['traffic_volume'].resample(
            agg_options[agg_choice]
        ).mean()
    
    # Grafico
    fig = px.line(
        df_plot.reset_index(),
        x='date_time',
        y='traffic_volume',
        title='Volume di Traffico nel Tempo'
    )
    fig.update_layout(
        xaxis_title='Data',
        yaxis_title='Traffico (veicoli/ora)',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # PREVIEW DATASET
    st.subheader(" Preview Dataset")
    
    # Opzioni di visualizzazione
    col1, col2 = st.columns([1, 3])
    with col1:
        n_rows = st.slider("Righe da mostrare", 10, 500, 100)
    
    # Mostra le colonne più rilevanti
    display_cols = ['date_time', 'traffic_volume', 'temp_celsius', 
                    'weather_main', 'is_holiday']
    available_cols = [c for c in display_cols if c in df_filtered.columns]
    
    st.dataframe(
        df_filtered[available_cols].head(n_rows),
        use_container_width=True,
        hide_index=True
    )
    
    # DISTRIBUZIONE
    st.subheader(" Distribuzione Traffic Volume")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_hist = px.histogram(
            df_filtered,
            x='traffic_volume',
            nbins=50,
            title='Istogramma Traffic Volume'
        )
        fig_hist.update_layout(
            xaxis_title='Traffico (veicoli/ora)',
            yaxis_title='Frequenza'
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        fig_box = px.box(
            df_filtered,
            y='traffic_volume',
            title='Boxplot Traffic Volume'
        )
        fig_box.update_layout(yaxis_title='Traffico (veicoli/ora)')
        st.plotly_chart(fig_box, use_container_width=True)
    
    # INFO DATASET
    with st.expander(" Informazioni sul Dataset"):
        st.markdown(f"""
        ### Dataset: Metro Interstate Traffic Volume
        
        **Fonte**: UCI Machine Learning Repository  
        **Posizione**: Interstate 94, Minneapolis-St.Paul, Minnesota, USA  
        **Periodo**: {df['date_time'].min().strftime('%Y-%m-%d')} - {df['date_time'].max().strftime('%Y-%m-%d')}  
        **Granularità**: Oraria  
        **Osservazioni totali**: {len(df):,}
        
        ### Variabili
        
        | Variabile | Descrizione |
        |-----------|-------------|
        | `traffic_volume` | Volume di traffico orario (target) |
        | `temp_celsius` | Temperatura in Celsius |
        | `weather_main` | Condizione meteo principale |
        | `is_holiday` | Flag festività |
        | `hour`, `day_of_week` | Componenti temporali |
        """)

# ESECUZIONE
if __name__ == "__main__":
    main()
