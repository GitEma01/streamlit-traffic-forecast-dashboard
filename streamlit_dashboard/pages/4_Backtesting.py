#PAGINA 4: BACKTESTING
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path

st.set_page_config(page_title="Backtesting", page_icon="🧪", layout="wide")

# CARICAMENTO DATI
@st.cache_data
def load_test_predictions():
    """Carica le previsioni sul test set con percorso assoluto"""
    file_path = Path(__file__).parent.parent / 'data' / 'test_predictions.csv'
    
    try:
        # 2. Prova a caricare il file reale
        df = pd.read_csv(file_path, parse_dates=['date_time'])
        
        # 3. Calcola le metriche di errore (necessarie per i grafici)
        df['error'] = df['traffic_volume'] - df['predicted']
        df['abs_error'] = np.abs(df['error'])
        # Evita divisione per zero
        df['pct_error'] = df['abs_error'] / df['traffic_volume'].replace(0, 1) * 100
        df['hour'] = df['date_time'].dt.hour
        df['day_of_week'] = df['date_time'].dt.dayofweek
        return df
        
    except FileNotFoundError:
        st.warning(f" File '{file_path.name}' non trovato. Generazione dati simulati per demo.")
        
        # --- LOGICA DI BACKUP (DATI SIMULATI) ---
        np.random.seed(42)
        dates = pd.date_range('2018-01-01', periods=6000, freq='H')
        base = 3000 + 2000 * np.sin(np.pi * dates.hour / 12 - np.pi/2)
        actual = (base + np.random.normal(0, 500, len(dates))).clip(0, 7000)
        predicted = actual + np.random.normal(0, 400, len(dates))
        
        df = pd.DataFrame({
            'date_time': dates,
            'traffic_volume': actual.astype(int),
            'predicted': predicted.clip(0, 7000).astype(int)
        })
        df['error'] = df['traffic_volume'] - df['predicted']
        df['abs_error'] = np.abs(df['error'])
        df['pct_error'] = df['abs_error'] / df['traffic_volume'].replace(0, 1) * 100
        df['hour'] = df['date_time'].dt.hour
        df['day_of_week'] = df['date_time'].dt.dayofweek
        return df

@st.cache_data
def load_metrics():
    """Carica le metriche del modello con percorso assoluto"""
    json_path = Path(__file__).parent.parent / 'data' / 'final_metrics.json'
    
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning(" Metriche non trovate. Uso valori di default.")
        return {'test_mae': 450, 'baseline_naive_mae': 625}

# PAGINA
def main():
    st.title(" Backtesting del Modello")
    st.markdown("Valutazione sistematica delle performance sul Test Set (2018)")
    
    df = load_test_predictions()
    metrics = load_metrics()
    
    st.markdown("---")
    
    # METRICHE RIASSUNTIVE
    st.subheader(" Metriche di Performance")
    
    mae = df['abs_error'].mean()
    rmse = np.sqrt((df['error'] ** 2).mean())
    r2 = 1 - (df['error'] ** 2).sum() / ((df['traffic_volume'] - df['traffic_volume'].mean()) ** 2).sum()
    mape = df['pct_error'].mean()
    baseline_mae = metrics.get('baseline_naive_mae', 625)
    improvement = (baseline_mae - mae) / baseline_mae * 100
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(label="MAE", value=f"{mae:.0f}", help="Mean Absolute Error")
    with col2:
        st.metric(label="RMSE", value=f"{rmse:.0f}", help="Root Mean Squared Error")
    with col3:
        st.metric(label="R²", value=f"{r2:.4f}", help="Coefficiente di determinazione")
    with col4:
        st.metric(label="MAPE", value=f"{mape:.1f}%", help="Mean Absolute Percentage Error")
    with col5:
        st.metric(label="vs Baseline", value=f"+{improvement:.0f}%", help="Miglioramento vs Naive")
    
    st.markdown("---")
    
    # SELEZIONE PERIODO
    st.subheader(" Seleziona Periodo di Analisi")
    
    col1, col2 = st.columns(2)
    with col1:
        date_range = st.date_input(
            "Intervallo date",
            value=(df['date_time'].min().date(), df['date_time'].max().date()),
            min_value=df['date_time'].min().date(),
            max_value=df['date_time'].max().date()
        )
    with col2:
        aggregation = st.selectbox("Aggregazione", ['Orario', 'Giornaliero', 'Settimanale'], index=1)
    
    if len(date_range) == 2:
        mask = (df['date_time'].dt.date >= date_range[0]) & (df['date_time'].dt.date <= date_range[1])
        df_filtered = df[mask].copy()
    else:
        df_filtered = df.copy()
    
    st.markdown("---")
    
    # GRAFICO ACTUAL VS PREDICTED
    st.subheader(" Actual vs Predicted nel Tempo")
    
    if aggregation == 'Giornaliero':
        df_plot = df_filtered.set_index('date_time').resample('D').agg({
            'traffic_volume': 'mean', 'predicted': 'mean'
        }).reset_index()
    elif aggregation == 'Settimanale':
        df_plot = df_filtered.set_index('date_time').resample('W').agg({
            'traffic_volume': 'mean', 'predicted': 'mean'
        }).reset_index()
    else:
        df_plot = df_filtered.sample(min(500, len(df_filtered))).sort_values('date_time')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot['date_time'], y=df_plot['traffic_volume'],
                             mode='lines', name='Actual', line=dict(color='steelblue', width=2)))
    fig.add_trace(go.Scatter(x=df_plot['date_time'], y=df_plot['predicted'],
                             mode='lines', name='Predicted', line=dict(color='coral', width=2, dash='dash')))
    fig.update_layout(title=f'Traffico Reale vs Predetto ({aggregation})',
                      xaxis_title='Data', yaxis_title='Traffico (veicoli/ora)', hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # SCATTER E DISTRIBUZIONE
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Scatter: Actual vs Predicted")
        df_sample = df_filtered.sample(min(2000, len(df_filtered)))
        fig_scatter = px.scatter(df_sample, x='traffic_volume', y='predicted', opacity=0.3,
                                 color='abs_error', color_continuous_scale='Reds')
        max_val = max(df_filtered['traffic_volume'].max(), df_filtered['predicted'].max())
        fig_scatter.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines',
                                         name='Perfetto', line=dict(color='black', dash='dash')))
        fig_scatter.update_layout(xaxis_title='Traffico Reale', yaxis_title='Traffico Predetto')
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        st.subheader(" Distribuzione Errori")
        fig_hist = px.histogram(df_filtered, x='error', nbins=50, color_discrete_sequence=['steelblue'])
        fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
        fig_hist.update_layout(xaxis_title='Errore (veicoli/ora)', yaxis_title='Frequenza')
        st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("---")
    
    # ERRORE PER ORA
    st.subheader(" Errore per Fascia Oraria")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        error_by_hour = df_filtered.groupby('hour')['abs_error'].mean()
        colors = ['green' if x < mae else 'orange' if x < mae * 1.2 else 'red' for x in error_by_hour.values]
        
        fig_hour = go.Figure(go.Bar(x=list(range(24)), y=error_by_hour.values, marker_color=colors,
                                    text=[f"{x:.0f}" for x in error_by_hour.values], textposition='outside'))
        fig_hour.add_hline(y=mae, line_dash="dash", line_color="black", annotation_text=f"MAE: {mae:.0f}")
        fig_hour.update_layout(title='MAE per Ora', xaxis_title='Ora', yaxis_title='MAE',
                               xaxis=dict(tickmode='linear', tick0=0, dtick=2))
        st.plotly_chart(fig_hour, use_container_width=True)
    
    with col2:
        st.markdown("###  Analisi")
        best_hours = error_by_hour.nsmallest(3)
        worst_hours = error_by_hour.nlargest(3)
        
        st.markdown("**Ore migliori:**")
        for h, err in best_hours.items():
            st.markdown(f"- {h:02d}:00 → MAE {err:.0f}")
        
        st.markdown("**Ore peggiori:**")
        for h, err in worst_hours.items():
            st.markdown(f"- {h:02d}:00 → MAE {err:.0f}")
    
    st.markdown("---")
    
    # TABELLA CAMPIONE
    st.subheader(" Campione di Previsioni")
    
    n_samples = st.slider("Numero di campioni", 10, 100, 20)
    sample = df_filtered.sample(n_samples).sort_values('date_time')
    
    display_df = sample[['date_time', 'traffic_volume', 'predicted', 'error', 'abs_error']].copy()
    display_df.columns = ['Data/Ora', 'Reale', 'Predetto', 'Errore', 'Errore Assoluto']
    display_df['Data/Ora'] = display_df['Data/Ora'].dt.strftime('%Y-%m-%d %H:%M')
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # CONCLUSIONI
    st.markdown("---")
    st.subheader(" Conclusioni del Backtesting")
    
    within_500 = (df_filtered['abs_error'] <= 500).mean() * 100
    within_1000 = (df_filtered['abs_error'] <= 1000).mean() * 100
    
    st.success(f"""
    **Risultati del Backtesting sul Test Set (2018):**
    
    -  **MAE: {mae:.0f}** veicoli/ora (miglioramento del {improvement:.0f}% vs baseline)
    -  **R²: {r2:.4f}** (il modello spiega il {r2*100:.1f}% della varianza)
    -  **{within_500:.1f}%** delle previsioni hanno errore ≤ 500 veicoli/ora
    -  **{within_1000:.1f}%** delle previsioni hanno errore ≤ 1000 veicoli/ora
    
    Il modello è pronto per il deployment in produzione.
    """)

if __name__ == "__main__":
    main()
