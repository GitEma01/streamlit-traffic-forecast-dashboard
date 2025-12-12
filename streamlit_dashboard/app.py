"""
================================================================================
TRAFFIC FORECAST DASHBOARD - I-94 MINNEAPOLIS-ST.PAUL
================================================================================
VERSIONE PREMIUM con HTML/CSS Custom - FIX RENDERING
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path

# =============================================================================
# CONFIGURAZIONE PAGINA
# =============================================================================
st.set_page_config(
    page_title="Traffic Forecast I-94",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# STILI CSS PREMIUM
# =============================================================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    /* Applica font a tutto */
    html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    }
    
    /* Nascondi header default Streamlit */
    #MainMenu {visibility: hidden;}
    
    
    /* Background principale */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* HEADER HERO */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        border-radius: 25px;
        padding: 50px 40px;
        text-align: center;
        margin-bottom: 40px;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }
    .hero-title {
        font-size: 4rem;
        font-weight: 800;
        color: white;
        margin: 0;
        text-shadow: 2px 4px 20px rgba(0,0,0,0.3);
    }
    .hero-subtitle {
        font-size: 1.5rem;
        color: rgba(255,255,255,0.9);
        margin-top: 15px;
        font-weight: 400;
    }
    
    /* METRIC CARDS */
    .metrics-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 25px;
        margin-bottom: 40px;
    }
    .metric-card {
        background: linear-gradient(145deg, #1e2a4a 0%, #2d3a5a 100%);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
    }
    .metric-icon {
        font-size: 2.5rem;
        margin-bottom: 15px;
    }
    .metric-value {
        font-size: 2.8rem;
        font-weight: 700;
        color: #fff;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 1rem;
        color: rgba(255,255,255,0.7);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-delta {
        font-size: 0.9rem;
        color: #4ade80;
        margin-top: 8px;
        font-weight: 500;
    }
    
    .metric-card.blue { border-top: 4px solid #3b82f6; }
    .metric-card.purple { border-top: 4px solid #8b5cf6; }
    .metric-card.green { border-top: 4px solid #10b981; }
    .metric-card.orange { border-top: 4px solid #f59e0b; }
    
    /* SECTION TITLE */
    .section-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #fff;
        margin: 30px 0 20px 0;
        display: flex;
        align-items: center;
        gap: 15px;
    }
    .section-title-icon {
        font-size: 2.2rem;
    }
    
    /* SECTION CARD */
    .section-card {
        background: linear-gradient(145deg, #1e2a4a 0%, #2d3a5a 100%);
        border-radius: 25px;
        padding: 35px;
        margin-bottom: 25px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* SECTION CONTENT */
    .section-content {
        color: rgba(255,255,255,0.85);
        font-size: 1.15rem;
        line-height: 1.8;
    }
    .section-content strong {
        color: #a5b4fc;
    }
    
    /* LISTA STILIZZATA */
    .styled-list {
        list-style: none;
        padding: 0;
        margin: 15px 0;
    }
    .styled-list li {
        padding: 15px 20px;
        margin: 10px 0;
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        display: flex;
        align-items: center;
        gap: 15px;
        transition: background 0.3s ease;
    }
    .styled-list li:hover {
        background: rgba(255,255,255,0.1);
    }
    .list-icon {
        font-size: 1.5rem;
    }
    
    /* LOCATION BOX */
    .location-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 25px;
        padding: 35px;
        color: white;
        text-align: center;
        box-shadow: 0 15px 50px rgba(102, 126, 234, 0.4);
        margin-bottom: 20px;
    }
    .location-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 20px;
    }
    .location-detail {
        font-size: 1.15rem;
        margin: 12px 0;
        opacity: 0.95;
    }
    .location-link {
        display: inline-block;
        margin-top: 20px;
        padding: 15px 30px;
        background: rgba(255,255,255,0.2);
        border-radius: 30px;
        color: white;
        text-decoration: none;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        border: 2px solid rgba(255,255,255,0.3);
    }
    .location-link:hover {
        background: rgba(255,255,255,0.3);
        transform: scale(1.05);
        color: white;
    }
    
    /* NAV CARDS */
    .nav-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 20px;
        margin-top: 20px;
    }
    .nav-card {
        background: rgba(255,255,255,0.05);
        border-radius: 15px;
        padding: 25px;
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .nav-card:hover {
        background: rgba(255,255,255,0.1);
        transform: translateX(10px);
        border-color: #667eea;
    }
    .nav-card-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #fff;
        margin-bottom: 8px;
    }
    .nav-card-desc {
        font-size: 1rem;
        color: rgba(255,255,255,0.6);
    }
    
    /* TECH GRID */
    .tech-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 30px;
        margin-top: 15px;
    }
    .tech-column h4 {
        font-size: 1.4rem;
        color: #a5b4fc;
        margin-bottom: 15px;
    }
    .tech-column ul {
        list-style: none;
        padding: 0;
    }
    .tech-column li {
        padding: 8px 0;
        color: rgba(255,255,255,0.8);
        font-size: 1.05rem;
    }
    
    /* FOOTER */
    .footer {
        text-align: center;
        padding: 40px 20px;
        margin-top: 40px;
        border-top: 1px solid rgba(255,255,255,0.1);
    }
    .footer-text {
        color: rgba(255,255,255,0.6);
        font-size: 1rem;
        margin: 8px 0;
    }
    .footer-authors {
        color: rgba(255,255,255,0.5);
        font-size: 0.95rem;
    }
    
    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%) !important;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Rinomina "app" in "Menu Principale" */
    [data-testid="stSidebarNav"] li:first-child a span {
        visibility: hidden;
        position: relative;
    }
    [data-testid="stSidebarNav"] li:first-child a span::after {
        content: "Menu Principale";
        visibility: visible;
        position: absolute;
        left: 0;
        font-weight: 600;
    }
    
    /* RESPONSIVE */
    @media (max-width: 768px) {
        .metrics-container {
            grid-template-columns: repeat(2, 1fr);
        }
        .nav-grid, .tech-grid {
            grid-template-columns: 1fr;
        }
        .hero-title {
            font-size: 2.5rem;
        }
    }
    /* =========================================
       FIX MENU SIDEBAR ATTIVO
       ========================================= */
    div[data-testid="stSidebarNav"] li a[aria-current="page"] {
        background-color: #667eea !important;
        border: 1px solid rgba(255,255,255,0.2);
    }

    div[data-testid="stSidebarNav"] li a[aria-current="page"] span {
        color: white !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# FUNZIONI UTILITY
# =============================================================================
@st.cache_data
def load_metrics():
    try:
        # Trova il percorso assoluto della cartella dove si trova app.py
        base_path = Path(__file__).parent
        # Costruisce il percorso completo al file json
        file_path = base_path / 'data' / 'final_metrics.json'
        
        with open(file_path, 'r') as f:
            metrics = json.load(f)
            return {
                'test_mae': metrics.get('test_mae', metrics.get('mae_test', 141)),
                'test_r2': metrics.get('test_r2', metrics.get('r2_test', 0.988)),
                'improvement_vs_naive_pct': metrics.get('improvement_vs_naive_pct', 76),
            }
    except Exception as e:
        # Stampa l'errore nei log di Streamlit Cloud per il debug
        print(f"Errore caricamento metriche: {e}")
        return {'test_mae': 141, 'test_r2': 0.988, 'improvement_vs_naive_pct': 76}

@st.cache_data  
def load_data_summary():
    try:
        # Trova il percorso assoluto della cartella dove si trova app.py
        base_path = Path(__file__).parent
        # Costruisce il percorso completo al file csv
        file_path = base_path / 'data' / 'traffic_processed.csv'
        
        df = pd.read_csv(file_path, parse_dates=['date_time'])
        return {
            'n_rows': len(df),
            'date_start': df['date_time'].min().strftime('%b %Y'),
            'date_end': df['date_time'].max().strftime('%b %Y'),
            'mean_traffic': df['traffic_volume'].mean(),
        }
    except Exception as e:
        # Stampa l'errore nei log di Streamlit Cloud per il debug
        print(f"Errore caricamento CSV: {e}")
        return {'n_rows': 40575, 'date_start': 'Oct 2012', 'date_end': 'Sep 2018', 'mean_traffic': 3290}

# =============================================================================
# CONTENUTO HOMEPAGE
# =============================================================================
def main():
    metrics = load_metrics()
    summary = load_data_summary()
    
    # HERO SECTION
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">🚗 Traffic Forecast Dashboard</h1>
        <p class="hero-subtitle">Interstate 94 — Minneapolis-St.Paul, USA</p>
    </div>
    """, unsafe_allow_html=True)
    
    # METRICS CARDS
    st.markdown(f"""
    <div class="metrics-container">
        <div class="metric-card blue">
            <div class="metric-icon">🎯</div>
            <div class="metric-value">{metrics['test_mae']:.0f}</div>
            <div class="metric-label">MAE (Test Set)</div>
            <div class="metric-delta">+{metrics['improvement_vs_naive_pct']:.0f}% vs baseline</div>
        </div>
        <div class="metric-card purple">
            <div class="metric-icon">📈</div>
            <div class="metric-value">{metrics['test_r2']:.3f}</div>
            <div class="metric-label">R² Score</div>
            <div class="metric-delta">Varianza spiegata</div>
        </div>
        <div class="metric-card green">
            <div class="metric-icon">📊</div>
            <div class="metric-value">{summary['n_rows']:,}</div>
            <div class="metric-label">Osservazioni</div>
            <div class="metric-delta">{summary['date_start']} - {summary['date_end']}</div>
        </div>
        <div class="metric-card orange">
            <div class="metric-icon">🚗</div>
            <div class="metric-value">{summary['mean_traffic']:,.0f}</div>
            <div class="metric-label">Traffico Medio</div>
            <div class="metric-delta">veicoli/ora</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # DUE COLONNE
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # INFORMAZIONI - Card separata
        st.markdown("""
        <div class="section-card">
            <div class="section-title">
                <span class="section-title-icon">📖</span>
                Informazioni sul Progetto
            </div>
            <div class="section-content">
                Questa dashboard presenta un sistema di <strong>previsione del traffico</strong> per 
                l'Interstate 94, un'importante autostrada che collega Minneapolis e St.Paul 
                in Minnesota, USA.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # OBIETTIVO - Card separata
        st.markdown("""
        <div class="section-card">
            <div class="section-title">
                <span class="section-title-icon">🎯</span>
                Obiettivo
            </div>
            <div class="section-content">
                Prevedere il volume di traffico orario utilizzando:
            </div>
            <ul class="styled-list">
                <li><span class="list-icon">📅</span> <strong>Pattern temporali</strong> — ora, giorno, festività</li>
                <li><span class="list-icon">🌤️</span> <strong>Condizioni meteo</strong> — temperatura, pioggia, neve</li>
                <li><span class="list-icon">📊</span> <strong>Dati storici</strong> — lag features, rolling statistics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # MODELLO - Card separata
        st.markdown("""
        <div class="section-card">
            <div class="section-title">
                <span class="section-title-icon">🤖</span>
                Modello
            </div>
            <div class="section-content">
                Il sistema utilizza <strong>XGBoost</strong>, un algoritmo di Machine Learning
                stato dell'arte per dati tabulari, addestrato su 6 anni di dati storici.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # LOCATION BOX
        google_maps_url = "https://www.google.com/maps/place/Interstate+94,+Minnesota,+USA/@44.9778,-93.2650,11z"
        
        st.markdown(f"""
        <div class="location-box">
            <div class="location-title">📍 Interstate 94</div>
            <div class="location-detail">🏙️ Minneapolis-St.Paul</div>
            <div class="location-detail">🇺🇸 Minnesota, USA</div>
            <div class="location-detail">📅 Oct 2012 - Sep 2018</div>
            <div class="location-detail">⏰ Granularità: Oraria</div>
            <a href="{google_maps_url}" target="_blank" class="location-link">
                🗺️ Apri in Google Maps
            </a>
        </div>
        """, unsafe_allow_html=True)
        
        # Mappa
        st.components.v1.iframe(
            src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d181105.2535594!2d-93.3499489!3d44.9706993!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x87f6290f1e64e181%3A0x8e3b8d0c6e2b6e0d!2sInterstate%2094%2C%20Minnesota!5e0!3m2!1sen!2sus!4v1234567890",
            height=220
        )
    
    # NAVIGAZIONE
    st.markdown("""
    <div class="section-card">
        <div class="section-title">
            <span class="section-title-icon">📚</span>
            Navigazione
        </div>
        <div class="section-content">
            Usa il <strong>menu laterale a sinistra</strong> per esplorare le diverse sezioni:
        </div>
        <div class="nav-grid">
            <div class="nav-card">
                <div class="nav-card-title">📊 Esplorazione</div>
                <div class="nav-card-desc">Visualizza il dataset completo e la serie storica</div>
            </div>
            <div class="nav-card">
                <div class="nav-card-title">📈 Analisi e KPI</div>
                <div class="nav-card-desc">Pattern temporali, stagionalità e indicatori chiave</div>
            </div>
            <div class="nav-card">
                <div class="nav-card-title">🔮 Previsioni</div>
                <div class="nav-card-desc">Effettua previsioni interattive con il modello</div>
            </div>
            <div class="nav-card">
                <div class="nav-card-title">🧪 Backtesting</div>
                <div class="nav-card-desc">Valuta le performance del modello sul Test Set</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # DETTAGLI TECNICI
    st.markdown("""
    <div class="section-card">
        <div class="section-title">
            <span class="section-title-icon">⚙️</span>
            Dettagli Tecnici
        </div>
        <div class="tech-grid">
            <div class="tech-column">
                <h4>🛠️ Stack Tecnologico</h4>
                <ul>
                    <li>• <strong>Python</strong> 3.10+</li>
                    <li>• <strong>Streamlit</strong> per la dashboard</li>
                    <li>• <strong>XGBoost</strong> per il modello</li>
                    <li>• <strong>Plotly</strong> per visualizzazioni</li>
                    <li>• <strong>Pandas</strong> per data processing</li>
                </ul>
            </div>
            <div class="tech-column">
                <h4>📋 Metodologia</h4>
                <ul>
                    <li>• Split: Train(2012-16)/Val(2017)/Test(2018)</li>
                    <li>• Tuning con TimeSeriesSplit (5 fold)</li>
                    <li>• Feature engineering: 40+ features</li>
                    <li>• Metriche: MAE, RMSE, R², MAPE</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # FOOTER
    st.markdown("""
    <div class="footer">
        <p class="footer-text">Progetto realizzato per Sistemi Informativi and Business Intelligence — Università di Napoli Federico II</p>
        <p class="footer-authors">Progetto realizzato da: Emanuele Santacroce, Alberto Stravato, Francesco Floriano</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
