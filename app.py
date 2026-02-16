"""
═══════════════════════════════════════════════════════════════════════
  MMM Studio — Marketing Mix Modeling powered by Google Meridian
═══════════════════════════════════════════════════════════════════════
  Application Streamlit pour l'analyse des performances marketing,
  la modélisation MMM et l'optimisation budgétaire.

  Lancer avec:  streamlit run app.py
═══════════════════════════════════════════════════════════════════════
"""

import streamlit as st

# ── Page config (doit être le premier appel Streamlit) ──
st.set_page_config(
    page_title="MMM Studio · Yuri & Neil",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — Yuri & Neil branding ──
st.markdown("""
<style>
    /* ── Typographie & couleurs ── */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

    .stApp {
        font-family: 'DM Sans', sans-serif;
    }
    code, .stCode {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ── Sidebar — Yuri & Neil dark theme ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020002 0%, #1a0a2e 50%, #2e1066 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0 !important;
    }

    /* ── Métriques ── */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8fafc 0%, #f3f0f9 100%);
        border: 1px solid #e2d8f0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(46,16,102,0.08);
    }
    [data-testid="stMetric"] label {
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-weight: 700 !important;
        color: #1a0a2e !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #f3f0f9;
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        background: white !important;
        box-shadow: 0 1px 3px rgba(46,16,102,0.12);
    }

    /* ── Cards ── */
    .metric-card {
        background: white;
        border: 1px solid #e2d8f0;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(46,16,102,0.06);
        margin-bottom: 16px;
    }
    .metric-card h3 {
        margin: 0 0 8px 0;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a0a2e;
        line-height: 1.1;
    }
    .metric-card .delta {
        font-size: 0.85rem;
        margin-top: 4px;
    }
    .delta-positive { color: #16a34a; }
    .delta-negative { color: #dc2626; }

    /* ── Status badges ── */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 100px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .badge-green { background: #dcfce7; color: #166534; }
    .badge-amber { background: #fef3c7; color: #92400e; }
    .badge-red { background: #fee2e2; color: #991b1b; }
    .badge-blue { background: #ede9fe; color: #2e1066; }

    /* ── Expanders ── */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    /* ── Dividers ── */
    hr {
        border: none;
        border-top: 1px solid #e2d8f0;
        margin: 2rem 0;
    }

    /* ── Hero header — Yuri & Neil gradient ── */
    .hero {
        background: linear-gradient(135deg, #020002 0%, #2e1066 50%, #c079e1 100%);
        color: white;
        padding: 40px 48px;
        border-radius: 16px;
        margin-bottom: 32px;
    }
    .hero h1 {
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.2;
    }
    .hero p {
        font-size: 1.1rem;
        opacity: 0.85;
        margin: 12px 0 0 0;
        max-width: 600px;
    }
    .hero .version {
        font-size: 0.75rem;
        opacity: 0.5;
        margin-top: 16px;
    }

    /* ── Pipeline steps ── */
    .pipeline-step {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 16px 20px;
        background: white;
        border: 1px solid #e2d8f0;
        border-radius: 10px;
        margin-bottom: 8px;
        transition: all 0.15s;
    }
    .pipeline-step:hover {
        border-color: #c079e1;
        box-shadow: 0 2px 6px rgba(46,16,102,0.1);
    }
    .pipeline-step .step-num {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: #f3f0f9;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        color: #2e1066;
        flex-shrink: 0;
    }
    .pipeline-step.active .step-num {
        background: #c079e1;
        color: white;
    }
    .pipeline-step.done .step-num {
        background: #16a34a;
        color: white;
    }

    /* ── Yuri & Neil logo text ── */
    .yurineil-logo {
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        background: linear-gradient(90deg, #ffffff 0%, #c079e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* ── Footer ── */
    .yurineil-footer {
        font-size: 0.75rem;
        color: #c079e1;
        text-align: center;
        padding: 8px 0;
    }
    .yurineil-footer a {
        color: #c079e1;
        text-decoration: none;
        font-weight: 600;
    }
    .yurineil-footer a:hover {
        color: #ffffff;
        text-decoration: underline;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <div style="font-size: 1.4rem; font-weight: 700; color: #e2e8f0; letter-spacing: 0.02em;">
            Yuri & Neil · MMM Studio
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Navigation
    page = st.radio(
        "Navigation",
        [
            "🏠 Accueil",
            "📁 Import & Exploration",
            "📈 Performance Canaux",
            "⚙️ Configuration Modèle",
            "🔬 Résultats & Diagnostics",
            "💰 Optimisation Budget",
            "📋 Planification Budget",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Pipeline status
    st.markdown("##### 📋 Pipeline Status")

    steps = {
        "data_loaded": "Données chargées",
        "eda_done": "EDA complète",
        "model_configured": "Modèle configuré",
        "model_trained": "Modèle entraîné",
        "optimization_done": "Optimisation faite",
        "budget_planning_done": "Budget planifié",
    }

    for key, label in steps.items():
        status = st.session_state.get(key, False)
        icon = "✅" if status else "⬜"
        st.markdown(f"{icon} {label}")

    st.markdown("---")
    st.markdown(
        '<div class="yurineil-footer">'
        'Powered by <a href="#" target="_blank">Yuri & Neil</a>'
        '</div>',
        unsafe_allow_html=True,
    )


# ── Router ──
if page == "🏠 Accueil":
    from views import home
    home.render()

elif page == "📁 Import & Exploration":
    from views import data_import
    data_import.render()

elif page == "📈 Performance Canaux":
    from views import channel_performance
    channel_performance.render()

elif page == "⚙️ Configuration Modèle":
    from views import model_config
    model_config.render()

elif page == "🔬 Résultats & Diagnostics":
    from views import results
    results.render()

elif page == "💰 Optimisation Budget":
    from views import optimization
    optimization.render()

elif page == "📋 Planification Budget":
    from views import budget_planner
    budget_planner.render()
