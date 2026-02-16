"""Page d'accueil — Vue d'ensemble de la pipeline MMM."""
import streamlit as st


def render():
    # Hero
    st.markdown("""
    <div class="hero">
        <h1><span class="yurineil-logo" style="font-size:2.4rem;">Yuri & Neil</span> · MMM Studio</h1>
        <p>
            Plateforme d'analyse Marketing Mix Modeling propulsée par
            Google Meridian. Analysez vos performances, modélisez l'impact
            de chaque levier et optimisez votre allocation budgétaire.
        </p>
        <div class="version">Meridian MMM Framework · v1.0</div>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline overview
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 🔄 Pipeline en 6 étapes")

        steps = [
            ("1", "Import & Exploration", "Chargez vos données marketing et explorez les métriques clés par pays et canal.", "📁"),
            ("2", "Analyse des Canaux", "Comparez les performances (ROAS, contributions) et identifiez les patterns.", "📈"),
            ("3", "Configuration du Modèle", "Sélectionnez les canaux, ajustez les priors bayésiens et les hyperparamètres.", "⚙️"),
            ("4", "Résultats & Diagnostics", "Évaluez le fit du modèle, les courbes de réponse et le ROI incrémental.", "🔬"),
            ("5", "Optimisation Budget", "Simulez des scénarios de réallocation pour maximiser le ROI global.", "💰"),
            ("6", "Planification Budget", "Saisissez un budget et une période pour obtenir la ventilation optimale par canal.", "📋"),
        ]

        for num, title, desc, icon in steps:
            state_key = ['data_loaded', 'eda_done', 'model_configured',
                         'model_trained', 'optimization_done', 'budget_planning_done'][int(num) - 1]
            done = st.session_state.get(state_key, False)
            status_class = 'done' if done else ''

            st.markdown(f"""
            <div class="pipeline-step {status_class}">
                <div class="step-num">{icon}</div>
                <div>
                    <strong>{title}</strong><br>
                    <span style="color:#64748b; font-size:0.85rem;">{desc}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 📖 Concepts clés")

        with st.expander("Marketing Mix Modeling", expanded=False):
            st.markdown("""
            Le MMM est une technique d'analyse statistique qui mesure l'impact
            des campagnes marketing sur les ventes ou KPIs. Il utilise des données
            agrégées (pas de cookies) et est donc privacy-safe.
            """)

        with st.expander("Meridian vs LightweightMMM"):
            st.markdown("""
            Meridian est le successeur de LightweightMMM de Google.
            Il apporte le modèle géo-hiérarchique, le support Reach & Frequency,
            et l'intégration de Google Query Volume pour débiaser Paid Search.
            """)

        with st.expander("Bayesian Inference"):
            st.markdown("""
            Meridian utilise l'inférence bayésienne (MCMC/NUTS) pour estimer
            les paramètres du modèle. Cela permet de quantifier l'incertitude
            et d'intégrer des connaissances a priori (priors) sur les ROI.
            """)

        with st.expander("Adstock & Hill Transform"):
            st.markdown("""
            **Adstock** modélise l'effet résiduel (carryover) des publicités
            dans le temps. **Hill** modélise la saturation : l'efficacité
            marginale diminue quand on augmente les dépenses.
            """)

        st.markdown("---")
        st.markdown("### 🔗 Ressources")
        st.markdown("""
        - [📘 Documentation Meridian](https://developers.google.com/meridian)
        - [💻 GitHub Meridian](https://github.com/google/meridian)
        - [💬 Discord Communauté](https://discord.gg/PU7Prfjc8U)
        """)

    # Quick data status
    st.markdown("---")
    if 'df_clean' in st.session_state:
        df = st.session_state['df_clean']
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Geos", df['country'].nunique())
        c2.metric("Semaines", df['date'].nunique())
        c3.metric("Canaux", df['channel_grouping'].nunique())
        c4.metric("Revenue total", f"{df['revenue'].sum():,.0f} €")
        c5.metric("Spend total", f"{df['cost'].sum():,.0f} €")
    else:
        st.info("👈 Commencez par **Import & Exploration** pour charger vos données.")
