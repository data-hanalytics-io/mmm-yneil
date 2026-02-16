"""Page de configuration du modèle Meridian."""
import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processing import (
    CHANNEL_CATEGORIES, pivot_to_meridian_wide, format_number
)
from utils.meridian_wrapper import (
    ModelConfig, try_import_meridian, build_meridian_input,
    train_meridian_model, extract_results, generate_simulated_results,
    load_model_bundle,
)


def render():
    st.markdown("## ⚙️ Configuration du Modèle Meridian")

    if 'df_clean' not in st.session_state:
        st.warning("⚠️ Chargez d'abord vos données dans **Import & Exploration**.")
        return

    df = st.session_state['df_clean']

    # ── Charger un modèle pré-entraîné ──
    with st.expander("📦 Charger un modèle pré-entraîné (depuis Colab)", expanded=False):
        st.markdown(
            "Si vous avez entraîné le modèle dans **Google Colab**, "
            "uploadez le fichier `.pkl` généré."
        )
        uploaded_model = st.file_uploader(
            "Fichier de résultats (.pkl)",
            type=["pkl"],
            key="model_bundle_upload",
        )
        if uploaded_model is not None:
            try:
                import tempfile, os
                tmp_path = os.path.join(tempfile.gettempdir(), uploaded_model.name)
                with open(tmp_path, "wb") as f:
                    f.write(uploaded_model.getbuffer())

                results, config_loaded, meta = load_model_bundle(tmp_path)
                os.remove(tmp_path)

                st.session_state["model_results"] = results
                st.session_state["model_config"] = config_loaded
                st.session_state["model_trained"] = True
                st.session_state["model_configured"] = True

                st.success("Modèle chargé avec succès !")
                st.markdown(f"""
                | Info | Valeur |
                |---|---|
                | **Date d'entraînement** | {meta.get('created_at', 'N/A')[:19]} |
                | **Version Meridian** | {meta.get('meridian_version', 'N/A')} |
                | **Durée** | {meta.get('training_duration_seconds', 0):.0f}s |
                | **Simulé** | {'Oui' if results.is_simulated else 'Non'} |
                | **R²** | {results.r_squared:.4f} |
                | **Canaux** | {len(results.roi_by_channel)} |
                """)
                st.info("Allez dans **Résultats & Diagnostics** pour visualiser.")
                return
            except Exception as e:
                st.error(f"Erreur lors du chargement: {e}")

    # ── Check Meridian installation ──
    meridian = try_import_meridian()
    if meridian is None:
        st.warning(
            "⚠️ **Google Meridian n'est pas installé** dans cet environnement. "
            "L'app fonctionnera en **mode simulation** avec des résultats réalistes "
            "basés sur vos données réelles.\n\n"
            "Pour installer Meridian: `pip install google-meridian`"
        )
        simulation_mode = True
    else:
        st.success("✅ Google Meridian détecté !")
        simulation_mode = False

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════
    # SECTION 1: Sélection des canaux
    # ══════════════════════════════════════════════════════════════
    st.markdown("### 1️⃣ Sélection des canaux")

    present_channels = set(df['channel_grouping'].unique())

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**💰 Paid Media** (ROI calculé)")
        paid_options = [
            ch for ch, cfg in CHANNEL_CATEGORIES.items()
            if cfg['category'] == 'paid_media' and ch in present_channels
        ]
        selected_paid = []
        for ch in paid_options:
            ch_cost = df[df['channel_grouping'] == ch]['cost'].sum()
            default = ch_cost > 0
            label = f"{ch} ({format_number(ch_cost, suffix='€')})" if ch_cost > 0 else f"{ch} (no spend)"
            if st.checkbox(label, value=default, key=f"paid_{ch}"):
                selected_paid.append(ch)

    with col2:
        st.markdown("**🌱 Organic Media**")
        organic_options = [
            ch for ch, cfg in CHANNEL_CATEGORIES.items()
            if cfg['category'] == 'organic_media' and ch in present_channels
        ]
        selected_organic = []
        for ch in organic_options:
            ch_sessions = df[df['channel_grouping'] == ch]['sessions'].sum()
            if st.checkbox(f"{ch} ({format_number(ch_sessions)} sessions)",
                          value=True, key=f"org_{ch}"):
                selected_organic.append(ch)

    with col3:
        st.markdown("**🎛️ Contrôles**")
        control_options = [
            ch for ch, cfg in CHANNEL_CATEGORIES.items()
            if cfg['category'] == 'control' and ch in present_channels
        ]
        selected_controls = []
        for ch in control_options:
            if st.checkbox(ch, value=True, key=f"ctrl_{ch}"):
                selected_controls.append(ch)

        st.markdown("---")
        st.markdown("**📊 Contrôles additionnels**")
        st.caption("Recommandés si disponibles:")
        st.markdown("""
        - ⬜ Google Query Volume (GQV)
        - ⬜ Saisonnalité / Jours fériés
        - ⬜ Indicateurs macro-économiques
        """)

    # ══════════════════════════════════════════════════════════════
    # SECTION 2: KPI & Paramètres du modèle
    # ══════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 2️⃣ Paramètres du modèle")

    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown("**KPI**")
        kpi_col = st.selectbox("Variable cible", ['revenue', 'transactions'],
                               help="Revenue recommandé pour le ROAS. Transactions si KPI non-revenue.")
        kpi_type = 'revenue' if kpi_col == 'revenue' else 'non_revenue'

    with p2:
        st.markdown("**Adstock & Saturation**")
        max_lag = st.slider("Max Lag (semaines)", 1, 16, 8,
                            help="Durée max de l'effet résiduel des pubs")
        hill_before = st.checkbox("Hill avant Adstock",
                                   help="Appliquer la saturation avant le carryover (défaut: après)")

    with p3:
        st.markdown("**Tendance temporelle**")
        n_weeks = df['date'].nunique()
        suggested = max(2, min(n_weeks // 12, 10))
        n_knots = st.slider("Nombre de knots", 2, 15, suggested,
                            help=f"~1 knot par 10-15 semaines ({n_weeks} semaines dans les données)")

    p4, p5, p6 = st.columns(3)

    with p4:
        st.markdown("**Priors**")
        prior_type = st.selectbox(
            "Type de prior média",
            ['roi', 'coefficient'],
            help="'roi' = prior sur le ROI (recommandé). 'coefficient' = prior sur les coefficients bruts."
        )

    with p5:
        st.markdown("**Distribution effets**")
        effects_dist = st.selectbox(
            "Distribution des effets media",
            ['log_normal', 'normal'],
            help="log_normal (défaut) assure des effets positifs."
        )

    with p6:
        st.markdown("**MCMC Sampling**")
        n_chains = st.number_input("Chaînes", 1, 8, 4)
        n_adapt = st.number_input("Adaptation", 100, 2000, 500, step=100,
                                   help="Nombre de draws d'adaptation par chaîne")
        n_burnin = st.number_input("Burn-in", 50, 1000, 100, step=50,
                                    help="Nombre de draws de burn-in après adaptation")
        n_keep = st.number_input("Samples à garder", 100, 2000, 500, step=100,
                                  help="Nombre de draws gardés pour l'inférence")

    # ══════════════════════════════════════════════════════════════
    # SECTION 3: Priors ROI personnalisés
    # ══════════════════════════════════════════════════════════════
    st.markdown("---")
    with st.expander("### 3️⃣ Priors ROI personnalisés (optionnel)", expanded=False):
        st.markdown("""
        Ajustez les priors de ROI si vous avez des résultats d'experiments
        (A/B tests, geo-lift) ou des benchmarks sectoriels.
        Distribution: LogNormal(μ, σ) — μ = centre, σ = incertitude.
        """)

        custom_priors = {}
        for ch in selected_paid:
            meridian_name = CHANNEL_CATEGORIES[ch]['meridian']
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                st.markdown(f"**{ch}**")
            with c2:
                mu = st.number_input(f"μ (ROI moyen)", 0.0, 20.0, 2.0,
                                     step=0.5, key=f"prior_mu_{ch}")
            with c3:
                sigma = st.number_input(f"σ (incertitude)", 0.1, 5.0, 1.0,
                                        step=0.1, key=f"prior_sigma_{ch}")
            custom_priors[meridian_name] = {'mu': mu, 'sigma': sigma}

    # ══════════════════════════════════════════════════════════════
    # SECTION 4: Résumé & Lancement
    # ══════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 4️⃣ Résumé de la configuration")

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:
        st.markdown(f"""
        | Paramètre | Valeur |
        |---|---|
        | **KPI** | `{kpi_col}` ({kpi_type}) |
        | **Paid media** | {len(selected_paid)} canaux |
        | **Organic media** | {len(selected_organic)} canaux |
        | **Contrôles** | {len(selected_controls)} variables |
        | **Max lag** | {max_lag} semaines |
        | **Knots** | {n_knots} |
        """)

    with summary_col2:
        n_geos = df['country'].nunique()
        n_time = df['date'].nunique()
        total_dp = n_geos * n_time
        total_effects = (len(selected_paid) * n_geos) + (len(selected_organic) * n_geos) + n_knots
        ratio = total_dp / max(total_effects, 1)

        st.markdown(f"""
        | Métrique | Valeur |
        |---|---|
        | **Data points** | {total_dp:,} ({n_geos} geos × {n_time} semaines) |
        | **Effets à estimer** | ~{total_effects:,} |
        | **Ratio dp/effets** | {ratio:.1f} {'✅' if ratio > 8 else '⚠️'} |
        | **MCMC** | {n_chains} chaînes × {n_keep} samples |
        | **Mode** | {'🔬 Meridian' if not simulation_mode else '🎭 Simulation'} |
        """)

    # Sauvegarder la config
    config = ModelConfig(
        kpi_column=kpi_col,
        kpi_type=kpi_type,
        max_lag=max_lag,
        knots=n_knots,
        n_chains=n_chains,
        n_adapt=n_adapt,
        n_burnin=n_burnin,
        n_keep=n_keep,
        hill_before_adstock=hill_before,
        paid_media_prior_type=prior_type,
        media_effects_dist=effects_dist,
        paid_channels=[CHANNEL_CATEGORIES[ch]['meridian'] for ch in selected_paid],
        organic_channels=[CHANNEL_CATEGORIES[ch]['meridian'] for ch in selected_organic],
        control_columns=[CHANNEL_CATEGORIES[ch]['meridian'] for ch in selected_controls],
        custom_roi_priors=custom_priors,
    )
    st.session_state['model_config'] = config
    st.session_state['model_configured'] = True

    # ── Bouton de lancement ──
    st.markdown("---")

    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        run_model = st.button(
            "🚀 Lancer le modèle" if not simulation_mode else "🎭 Lancer la simulation",
            type="primary",
            use_container_width=True,
        )

    with col_info:
        if simulation_mode:
            st.caption(
                "En mode simulation, les résultats sont générés à partir d'heuristiques "
                "basées sur vos données réelles. Installez Meridian pour l'entraînement MCMC."
            )
        else:
            st.caption(
                f"⏱️ Estimation: {n_chains * n_keep * 0.01:.0f}-{n_chains * n_keep * 0.05:.0f} "
                f"minutes sur GPU T4. Beaucoup plus long sur CPU."
            )

    if run_model:
        # Pivoter les données
        with st.spinner("🔄 Transformation des données en format Meridian..."):
            wide_df = pivot_to_meridian_wide(df)
            st.session_state['wide_df'] = wide_df

        if simulation_mode:
            with st.spinner("🎭 Génération des résultats simulés..."):
                results = generate_simulated_results(wide_df, config)
                st.session_state['model_results'] = results
                st.session_state['model_trained'] = True
                st.success("✅ Simulation terminée ! Allez dans **Résultats & Diagnostics**.")
        else:
            try:
                with st.spinner("📦 Construction de l'InputData Meridian..."):
                    input_data = build_meridian_input(wide_df, config, meridian)

                with st.spinner("🔬 Entraînement MCMC en cours... (peut prendre plusieurs minutes)"):
                    mmm = train_meridian_model(input_data, config, meridian)
                    st.session_state['mmm_model'] = mmm

                with st.spinner("📊 Extraction des résultats..."):
                    results = extract_results(mmm, meridian)
                    st.session_state['model_results'] = results
                    st.session_state['model_trained'] = True

                st.success("✅ Modèle entraîné ! Allez dans **Résultats & Diagnostics**.")
            except Exception as e:
                st.error(f"Erreur lors de l'entraînement: {e}")
                st.info("💡 Astuce: vérifiez que vous avez un GPU disponible et assez de RAM.")

    # ── Export de la config ──
    with st.expander("📋 Exporter la configuration"):
        st.json({
            'kpi': kpi_col,
            'kpi_type': kpi_type,
            'max_lag': max_lag,
            'knots': n_knots,
            'n_chains': n_chains,
            'n_adapt': n_adapt,
            'n_burnin': n_burnin,
            'n_keep': n_keep,
            'prior_type': prior_type,
            'paid_channels': selected_paid,
            'organic_channels': selected_organic,
            'controls': selected_controls,
        })
