"""Page Résultats & Diagnostics du modèle."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.data_processing import format_number


def render():
    st.markdown("## 🔬 Résultats & Diagnostics")

    if 'model_results' not in st.session_state:
        st.warning(
            "⚠️ Aucun modèle entraîné. Allez dans **Configuration Modèle** "
            "pour lancer l'entraînement."
        )
        return

    results = st.session_state['model_results']

    if results.is_simulated:
        st.info(
            "🎭 **Mode Simulation** — Les résultats ci-dessous sont simulés "
            "à partir d'heuristiques basées sur vos données réelles. "
            "Installez Google Meridian pour des résultats bayésiens réels."
        )

    # ══════════════════════════════════════════════════════════════
    # SECTION 1: Model Fit
    # ══════════════════════════════════════════════════════════════
    st.markdown("### 📐 Qualité du modèle (Model Fit)")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R²", f"{results.r_squared:.4f}",
              help="Proportion de variance expliquée. > 0.85 = bon")
    m2.metric("MAPE", f"{results.mape:.2%}",
              help="Mean Absolute Percentage Error. < 15% = bon")
    m3.metric("wMAPE", f"{results.wmape:.2%}",
              help="Weighted MAPE. < 10% = excellent")

    r2 = results.r_squared
    if r2 > 0.90:
        m4.markdown('<span class="badge badge-green">Excellent Fit</span>', unsafe_allow_html=True)
    elif r2 > 0.80:
        m4.markdown('<span class="badge badge-amber">Bon Fit</span>', unsafe_allow_html=True)
    else:
        m4.markdown('<span class="badge badge-red">Fit à améliorer</span>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # SECTION 2: ROI par canal
    # ══════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 📊 ROI Incrémental par Canal")

    if results.roi_by_channel:
        roi_df = pd.DataFrame(results.roi_by_channel).T.reset_index()
        roi_df.columns = ['channel'] + list(roi_df.columns[1:])
        roi_df = roi_df.sort_values('roi_mean', ascending=True)

        # Waterfall-style ROI chart avec intervalle de confiance
        fig_roi = go.Figure()

        colors = ['#6366f1' if r > 1 else '#f97316' for r in roi_df['roi_mean']]

        fig_roi.add_trace(go.Bar(
            y=roi_df['channel'],
            x=roi_df['roi_mean'],
            orientation='h',
            marker_color=colors,
            name='ROI moyen',
            text=roi_df['roi_mean'].apply(lambda x: f'{x:.2f}x'),
            textposition='outside',
        ))

        # Error bars (CI)
        if 'roi_ci_lo' in roi_df.columns and 'roi_ci_hi' in roi_df.columns:
            fig_roi.add_trace(go.Scatter(
                y=roi_df['channel'],
                x=roi_df['roi_ci_lo'],
                mode='markers', marker=dict(symbol='line-ns', size=12, color='#94a3b8'),
                name='CI bas', showlegend=False,
            ))
            fig_roi.add_trace(go.Scatter(
                y=roi_df['channel'],
                x=roi_df['roi_ci_hi'],
                mode='markers', marker=dict(symbol='line-ns', size=12, color='#94a3b8'),
                name='CI haut', showlegend=False,
            ))

        # Ligne ROAS = 1
        fig_roi.add_vline(x=1, line_dash="dash", line_color="#dc2626",
                          annotation_text="Break-even (1.0)")

        fig_roi.update_layout(
            title='ROI Incrémental par Canal (avec intervalle de confiance)',
            xaxis_title='ROI (€ revenue / € spent)',
            height=max(350, len(roi_df) * 55),
            showlegend=False,
        )
        st.plotly_chart(fig_roi, use_container_width=True)

        # Table ROI détaillée
        with st.expander("📋 Table ROI détaillée"):
            display_df = roi_df.copy()
            display_df['spend'] = display_df.get('spend', 0)
            display_df['contribution_pct'] = display_df.get('contribution_pct', 0)
            st.dataframe(
                display_df.sort_values('roi_mean', ascending=False).style.format({
                    'roi_mean': '{:.2f}x',
                    'roi_ci_lo': '{:.2f}x',
                    'roi_ci_hi': '{:.2f}x',
                    'spend': '€{:,.0f}',
                    'contribution_pct': '{:.1f}%',
                }),
                use_container_width=True, hide_index=True,
            )

    # ══════════════════════════════════════════════════════════════
    # SECTION 3: Contribution & Décomposition
    # ══════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 🍩 Décomposition du Revenue")

    c1, c2 = st.columns(2)

    with c1:
        if results.contribution_by_channel:
            contrib_data = pd.DataFrame([
                {'channel': k, 'contribution': v}
                for k, v in results.contribution_by_channel.items()
            ])
            # Ajouter baseline
            total_rev = sum(results.contribution_by_channel.values())
            if results.time_decomposition is not None and 'baseline' in results.time_decomposition.columns:
                baseline = results.time_decomposition['baseline'].sum()
                contrib_data = pd.concat([
                    pd.DataFrame([{'channel': 'Baseline', 'contribution': baseline}]),
                    contrib_data,
                ])

            fig_pie = px.pie(
                contrib_data, values='contribution', names='channel',
                title='Contribution au Revenue',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_pie.update_layout(height=420)
            st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        # Spend share vs Revenue share
        if results.roi_by_channel:
            share_data = []
            total_spend = sum(v.get('spend', 0) for v in results.roi_by_channel.values())
            total_contrib = sum(results.contribution_by_channel.values()) if results.contribution_by_channel else 1

            for ch, roi_info in results.roi_by_channel.items():
                spend = roi_info.get('spend', 0)
                contrib = results.contribution_by_channel.get(ch, 0)
                share_data.append({
                    'channel': ch,
                    'spend_share': spend / max(total_spend, 1) * 100,
                    'revenue_share': contrib / max(total_contrib, 1) * 100,
                })

            share_df = pd.DataFrame(share_data)

            fig_share = go.Figure()
            fig_share.add_trace(go.Bar(
                y=share_df['channel'], x=share_df['spend_share'],
                name='% du Spend', marker_color='#f97316', orientation='h',
            ))
            fig_share.add_trace(go.Bar(
                y=share_df['channel'], x=share_df['revenue_share'],
                name='% du Revenue', marker_color='#6366f1', orientation='h',
            ))
            fig_share.update_layout(
                title='Spend Share vs Revenue Share',
                barmode='group', height=420,
                xaxis_title='%',
            )
            st.plotly_chart(fig_share, use_container_width=True)

    # ══════════════════════════════════════════════════════════════
    # SECTION 4: Time Decomposition
    # ══════════════════════════════════════════════════════════════
    if results.time_decomposition is not None:
        st.markdown("---")
        st.markdown("### 📅 Décomposition temporelle")

        decomp = results.time_decomposition
        cols_to_stack = [c for c in decomp.columns if c != 'time']

        fig_decomp = go.Figure()
        colors = ['#94a3b8', '#6366f1', '#8b5cf6', '#a855f7', '#d946ef', '#ec4899',
                  '#f43f5e', '#f97316', '#eab308']

        def _hex_to_rgba(hex_color, alpha=0.5):
            h = hex_color.lstrip('#')
            return f'rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{alpha})'

        for i, col in enumerate(cols_to_stack):
            fig_decomp.add_trace(go.Scatter(
                x=decomp['time'], y=decomp[col],
                name=col.replace('_', ' ').title(),
                stackgroup='one',
                fillcolor=_hex_to_rgba(colors[i % len(colors)], 0.5),
                line=dict(color=colors[i % len(colors)], width=0.5),
            ))

        fig_decomp.update_layout(
            title='Décomposition du Revenue dans le temps (stacked area)',
            height=450,
            xaxis_title='Semaine',
            yaxis_title='Revenue (€)',
            legend=dict(orientation='h', y=-0.15),
        )
        st.plotly_chart(fig_decomp, use_container_width=True)

    # ══════════════════════════════════════════════════════════════
    # SECTION 5: Response Curves
    # ══════════════════════════════════════════════════════════════
    if results.response_curves:
        st.markdown("---")
        st.markdown("### 📈 Courbes de réponse (Response Curves)")
        st.caption(
            "La courbe de réponse montre comment le revenue incrémental évolue "
            "en fonction du spend. La pente décroissante reflète la saturation."
        )

        # Sélection du canal
        rc_channels = list(results.response_curves.keys())
        n_cols = min(3, len(rc_channels))

        for i in range(0, len(rc_channels), n_cols):
            cols = st.columns(n_cols)
            for j, col in enumerate(cols):
                idx = i + j
                if idx >= len(rc_channels):
                    break
                ch = rc_channels[idx]
                rc_data = results.response_curves[ch]

                with col:
                    fig_rc = go.Figure()
                    fig_rc.add_trace(go.Scatter(
                        x=rc_data['spend'], y=rc_data['incremental_revenue'],
                        mode='lines', fill='tozeroy',
                        line=dict(color='#6366f1', width=2),
                        fillcolor='rgba(99,102,241,0.1)',
                    ))
                    # Point actuel
                    current_spend = results.roi_by_channel.get(ch, {}).get('spend', 0)
                    if current_spend > 0:
                        # Trouver le y correspondant
                        closest_idx = (rc_data['spend'] - current_spend).abs().idxmin()
                        current_rev = rc_data.loc[closest_idx, 'incremental_revenue']
                        fig_rc.add_trace(go.Scatter(
                            x=[current_spend], y=[current_rev],
                            mode='markers+text',
                            marker=dict(size=12, color='#f97316'),
                            text=['Actuel'],
                            textposition='top center',
                            showlegend=False,
                        ))

                    fig_rc.update_layout(
                        title=ch.replace('_', ' ').title(),
                        xaxis_title='Spend (€)',
                        yaxis_title='Revenue incrémental (€)',
                        height=300,
                        margin=dict(t=40, b=40),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_rc, use_container_width=True)

    # ══════════════════════════════════════════════════════════════
    # SECTION 6: Adstock Parameters
    # ══════════════════════════════════════════════════════════════
    if results.adstock_params:
        st.markdown("---")
        st.markdown("### ⏳ Paramètres Adstock (Carryover)")

        adstock_df = pd.DataFrame(results.adstock_params).T.reset_index()
        adstock_df.columns = ['channel', 'decay', 'peak_lag']

        c1, c2 = st.columns(2)

        with c1:
            fig_decay = px.bar(
                adstock_df.sort_values('decay', ascending=True),
                x='decay', y='channel', orientation='h',
                color='decay', color_continuous_scale='Viridis',
                title='Taux de Decay (persistence de l\'effet)',
                labels={'decay': 'Decay rate', 'channel': ''},
            )
            fig_decay.update_layout(height=350)
            st.plotly_chart(fig_decay, use_container_width=True)
            st.caption("Decay élevé = l'effet publicitaire persiste longtemps")

        with c2:
            fig_lag = px.bar(
                adstock_df.sort_values('peak_lag'),
                x='peak_lag', y='channel', orientation='h',
                color='peak_lag', color_continuous_scale='Plasma',
                title='Peak Lag (semaines avant effet max)',
                labels={'peak_lag': 'Peak lag (semaines)', 'channel': ''},
            )
            fig_lag.update_layout(height=350)
            st.plotly_chart(fig_lag, use_container_width=True)
            st.caption("Peak lag > 0 = l'effet maximal est retardé")
