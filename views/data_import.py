"""Page Import & Exploration des données."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.data_processing import (
    load_and_clean, compute_geo_summary, compute_channel_summary,
    compute_time_series, get_data_quality_report, GEO_NAMES,
    CHANNEL_CATEGORIES, EXCLUDED_CHANNELS, CATEGORY_LABELS, format_number,
)


def render():
    st.markdown("## 📁 Import & Exploration des Données")
    st.markdown("Chargez votre fichier de données marketing et explorez les métriques clés.")

    # ── Upload ──
    with st.expander("📤 Charger les données", expanded='df_clean' not in st.session_state):
        col_upload, col_format = st.columns([2, 1])

        with col_upload:
            uploaded = st.file_uploader(
                "Fichier CSV",
                type=['csv'],
                help="Format attendu: year, week, country, channel_grouping, sessions, revenue, cost, transactions, impressions, clicks"
            )

        with col_format:
            st.markdown("**Format attendu:**")
            st.code("year, week, country, channel_grouping,\nsessions, revenue, new_customers,\ncost, transactions, impressions, clicks", language=None)

            st.markdown("**Exemple:**")
            st.code('2026,6,"FR","Paid Search Brand",\n1208008,2347311,7492,\n34925,31598,2908736,1300895', language=None)

        if uploaded is not None:
            try:
                df_raw = pd.read_csv(uploaded)
                df_clean = load_and_clean(df_raw)
                st.session_state['df_raw'] = df_raw
                st.session_state['df_clean'] = df_clean
                st.session_state['data_loaded'] = True
                st.success(f"✅ {len(df_clean):,} lignes chargées après nettoyage")
            except Exception as e:
                st.error(f"Erreur: {e}")
                return

    # ── Si pas de données, on s'arrête ──
    if 'df_clean' not in st.session_state:
        st.info("☝️ Uploadez votre fichier CSV pour commencer l'analyse.")
        return

    df = st.session_state['df_clean']

    # ── KPIs globaux ──
    st.markdown("---")
    st.markdown("### 🎯 KPIs Globaux")

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    total_rev = df['revenue'].sum()
    total_cost = df['cost'].sum()
    total_tx = df['transactions'].sum()
    overall_roas = total_rev / total_cost if total_cost > 0 else 0

    k1.metric("Revenue", format_number(total_rev, suffix=' €'))
    k2.metric("Spend Media", format_number(total_cost, suffix=' €'))
    k3.metric("ROAS Global", f"{overall_roas:.2f}x")
    k4.metric("Transactions", format_number(total_tx))
    k5.metric("Pays", f"{df['country'].nunique()}")
    k6.metric("Semaines", f"{df['date'].nunique()}")

    # ── Tabs d'analyse ──
    st.markdown("---")
    tab_quality, tab_geo, tab_channels, tab_time, tab_raw = st.tabs([
        "🔍 Qualité", "🌍 Par Pays", "📊 Par Canal",
        "📅 Temporel", "📋 Données brutes"
    ])

    # ── Tab Qualité ──
    with tab_quality:
        report = get_data_quality_report(df)

        st.markdown("#### Checks de qualité pour Meridian")
        for check in report['quality_checks']:
            if check['status'] == 'ok':
                badge = '<span class="badge badge-green">OK</span>'
            elif check['status'] == 'warning':
                badge = '<span class="badge badge-amber">⚠ ATTENTION</span>'
            else:
                badge = '<span class="badge badge-red">❌ PROBLÈME</span>'

            st.markdown(
                f"{badge} **{check['name']}**: {check['value']} "
                f"(cible: {check['target']})",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("#### Couverture des canaux par catégorie Meridian")

        present_channels = set(df['channel_grouping'].unique())
        for category, label in CATEGORY_LABELS.items():
            channels_in_cat = [
                ch for ch, cfg in CHANNEL_CATEGORIES.items()
                if cfg['category'] == category
            ]
            present = [ch for ch in channels_in_cat if ch in present_channels]
            absent = [ch for ch in channels_in_cat if ch not in present_channels]

            st.markdown(f"**{label}** ({len(present)}/{len(channels_in_cat)})")
            for ch in present:
                ch_data = df[df['channel_grouping'] == ch]
                has_spend = ch_data['cost'].sum() > 0
                has_imp = ch_data['impressions'].sum() > 0
                st.markdown(
                    f"&nbsp;&nbsp; ✅ {ch} — "
                    f"{'💰 spend' if has_spend else '⬜ no spend'} · "
                    f"{'📊 impressions' if has_imp else '📊 sessions only'}"
                )
            for ch in absent:
                st.markdown(f"&nbsp;&nbsp; ⬜ {ch} — absent des données")

        # Canaux non mappés
        unmapped = present_channels - set(CHANNEL_CATEGORIES.keys()) - set(EXCLUDED_CHANNELS)
        if unmapped:
            st.markdown(f"\n**⚠ Canaux non mappés**: {', '.join(unmapped)}")

    # ── Tab Par Pays ──
    with tab_geo:
        geo_df = compute_geo_summary(df)

        # Barchart revenue par pays
        fig_geo = px.bar(
            geo_df.sort_values('revenue', ascending=True),
            x='revenue', y='country', orientation='h',
            color='roas', color_continuous_scale='Tealgrn',
            labels={'revenue': 'Revenue (€)', 'country': 'Pays', 'roas': 'ROAS'},
            title='Revenue par pays (couleur = ROAS)',
        )
        fig_geo.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_geo, use_container_width=True)

        # Table détaillée
        st.dataframe(
            geo_df[['geo_name', 'revenue', 'cost', 'roas', 'transactions',
                     'sessions', 'n_channels', 'n_weeks']].style.format({
                'revenue': '{:,.0f}', 'cost': '{:,.0f}',
                'roas': '{:.2f}x', 'transactions': '{:,.0f}',
                'sessions': '{:,.0f}',
            }),
            use_container_width=True, hide_index=True,
        )

    # ── Tab Par Canal ──
    with tab_channels:
        ch_df = compute_channel_summary(df)
        ch_df_paid = ch_df[ch_df['cost'] > 0].copy()

        c1, c2 = st.columns(2)

        with c1:
            fig_ch_rev = px.bar(
                ch_df.head(15).sort_values('revenue', ascending=True),
                x='revenue', y='channel_grouping', orientation='h',
                color='category',
                color_discrete_map={
                    'paid_media': '#6366f1', 'organic_media': '#22c55e',
                    'control': '#94a3b8', 'excluded': '#e2e8f0',
                },
                title='Top 15 canaux par Revenue',
                labels={'revenue': 'Revenue (€)', 'channel_grouping': ''},
            )
            fig_ch_rev.update_layout(height=500, showlegend=True,
                                      yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_ch_rev, use_container_width=True)

        with c2:
            if len(ch_df_paid) > 0:
                fig_roas = px.bar(
                    ch_df_paid.sort_values('roas', ascending=True),
                    x='roas', y='channel_grouping', orientation='h',
                    color='roas', color_continuous_scale='RdYlGn',
                    title='ROAS par canal payant',
                    labels={'roas': 'ROAS', 'channel_grouping': ''},
                )
                fig_roas.update_layout(height=500,
                                        yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_roas, use_container_width=True)

        # Scatter spend vs revenue
        if len(ch_df_paid) > 0:
            fig_scatter = px.scatter(
                ch_df_paid, x='cost', y='revenue',
                size='transactions', color='channel_grouping',
                hover_data=['roas', 'n_geos'],
                title='Spend vs Revenue (taille = transactions)',
                labels={'cost': 'Spend (€)', 'revenue': 'Revenue (€)'},
            )
            fig_scatter.add_trace(go.Scatter(
                x=[0, ch_df_paid['cost'].max()],
                y=[0, ch_df_paid['cost'].max()],
                mode='lines', line=dict(dash='dash', color='gray'),
                name='ROAS = 1',
            ))
            fig_scatter.update_layout(height=500)
            st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Tab Temporel ──
    with tab_time:
        ts = compute_time_series(df)

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Bar(
            x=ts['date'], y=ts['revenue'],
            name='Revenue', marker_color='#6366f1', opacity=0.7,
        ))
        fig_ts.add_trace(go.Bar(
            x=ts['date'], y=ts['cost'],
            name='Spend', marker_color='#f97316', opacity=0.7,
        ))
        fig_ts.add_trace(go.Scatter(
            x=ts['date'], y=ts['roas'],
            name='ROAS', yaxis='y2',
            line=dict(color='#22c55e', width=3),
        ))
        fig_ts.update_layout(
            title='Revenue, Spend & ROAS par semaine',
            barmode='group', height=450,
            yaxis=dict(title='€'),
            yaxis2=dict(title='ROAS', overlaying='y', side='right'),
            legend=dict(orientation='h', y=-0.15),
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        # Heatmap pays × semaine
        st.markdown("#### Heatmap Revenue: Pays × Semaine")
        heat_data = df.pivot_table(
            index='country', columns='date', values='revenue', aggfunc='sum', fill_value=0
        )
        heat_data = heat_data[sorted(heat_data.columns)]
        fig_heat = px.imshow(
            heat_data, aspect='auto', color_continuous_scale='Blues',
            labels={'x': 'Semaine', 'y': 'Pays', 'color': 'Revenue'},
        )
        fig_heat.update_layout(height=400)
        st.plotly_chart(fig_heat, use_container_width=True)

    # ── Tab Données brutes ──
    with tab_raw:
        st.markdown("#### Filtrer les données")
        c1, c2 = st.columns(2)
        with c1:
            geo_filter = st.multiselect("Pays", sorted(df['country'].unique()))
        with c2:
            ch_filter = st.multiselect("Canaux", sorted(df['channel_grouping'].unique()))

        filtered = df.copy()
        if geo_filter:
            filtered = filtered[filtered['country'].isin(geo_filter)]
        if ch_filter:
            filtered = filtered[filtered['channel_grouping'].isin(ch_filter)]

        st.dataframe(filtered, use_container_width=True, height=500)
        st.download_button(
            "📥 Télécharger les données filtrées",
            filtered.to_csv(index=False),
            file_name="mmm_data_filtered.csv",
            mime="text/csv",
        )

    st.session_state['eda_done'] = True
