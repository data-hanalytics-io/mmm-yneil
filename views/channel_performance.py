"""Page d'analyse détaillée des performances par canal."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.data_processing import (
    CHANNEL_CATEGORIES, GEO_NAMES, compute_channel_geo_matrix, format_number
)


def render():
    st.markdown("## 📈 Performance des Canaux")

    if 'df_clean' not in st.session_state:
        st.warning("⚠️ Chargez d'abord vos données dans **Import & Exploration**.")
        return

    df = st.session_state['df_clean']

    # ── Filtres ──
    with st.container():
        f1, f2, f3 = st.columns(3)
        with f1:
            selected_geos = st.multiselect(
                "Pays", sorted(df['country'].unique()),
                default=sorted(df['country'].unique()),
            )
        with f2:
            categories = ['paid_media', 'organic_media', 'control']
            selected_cats = st.multiselect(
                "Catégories", categories, default=['paid_media', 'organic_media'],
                format_func=lambda x: {'paid_media': '💰 Paid', 'organic_media': '🌱 Organic', 'control': '🎛️ Control'}[x],
            )
        with f3:
            metric_choice = st.selectbox("Métrique principale", ['revenue', 'transactions', 'sessions'])

    # Filtrer
    mapped_channels = [ch for ch, cfg in CHANNEL_CATEGORIES.items()
                       if cfg['category'] in selected_cats]
    dff = df[
        (df['country'].isin(selected_geos)) &
        (df['channel_grouping'].isin(mapped_channels))
    ].copy()

    if len(dff) == 0:
        st.warning("Aucune donnée avec ces filtres.")
        return

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════
    # SECTION 1: Vue d'ensemble Spend vs Revenue
    # ══════════════════════════════════════════════════════════════
    st.markdown("### 💰 Allocation Spend vs Revenue")

    ch_agg = dff.groupby('channel_grouping').agg(
        revenue=('revenue', 'sum'),
        cost=('cost', 'sum'),
        sessions=('sessions', 'sum'),
        transactions=('transactions', 'sum'),
        impressions=('impressions', 'sum'),
    ).reset_index()
    ch_agg['roas'] = np.where(ch_agg['cost'] > 0, ch_agg['revenue'] / ch_agg['cost'], 0)
    ch_agg['category'] = ch_agg['channel_grouping'].map(
        lambda x: CHANNEL_CATEGORIES.get(x, {}).get('category', 'unknown')
    )

    c1, c2 = st.columns(2)

    with c1:
        # Treemap des dépenses
        paid_only = ch_agg[ch_agg['cost'] > 0].copy()
        if len(paid_only) > 0:
            fig_tree = px.treemap(
                paid_only, path=['category', 'channel_grouping'],
                values='cost', color='roas',
                color_continuous_scale='RdYlGn',
                color_continuous_midpoint=2,
                title='Répartition du Spend (couleur = ROAS)',
            )
            fig_tree.update_layout(height=450)
            st.plotly_chart(fig_tree, use_container_width=True)

    with c2:
        # Treemap des revenus
        rev_data = ch_agg[ch_agg['revenue'] > 0].copy()
        if len(rev_data) > 0:
            fig_tree_rev = px.treemap(
                rev_data, path=['category', 'channel_grouping'],
                values='revenue',
                color='category',
                color_discrete_map={
                    'paid_media': '#6366f1', 'organic_media': '#22c55e', 'control': '#94a3b8'
                },
                title='Répartition du Revenue par canal',
            )
            fig_tree_rev.update_layout(height=450)
            st.plotly_chart(fig_tree_rev, use_container_width=True)

    # ══════════════════════════════════════════════════════════════
    # SECTION 2: ROAS par canal et pays
    # ══════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 📊 ROAS par Canal × Pays")

    roas_matrix = dff[dff['cost'] > 0].pivot_table(
        index='channel_grouping', columns='country',
        values=['revenue', 'cost'], aggfunc='sum', fill_value=0,
    )

    if len(roas_matrix) > 0:
        roas_vals = pd.DataFrame(index=roas_matrix.index)
        for geo in selected_geos:
            if geo in roas_matrix['revenue'].columns:
                rev = roas_matrix[('revenue', geo)]
                cost = roas_matrix[('cost', geo)]
                roas_vals[geo] = np.where(cost > 0, rev / cost, 0)

        if len(roas_vals.columns) > 0:
            fig_roas_heat = px.imshow(
                roas_vals.round(2),
                color_continuous_scale='RdYlGn',
                color_continuous_midpoint=2,
                aspect='auto',
                title='ROAS par Canal × Pays',
                labels={'color': 'ROAS'},
                text_auto='.1f',
            )
            fig_roas_heat.update_layout(height=max(300, len(roas_vals) * 40))
            st.plotly_chart(fig_roas_heat, use_container_width=True)

    # ══════════════════════════════════════════════════════════════
    # SECTION 3: Analyse par canal individuel
    # ══════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 🔎 Analyse détaillée par canal")

    selected_channel = st.selectbox(
        "Sélectionnez un canal",
        sorted(dff['channel_grouping'].unique()),
    )

    ch_data = dff[dff['channel_grouping'] == selected_channel]
    ch_config = CHANNEL_CATEGORIES.get(selected_channel, {})

    c1, c2, c3, c4 = st.columns(4)
    ch_rev = ch_data['revenue'].sum()
    ch_cost = ch_data['cost'].sum()
    ch_tx = ch_data['transactions'].sum()
    ch_roas = ch_rev / ch_cost if ch_cost > 0 else 0

    c1.metric("Revenue", format_number(ch_rev, suffix=' €'))
    c2.metric("Spend", format_number(ch_cost, suffix=' €'))
    c3.metric("ROAS", f"{ch_roas:.2f}x")
    c4.metric("Transactions", format_number(ch_tx))

    col_a, col_b = st.columns(2)

    with col_a:
        # Revenue par pays pour ce canal
        geo_ch = ch_data.groupby('country').agg(
            revenue=('revenue', 'sum'),
            cost=('cost', 'sum'),
        ).reset_index()
        geo_ch['roas'] = np.where(geo_ch['cost'] > 0, geo_ch['revenue'] / geo_ch['cost'], 0)

        fig_ch_geo = px.bar(
            geo_ch.sort_values('revenue', ascending=True),
            x='revenue', y='country', orientation='h',
            color='roas', color_continuous_scale='RdYlGn',
            title=f'{selected_channel} — Revenue par pays',
        )
        fig_ch_geo.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_ch_geo, use_container_width=True)

    with col_b:
        # Évolution temporelle
        ts_ch = ch_data.groupby('date').agg(
            revenue=('revenue', 'sum'),
            cost=('cost', 'sum'),
        ).reset_index().sort_values('date')

        fig_ts_ch = go.Figure()
        fig_ts_ch.add_trace(go.Bar(x=ts_ch['date'], y=ts_ch['revenue'],
                                    name='Revenue', marker_color='#6366f1'))
        if ch_cost > 0:
            fig_ts_ch.add_trace(go.Bar(x=ts_ch['date'], y=ts_ch['cost'],
                                        name='Spend', marker_color='#f97316'))
        fig_ts_ch.update_layout(
            title=f'{selected_channel} — Évolution temporelle',
            barmode='group', height=400,
        )
        st.plotly_chart(fig_ts_ch, use_container_width=True)

    # Info Meridian
    if ch_config:
        st.info(
            f"**Mapping Meridian**: {ch_config.get('category', '?')} → "
            f"`{ch_config.get('meridian', '?')}` · "
            f"Métrique: {ch_config.get('metric', '?')} "
            f"(fallback: {ch_config.get('fallback', '?')})"
        )

    # ══════════════════════════════════════════════════════════════
    # SECTION 4: Corrélations
    # ══════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 🔗 Corrélations entre canaux")

    # Matrice de corrélation revenue par canal × semaine
    rev_pivot = dff.pivot_table(
        index=['country', 'date'], columns='channel_grouping',
        values='revenue', aggfunc='sum', fill_value=0,
    )

    if rev_pivot.shape[1] > 2:
        corr = rev_pivot.corr()
        fig_corr = px.imshow(
            corr.round(2),
            color_continuous_scale='RdBu_r',
            color_continuous_midpoint=0,
            aspect='auto',
            title='Corrélation des revenues entre canaux',
            text_auto='.2f',
        )
        fig_corr.update_layout(height=max(400, len(corr) * 35))
        st.plotly_chart(fig_corr, use_container_width=True)

        st.caption(
            "💡 Des corrélations élevées entre canaux payants peuvent indiquer "
            "un risque de multicolinéarité dans le modèle MMM. Meridian gère cela "
            "via les priors bayésiens, mais c'est important d'en être conscient."
        )
