"""Page Optimisation Budgétaire & Scénarios."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.data_processing import format_number


def render():
    st.markdown("## 💰 Optimisation & Scénarios Budgétaires")

    if 'model_results' not in st.session_state:
        st.warning(
            "⚠️ Aucun modèle entraîné. Allez dans **Configuration Modèle** "
            "pour lancer l'entraînement d'abord."
        )
        return

    results = st.session_state['model_results']

    if results.is_simulated:
        st.info("🎭 **Mode Simulation** — Résultats basés sur des heuristiques.")

    # ══════════════════════════════════════════════════════════════
    # SECTION 1: Optimisation à budget fixe
    # ══════════════════════════════════════════════════════════════
    st.markdown("### 1️⃣ Réallocation à budget fixe")
    st.caption(
        "Même budget total, réparti différemment pour maximiser le ROI global."
    )

    if results.optimized_budget:
        opt = results.optimized_budget
        opt_df = pd.DataFrame(opt).T.reset_index()
        opt_df.columns = ['channel'] + list(opt_df.columns[1:])

        total_current = opt_df['current_spend'].sum()
        total_optimized = opt_df['optimized_spend'].sum()

        # KPIs
        k1, k2, k3 = st.columns(3)
        k1.metric("Budget actuel", format_number(total_current, suffix=' €'))
        k2.metric("Budget optimisé", format_number(total_optimized, suffix=' €'),
                  delta=format_number(total_optimized - total_current, suffix=' €'))

        # ROI moyen pondéré
        avg_current_roi = np.average(
            [v['expected_roi'] for v in opt.values()],
            weights=[v['current_spend'] for v in opt.values()]
        ) if total_current > 0 else 0
        k3.metric("ROI pondéré estimé", f"{avg_current_roi:.2f}x")

        # Waterfall des changements
        opt_df_sorted = opt_df.sort_values('change_pct', ascending=True)

        fig_waterfall = go.Figure()
        colors = ['#16a34a' if x > 0 else '#dc2626' for x in opt_df_sorted['change_pct']]

        fig_waterfall.add_trace(go.Bar(
            y=opt_df_sorted['channel'],
            x=opt_df_sorted['change_pct'],
            orientation='h',
            marker_color=colors,
            text=opt_df_sorted['change_pct'].apply(lambda x: f'{x:+.1f}%'),
            textposition='outside',
        ))
        fig_waterfall.update_layout(
            title='Changement de budget recommandé par canal (%)',
            xaxis_title='Variation (%)',
            height=max(350, len(opt_df) * 50),
        )
        fig_waterfall.add_vline(x=0, line_color='#94a3b8')
        st.plotly_chart(fig_waterfall, use_container_width=True)

        # Comparaison côte à côte
        fig_compare = go.Figure()
        fig_compare.add_trace(go.Bar(
            y=opt_df['channel'], x=opt_df['current_spend'],
            name='Budget actuel', marker_color='#94a3b8', orientation='h',
        ))
        fig_compare.add_trace(go.Bar(
            y=opt_df['channel'], x=opt_df['optimized_spend'],
            name='Budget optimisé', marker_color='#6366f1', orientation='h',
        ))
        fig_compare.update_layout(
            title='Budget actuel vs optimisé par canal',
            barmode='group', height=max(350, len(opt_df) * 50),
            xaxis_title='Budget (€)',
        )
        st.plotly_chart(fig_compare, use_container_width=True)

    # ══════════════════════════════════════════════════════════════
    # SECTION 2: Simulator de scénarios
    # ══════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 2️⃣ Simulateur de scénarios")
    st.caption(
        "Ajustez manuellement les budgets par canal et observez l'impact estimé sur le revenue."
    )

    if results.roi_by_channel:
        channels = list(results.roi_by_channel.keys())

        # Slider pour le budget total
        current_total = sum(v.get('spend', 0) for v in results.roi_by_channel.values())
        budget_factor = st.slider(
            "Facteur de budget global",
            0.5, 2.0, 1.0, 0.05,
            help="1.0 = budget actuel, 1.5 = +50%, etc."
        )
        target_budget = current_total * budget_factor
        st.markdown(f"**Budget cible: {format_number(target_budget, suffix=' €')}** "
                    f"(actuel: {format_number(current_total, suffix=' €')})")

        st.markdown("---")

        # Sliders par canal
        channel_budgets = {}
        cols_per_row = 3
        channel_list = list(results.roi_by_channel.items())

        for i in range(0, len(channel_list), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx >= len(channel_list):
                    break
                ch, info = channel_list[idx]
                current = info.get('spend', 0)

                with col:
                    roi = info.get('roi_mean', 1)
                    roi_color = '#16a34a' if roi > 1.5 else ('#f97316' if roi > 1 else '#dc2626')

                    st.markdown(
                        f"**{ch.replace('_', ' ').title()}**<br>"
                        f"ROI: <span style='color:{roi_color};font-weight:700'>{roi:.2f}x</span>",
                        unsafe_allow_html=True,
                    )

                    new_budget = st.slider(
                        f"Budget {ch}",
                        min_value=0,
                        max_value=int(current * 3) if current > 0 else 10000,
                        value=int(current * budget_factor),
                        step=max(1, int(current * 0.05)),
                        key=f"budget_{ch}",
                        label_visibility="collapsed",
                    )
                    channel_budgets[ch] = new_budget

                    change = ((new_budget - current) / max(current, 1)) * 100
                    delta_color = '#16a34a' if change > 0 else '#dc2626'
                    st.markdown(
                        f"<span style='font-size:0.8rem;color:{delta_color}'>"
                        f"{change:+.0f}% vs actuel</span>",
                        unsafe_allow_html=True,
                    )

        # Calculer l'impact estimé
        st.markdown("---")
        st.markdown("### 📊 Impact estimé du scénario")

        scenario_total_spend = sum(channel_budgets.values())
        estimated_revenue = 0
        scenario_details = []

        for ch, new_spend in channel_budgets.items():
            info = results.roi_by_channel.get(ch, {})
            roi = info.get('roi_mean', 1)
            current_spend = info.get('spend', 0)

            # Modèle simplifié: ajuster pour la saturation (Hill)
            if current_spend > 0 and new_spend > 0:
                ratio = new_spend / current_spend
                # Saturation: rendements décroissants
                effective_ratio = ratio ** 0.7  # exposant < 1 = saturation
                ch_revenue = roi * current_spend * effective_ratio
            else:
                ch_revenue = roi * new_spend * 0.8  # Pénalité pour nouveau canal

            estimated_revenue += ch_revenue

            scenario_details.append({
                'Canal': ch.replace('_', ' ').title(),
                'Budget actuel': current_spend,
                'Budget scénario': new_spend,
                'Variation': f"{((new_spend - current_spend) / max(current_spend, 1) * 100):+.0f}%",
                'ROI': roi,
                'Revenue estimé': ch_revenue,
                'mROI estimé': ch_revenue / max(new_spend, 1),
            })

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Budget total scénario", format_number(scenario_total_spend, suffix=' €'),
                  delta=format_number(scenario_total_spend - current_total, suffix=' €'))
        s2.metric("Revenue estimé", format_number(estimated_revenue, suffix=' €'))
        s3.metric("ROAS global estimé",
                  f"{estimated_revenue / max(scenario_total_spend, 1):.2f}x")
        s4.metric("Efficiency gain",
                  f"{((estimated_revenue / max(scenario_total_spend, 1)) / max(current_total / max(sum(v.get('spend', 0) * v.get('roi_mean', 1) for v in results.roi_by_channel.values()), 1), 0.01) - 1) * 100:+.1f}%")

        # Table détaillée
        scenario_df = pd.DataFrame(scenario_details)
        st.dataframe(
            scenario_df.style.format({
                'Budget actuel': '€{:,.0f}',
                'Budget scénario': '€{:,.0f}',
                'ROI': '{:.2f}x',
                'Revenue estimé': '€{:,.0f}',
                'mROI estimé': '{:.2f}x',
            }).background_gradient(subset=['mROI estimé'], cmap='RdYlGn'),
            use_container_width=True, hide_index=True,
        )

        # Chart du scénario
        fig_scenario = go.Figure()
        fig_scenario.add_trace(go.Bar(
            x=scenario_df['Canal'],
            y=scenario_df['Budget actuel'],
            name='Budget actuel', marker_color='#94a3b8',
        ))
        fig_scenario.add_trace(go.Bar(
            x=scenario_df['Canal'],
            y=scenario_df['Budget scénario'],
            name='Budget scénario', marker_color='#6366f1',
        ))
        fig_scenario.add_trace(go.Scatter(
            x=scenario_df['Canal'],
            y=scenario_df['Revenue estimé'],
            name='Revenue estimé',
            yaxis='y2',
            mode='lines+markers',
            line=dict(color='#22c55e', width=3),
            marker=dict(size=10),
        ))
        fig_scenario.update_layout(
            title='Scénario budgétaire: Spend vs Revenue estimé',
            barmode='group', height=450,
            yaxis=dict(title='Budget (€)'),
            yaxis2=dict(title='Revenue estimé (€)', overlaying='y', side='right'),
            legend=dict(orientation='h', y=-0.2),
        )
        st.plotly_chart(fig_scenario, use_container_width=True)

    # ══════════════════════════════════════════════════════════════
    # SECTION 3: Export
    # ══════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 📥 Export des résultats")

    ec1, ec2, ec3 = st.columns(3)

    with ec1:
        if results.roi_by_channel:
            roi_export = pd.DataFrame(results.roi_by_channel).T
            st.download_button(
                "📊 ROI par canal (CSV)",
                roi_export.to_csv(),
                file_name="mmm_roi_by_channel.csv",
                mime="text/csv",
            )

    with ec2:
        if results.optimized_budget:
            opt_export = pd.DataFrame(results.optimized_budget).T
            st.download_button(
                "💰 Budget optimisé (CSV)",
                opt_export.to_csv(),
                file_name="mmm_optimized_budget.csv",
                mime="text/csv",
            )

    with ec3:
        if 'scenario_df' in dir() and scenario_df is not None:
            st.download_button(
                "🎯 Scénario actuel (CSV)",
                scenario_df.to_csv(index=False),
                file_name="mmm_scenario.csv",
                mime="text/csv",
            )

    st.session_state['optimization_done'] = True
