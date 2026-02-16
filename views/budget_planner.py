"""Page Planification Budgétaire — Allocation optimale par canal."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from utils.data_processing import format_number


# ══════════════════════════════════════════════════════════════════════
# MOTEUR D'OPTIMISATION
# ══════════════════════════════════════════════════════════════════════

def _revenue_from_curve(spend: float, curve_df: pd.DataFrame) -> float:
    """Interpole le revenue incremental depuis la courbe de reponse."""
    if curve_df is None or len(curve_df) == 0 or spend <= 0:
        return 0.0
    return float(np.interp(spend, curve_df['spend'], curve_df['incremental_revenue']))


def _marginal_roi_from_curve(spend: float, curve_df: pd.DataFrame) -> float:
    """Interpole le ROI marginal depuis la courbe de reponse."""
    if curve_df is None or len(curve_df) == 0 or spend <= 0:
        return 0.0
    return float(np.interp(spend, curve_df['spend'], curve_df['marginal_roi']))


def optimize_budget(
    total_budget: float,
    channels: list,
    response_curves: dict,
    roi_by_channel: dict,
    min_spend: dict = None,
    max_spend: dict = None,
    n_steps: int = 500,
) -> dict:
    """
    Allocation optimale via approche marginale (greedy).
    A chaque iteration, alloue un increment au canal avec le meilleur mROI.
    """
    if min_spend is None:
        min_spend = {}
    if max_spend is None:
        max_spend = {}

    # Initialiser avec les minimums
    alloc = {}
    remaining = total_budget
    for ch in channels:
        m = min_spend.get(ch, 0)
        alloc[ch] = m
        remaining -= m

    remaining = max(remaining, 0)
    increment = remaining / max(n_steps, 1)

    # Allouer incrementalement au canal avec le meilleur mROI
    for _ in range(n_steps):
        if remaining <= 0:
            break

        best_ch = None
        best_mroi = -1

        for ch in channels:
            cap = max_spend.get(ch, float('inf'))
            if alloc[ch] >= cap:
                continue

            curve = response_curves.get(ch)
            if curve is not None and len(curve) > 0:
                mroi = _marginal_roi_from_curve(alloc[ch] + increment, curve)
            else:
                # Fallback: ROI moyen avec decroissance
                base_roi = roi_by_channel.get(ch, {}).get('roi_mean', 1)
                current_spend = roi_by_channel.get(ch, {}).get('spend', 1)
                ratio = (alloc[ch] + increment) / max(current_spend, 1)
                mroi = base_roi * (ratio ** -0.3) if ratio > 0 else base_roi

            if mroi > best_mroi:
                best_mroi = mroi
                best_ch = ch

        if best_ch is None:
            break

        step = min(increment, remaining)
        cap = max_spend.get(best_ch, float('inf'))
        step = min(step, cap - alloc[best_ch])
        alloc[best_ch] += step
        remaining -= step

    return alloc


# ══════════════════════════════════════════════════════════════════════
# VUE STREAMLIT
# ══════════════════════════════════════════════════════════════════════

def render():
    st.markdown("## 📋 Planification Budgétaire")

    if 'model_results' not in st.session_state:
        st.warning(
            "⚠️ Aucun modèle entraîné. Allez dans **Configuration Modèle** "
            "pour lancer l'entraînement d'abord."
        )
        return

    results = st.session_state['model_results']

    if results.is_simulated:
        st.info("🎭 **Mode Simulation** — Résultats basés sur des heuristiques.")

    if not results.roi_by_channel:
        st.warning("Aucun résultat de ROI disponible.")
        return

    channels = list(results.roi_by_channel.keys())
    current_spends = {ch: results.roi_by_channel[ch].get('spend', 0) for ch in channels}
    total_current = sum(current_spends.values())

    # ══════════════════════════════════════════════════════════════
    # INPUTS: Budget & Période
    # ══════════════════════════════════════════════════════════════
    st.markdown("### 1️⃣ Budget & Période")

    col_budget, col_dates = st.columns(2)

    with col_budget:
        default_budget = max(1000, int(total_current)) if total_current > 0 else 500_000
        total_budget = st.number_input(
            "Budget total à allouer (€)",
            min_value=100,
            max_value=100_000_000,
            value=default_budget,
            step=1_000,
            help="Le budget total à répartir sur la période sélectionnée.",
        )

    with col_dates:
        today = datetime.now().date()
        next_monday = today + timedelta(days=(7 - today.weekday()) % 7)

        date_range = st.date_input(
            "Période de planification",
            value=(next_monday, next_monday + timedelta(weeks=12)),
            min_value=today,
            help="Sélectionnez la période sur laquelle allouer le budget.",
        )

        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            n_weeks = max(1, (end_date - start_date).days // 7)
        else:
            start_date = next_monday
            end_date = next_monday + timedelta(weeks=12)
            n_weeks = 12

    st.markdown(
        f"**{format_number(total_budget, suffix=' €')}** sur **{n_weeks} semaines** "
        f"({start_date.strftime('%d/%m/%Y')} → {end_date.strftime('%d/%m/%Y')}) "
        f"— soit ~{format_number(total_budget / n_weeks, suffix=' €')}/semaine"
    )

    # ══════════════════════════════════════════════════════════════
    # CONTRAINTES PAR CANAL (optionnel)
    # ══════════════════════════════════════════════════════════════
    st.markdown("---")
    with st.expander("⚙️ Contraintes par canal (optionnel)", expanded=False):
        st.caption(
            "Définissez un minimum ou maximum par canal. "
            "Laisser à 0 / max pour aucune contrainte."
        )

        min_spend = {}
        max_spend = {}

        for i in range(0, len(channels), 3):
            cols = st.columns(3)
            for j, col in enumerate(cols):
                idx = i + j
                if idx >= len(channels):
                    break
                ch = channels[idx]
                ch_label = ch.replace('_', ' ').title()
                with col:
                    st.markdown(f"**{ch_label}**")
                    c1, c2 = st.columns(2)
                    with c1:
                        mn = st.number_input(
                            "Min €", min_value=0, value=0,
                            step=1000, key=f"min_{ch}",
                        )
                        min_spend[ch] = mn
                    with c2:
                        mx = st.number_input(
                            "Max €", min_value=0, value=0,
                            step=1000, key=f"max_{ch}",
                            help="0 = pas de limite",
                        )
                        if mx > 0:
                            max_spend[ch] = mx

    # ══════════════════════════════════════════════════════════════
    # OPTIMISATION
    # ══════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 2️⃣ Allocation optimale")

    alloc = optimize_budget(
        total_budget=total_budget,
        channels=channels,
        response_curves=results.response_curves,
        roi_by_channel=results.roi_by_channel,
        min_spend=min_spend,
        max_spend=max_spend if max_spend else None,
    )

    # Calculer les revenues attendus
    rows = []
    total_expected_rev = 0
    for ch in channels:
        spend = alloc[ch]
        curve = results.response_curves.get(ch)
        if curve is not None and len(curve) > 0:
            rev = _revenue_from_curve(spend, curve)
        else:
            roi = results.roi_by_channel[ch].get('roi_mean', 1)
            current = current_spends.get(ch, 1)
            if current > 0 and spend > 0:
                ratio = spend / current
                rev = roi * current * (ratio ** 0.7)
            else:
                rev = roi * spend * 0.8

        total_expected_rev += rev
        pct = (spend / total_budget * 100) if total_budget > 0 else 0
        weekly = spend / n_weeks

        rows.append({
            'Canal': ch.replace('_', ' ').title(),
            'canal_id': ch,
            'Budget alloué': spend,
            '% du budget': pct,
            'Budget / semaine': weekly,
            'Revenue attendu': rev,
            'ROAS estimé': rev / spend if spend > 0 else 0,
        })

    plan_df = pd.DataFrame(rows)

    # ── KPIs ──
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Budget total", format_number(total_budget, suffix=' €'))
    k2.metric("Revenue attendu", format_number(total_expected_rev, suffix=' €'))
    k3.metric("ROAS global", f"{total_expected_rev / max(total_budget, 1):.2f}x")
    k4.metric("Durée", f"{n_weeks} semaines")

    # ── Pie chart allocation ──
    col_pie, col_bar = st.columns(2)

    with col_pie:
        fig_pie = px.pie(
            plan_df[plan_df['Budget alloué'] > 0],
            values='Budget alloué',
            names='Canal',
            title='Répartition du budget',
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4,
        )
        fig_pie.update_traces(textinfo='percent+label', textposition='outside')
        fig_pie.update_layout(height=450, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_bar:
        fig_bar = go.Figure()
        plan_sorted = plan_df.sort_values('Budget alloué', ascending=True)

        fig_bar.add_trace(go.Bar(
            y=plan_sorted['Canal'],
            x=plan_sorted['Budget alloué'],
            orientation='h',
            marker_color='#6366f1',
            text=plan_sorted['Budget alloué'].apply(
                lambda x: format_number(x, suffix=' €')
            ),
            textposition='outside',
            name='Budget',
        ))
        fig_bar.update_layout(
            title='Budget par canal',
            height=450,
            xaxis_title='Budget (€)',
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Table détaillée ──
    st.markdown("#### Détail de l'allocation")
    st.dataframe(
        plan_df[['Canal', 'Budget alloué', '% du budget', 'Budget / semaine',
                 'Revenue attendu', 'ROAS estimé']].style.format({
            'Budget alloué': '€{:,.0f}',
            '% du budget': '{:.1f}%',
            'Budget / semaine': '€{:,.0f}',
            'Revenue attendu': '€{:,.0f}',
            'ROAS estimé': '{:.2f}x',
        }).background_gradient(subset=['ROAS estimé'], cmap='RdYlGn'),
        use_container_width=True, hide_index=True,
    )

    # ══════════════════════════════════════════════════════════════
    # SECTION 3: Planning hebdomadaire
    # ══════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 3️⃣ Planning hebdomadaire")

    # Générer le calendrier semaine par semaine
    weeks_list = []
    current_date = start_date
    week_num = 1
    while current_date < end_date:
        week_end = min(current_date + timedelta(days=6), end_date)
        weeks_list.append({
            'semaine': week_num,
            'debut': current_date.strftime('%d/%m/%Y'),
            'fin': week_end.strftime('%d/%m/%Y'),
            'label': f"S{week_num} ({current_date.strftime('%d/%m')})",
        })
        current_date += timedelta(weeks=1)
        week_num += 1

    # Saisonnalité: utiliser la décomposition temporelle si disponible
    has_seasonality = (
        results.time_decomposition is not None
        and len(results.time_decomposition) > 0
    )

    if has_seasonality:
        decomp = results.time_decomposition
        # Calculer les poids de saisonnalité à partir de la baseline
        if 'baseline' in decomp.columns:
            baseline = decomp['baseline'].values
            total_baseline = baseline.sum()
            if total_baseline > 0:
                season_weights = baseline / total_baseline * len(baseline)
            else:
                season_weights = np.ones(len(baseline))
        else:
            season_weights = np.ones(len(decomp))

        # Échantillonner n_weeks poids (cyclique si nécessaire)
        weekly_weights = np.array([
            season_weights[i % len(season_weights)]
            for i in range(len(weeks_list))
        ])
        weekly_weights = weekly_weights / weekly_weights.sum()
        st.caption("Pondération hebdomadaire basée sur la saisonnalité détectée par le modèle.")
    else:
        weekly_weights = np.ones(len(weeks_list)) / len(weeks_list)
        st.caption("Distribution uniforme (pas de données de saisonnalité).")

    # Construire le planning
    weekly_data = []
    for i, week_info in enumerate(weeks_list):
        week_row = {
            'Semaine': week_info['label'],
        }
        for ch in channels:
            ch_label = ch.replace('_', ' ').title()
            week_row[ch_label] = alloc[ch] * weekly_weights[i]
        week_row['Total'] = sum(
            alloc[ch] * weekly_weights[i] for ch in channels
        )
        weekly_data.append(week_row)

    weekly_df = pd.DataFrame(weekly_data)

    # Stacked bar chart du planning
    ch_labels = [ch.replace('_', ' ').title() for ch in channels]
    fig_weekly = go.Figure()

    for ch_label in ch_labels:
        if ch_label in weekly_df.columns and weekly_df[ch_label].sum() > 0:
            fig_weekly.add_trace(go.Bar(
                x=weekly_df['Semaine'],
                y=weekly_df[ch_label],
                name=ch_label,
            ))

    fig_weekly.update_layout(
        title='Budget hebdomadaire par canal',
        barmode='stack',
        height=450,
        yaxis_title='Budget (€)',
        xaxis_title='',
        legend=dict(orientation='h', y=-0.3),
    )
    st.plotly_chart(fig_weekly, use_container_width=True)

    # Table du planning (les 4 premières + dernière semaine)
    format_cols = {col: '€{:,.0f}' for col in weekly_df.columns if col != 'Semaine'}
    st.dataframe(
        weekly_df.style.format(format_cols),
        use_container_width=True, hide_index=True,
        height=min(400, len(weekly_df) * 38 + 40),
    )

    # ══════════════════════════════════════════════════════════════
    # SECTION 4: Comparaison avec allocation uniforme
    # ══════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 4️⃣ Gain vs allocation uniforme")

    equal_alloc = total_budget / len(channels)
    equal_rev = 0
    comparison_rows = []

    for ch in channels:
        curve = results.response_curves.get(ch)
        opt_spend = alloc[ch]

        if curve is not None and len(curve) > 0:
            opt_rev = _revenue_from_curve(opt_spend, curve)
            eq_rev = _revenue_from_curve(equal_alloc, curve)
        else:
            roi = results.roi_by_channel[ch].get('roi_mean', 1)
            current = current_spends.get(ch, 1)
            opt_rev = roi * current * ((opt_spend / max(current, 1)) ** 0.7) if opt_spend > 0 else 0
            eq_rev = roi * current * ((equal_alloc / max(current, 1)) ** 0.7) if equal_alloc > 0 else 0

        equal_rev += eq_rev

        comparison_rows.append({
            'Canal': ch.replace('_', ' ').title(),
            'Budget uniforme': equal_alloc,
            'Revenue uniforme': eq_rev,
            'Budget optimisé': opt_spend,
            'Revenue optimisé': opt_rev,
            'Gain': opt_rev - eq_rev,
        })

    comp_df = pd.DataFrame(comparison_rows)

    gain_pct = ((total_expected_rev - equal_rev) / max(equal_rev, 1)) * 100

    g1, g2, g3 = st.columns(3)
    g1.metric("Revenue (uniforme)", format_number(equal_rev, suffix=' €'))
    g2.metric("Revenue (optimisé)", format_number(total_expected_rev, suffix=' €'))
    g3.metric("Gain optimisation", f"+{gain_pct:.1f}%",
              delta=format_number(total_expected_rev - equal_rev, suffix=' €'))

    # ══════════════════════════════════════════════════════════════
    # EXPORT
    # ══════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 📥 Export du plan")

    e1, e2 = st.columns(2)

    with e1:
        st.download_button(
            "📊 Plan d'allocation (CSV)",
            plan_df[['Canal', 'Budget alloué', '% du budget', 'Budget / semaine',
                     'Revenue attendu', 'ROAS estimé']].to_csv(index=False),
            file_name=f"mmm_budget_plan_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

    with e2:
        st.download_button(
            "📅 Planning hebdomadaire (CSV)",
            weekly_df.to_csv(index=False),
            file_name=f"mmm_weekly_plan_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

    st.session_state['budget_planning_done'] = True
