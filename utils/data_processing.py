"""
Utilitaires de traitement de données pour la pipeline MMM.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional


# ══════════════════════════════════════════════════════════════════════
# CONSTANTES & CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

# Mapping des canaux vers les catégories Meridian
CHANNEL_CATEGORIES = {
    # Paid Media (ROI calculé par Meridian)
    'Paid Search Brand':     {'category': 'paid_media',     'meridian': 'paid_search_brand',     'metric': 'impressions', 'fallback': 'clicks',    'color': '#3b82f6'},
    'Paid Search Non Brand': {'category': 'paid_media',     'meridian': 'paid_search_non_brand', 'metric': 'impressions', 'fallback': 'clicks',    'color': '#6366f1'},
    'Paid Social Media':     {'category': 'paid_media',     'meridian': 'paid_social',           'metric': 'impressions', 'fallback': 'clicks',    'color': '#8b5cf6'},
    'Retargeting':           {'category': 'paid_media',     'meridian': 'retargeting',           'metric': 'impressions', 'fallback': 'clicks',    'color': '#a855f7'},
    'Affiliation':           {'category': 'paid_media',     'meridian': 'affiliation',           'metric': 'sessions',    'fallback': 'sessions',  'color': '#d946ef'},
    'Media':                 {'category': 'paid_media',     'meridian': 'display_media',         'metric': 'impressions', 'fallback': 'clicks',    'color': '#ec4899'},
    # Organic Media
    'Organic Search':        {'category': 'organic_media',  'meridian': 'organic_search',        'metric': 'sessions',    'fallback': 'sessions',  'color': '#22c55e'},
    'Organic Social Media':  {'category': 'organic_media',  'meridian': 'organic_social',        'metric': 'sessions',    'fallback': 'sessions',  'color': '#10b981'},
    'Emailing':              {'category': 'organic_media',  'meridian': 'emailing',              'metric': 'sessions',    'fallback': 'sessions',  'color': '#14b8a6'},
    'Push Notification':     {'category': 'organic_media',  'meridian': 'push_notification',     'metric': 'sessions',    'fallback': 'sessions',  'color': '#06b6d4'},
    # Controls / Non-media
    'Direct':                {'category': 'control',        'meridian': 'direct_traffic',        'metric': 'sessions',    'fallback': 'sessions',  'color': '#64748b'},
    'Referral':              {'category': 'control',        'meridian': 'referral_traffic',      'metric': 'sessions',    'fallback': 'sessions',  'color': '#94a3b8'},
}

EXCLUDED_CHANNELS = [
    '(Other)', 'Not Tracked', 'Sms', 'Qr code', 'Partnership',
    'Eprm', 'Generative AI', 'PSE', 'JOKO-AFFILIATION',
    'JOKO - AFFILIATION',
]

EXCLUDED_GEOS = ['Scandi', 'TO DELETE', 'NOT NOW', 'UNKNOW', 'AU', 'FIN', 'EME']

GEO_POPULATION = {
    'FR': 67_390_000, 'DE': 83_240_000, 'UK': 67_330_000,
    'ES': 47_420_000, 'IT': 59_550_000, 'PL': 37_750_000,
    'TR': 85_280_000, 'CZ': 10_700_000, 'RO': 19_120_000,
    'GR': 10_430_000, 'PT': 10_300_000, 'SE': 10_380_000,
    'DK':  5_870_000, 'CH':  8_700_000,
}

GEO_NAMES = {
    'FR': '🇫🇷 France', 'DE': '🇩🇪 Allemagne', 'UK': '🇬🇧 Royaume-Uni',
    'ES': '🇪🇸 Espagne', 'IT': '🇮🇹 Italie', 'PL': '🇵🇱 Pologne',
    'TR': '🇹🇷 Turquie', 'CZ': '🇨🇿 Tchéquie', 'RO': '🇷🇴 Roumanie',
    'GR': '🇬🇷 Grèce', 'PT': '🇵🇹 Portugal', 'SE': '🇸🇪 Suède',
    'DK': '🇩🇰 Danemark', 'CH': '🇨🇭 Suisse',
}

CATEGORY_LABELS = {
    'paid_media': '💰 Paid Media',
    'organic_media': '🌱 Organic Media',
    'control': '🎛️ Control',
}


# ══════════════════════════════════════════════════════════════════════
# FONCTIONS DE TRAITEMENT
# ══════════════════════════════════════════════════════════════════════

def week_to_date(week_num: int, year: int = 2024) -> str:
    """Convertit un numéro de semaine ISO + année en date (lundi)."""
    try:
        dt = datetime.strptime(f'{year}-W{week_num:02d}-1', '%G-W%V-%u')
        return dt.strftime('%Y-%m-%d')
    except ValueError:
        dt = datetime.strptime(f'{year}-W01-1', '%G-W%V-%u')
        return dt.strftime('%Y-%m-%d')


def load_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie le DataFrame brut."""
    df = df.copy()

    # Filtrer geos invalides
    df = df[~df['country'].isin(EXCLUDED_GEOS)]

    # Remplir NaN
    metrics = ['sessions', 'revenue', 'new_customers', 'cost',
               'transactions', 'impressions', 'clicks']
    for m in metrics:
        if m in df.columns:
            df[m] = df[m].fillna(0)

    # Ajouter la date à partir de year + week
    if 'week' in df.columns:
        if 'year' in df.columns:
            df['date'] = df.apply(
                lambda r: week_to_date(int(r['week']), int(r['year'])), axis=1
            )
        else:
            df['date'] = df['week'].apply(week_to_date)

    # Classifier les canaux
    df['channel_category'] = df['channel_grouping'].map(
        lambda x: CHANNEL_CATEGORIES.get(x, {}).get('category', 'excluded')
    )

    return df


def compute_geo_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Résumé par géo."""
    summary = df.groupby('country').agg(
        revenue=('revenue', 'sum'),
        cost=('cost', 'sum'),
        transactions=('transactions', 'sum'),
        sessions=('sessions', 'sum'),
        impressions=('impressions', 'sum'),
        n_channels=('channel_grouping', 'nunique'),
        n_weeks=('date', 'nunique'),
    ).reset_index()
    summary['roas'] = np.where(
        summary['cost'] > 0,
        summary['revenue'] / summary['cost'],
        0,
    )
    summary['geo_name'] = summary['country'].map(GEO_NAMES)
    return summary.sort_values('revenue', ascending=False)


def compute_channel_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Résumé par canal."""
    summary = df.groupby('channel_grouping').agg(
        revenue=('revenue', 'sum'),
        cost=('cost', 'sum'),
        transactions=('transactions', 'sum'),
        sessions=('sessions', 'sum'),
        impressions=('impressions', 'sum'),
        n_geos=('country', 'nunique'),
    ).reset_index()
    summary['roas'] = np.where(
        summary['cost'] > 0,
        summary['revenue'] / summary['cost'],
        0,
    )
    summary['category'] = summary['channel_grouping'].map(
        lambda x: CHANNEL_CATEGORIES.get(x, {}).get('category', 'excluded')
    )
    summary['color'] = summary['channel_grouping'].map(
        lambda x: CHANNEL_CATEGORIES.get(x, {}).get('color', '#94a3b8')
    )
    return summary.sort_values('revenue', ascending=False)


def compute_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """Séries temporelles par semaine."""
    ts = df.groupby('date').agg(
        revenue=('revenue', 'sum'),
        cost=('cost', 'sum'),
        transactions=('transactions', 'sum'),
        sessions=('sessions', 'sum'),
    ).reset_index().sort_values('date')
    ts['roas'] = np.where(ts['cost'] > 0, ts['revenue'] / ts['cost'], 0)
    return ts


def compute_channel_geo_matrix(df: pd.DataFrame, metric: str = 'revenue') -> pd.DataFrame:
    """Matrice canaux × géos pour un metric donné."""
    pivot = df.pivot_table(
        index='channel_grouping',
        columns='country',
        values=metric,
        aggfunc='sum',
        fill_value=0,
    )
    return pivot


def pivot_to_meridian_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme les données long-format en wide-format compatible Meridian.
    1 ligne par (geo, time), 1 colonne par channel_metric.
    Garantit une grille complète geo × time (régulièrement espacée).
    """
    # Créer la grille complète geo × time régulièrement espacée (hebdomadaire)
    all_geos = sorted(df['country'].unique())
    date_min = pd.to_datetime(df['date'].min())
    date_max = pd.to_datetime(df['date'].max())
    all_dates = pd.date_range(start=date_min, end=date_max, freq='W-MON')
    all_dates_str = [d.strftime('%Y-%m-%d') for d in all_dates]
    grid = pd.MultiIndex.from_product(
        [all_geos, all_dates_str], names=['geo', 'time']
    )
    base = pd.DataFrame(index=grid).reset_index()

    # Agréger les KPIs et merger sur la grille
    kpi_agg = df.groupby(['country', 'date']).agg(
        revenue=('revenue', 'sum'),
        transactions=('transactions', 'sum'),
    ).reset_index().rename(columns={'country': 'geo', 'date': 'time'})

    base = base.merge(kpi_agg, on=['geo', 'time'], how='left')

    # Pour chaque canal mapped, pivoter
    for ch_name, ch_cfg in CHANNEL_CATEGORIES.items():
        ch_data = df[df['channel_grouping'] == ch_name]
        if len(ch_data) == 0:
            continue

        meridian_name = ch_cfg['meridian']
        primary_metric = ch_cfg['metric']
        fallback = ch_cfg['fallback']
        category = ch_cfg['category']

        agg = ch_data.groupby(['country', 'date']).agg({
            primary_metric: 'sum',
            fallback: 'sum',
            'cost': 'sum',
        }).reset_index().rename(columns={'country': 'geo', 'date': 'time'})

        # Media metric
        col_name = f'{meridian_name}_media' if category == 'paid_media' else f'{meridian_name}_sessions'
        agg[col_name] = agg[primary_metric]
        if primary_metric != fallback:
            mask = agg[col_name] == 0
            agg.loc[mask, col_name] = agg.loc[mask, fallback]

        cols_to_merge = ['geo', 'time', col_name]

        # Spend pour paid media
        if category == 'paid_media':
            spend_col = f'{meridian_name}_spend'
            agg[spend_col] = agg['cost']
            cols_to_merge.append(spend_col)

        base = base.merge(agg[cols_to_merge], on=['geo', 'time'], how='left')

    # Population
    base['population'] = base['geo'].map(GEO_POPULATION)
    base = base.fillna(0)
    base = base.sort_values(['geo', 'time']).reset_index(drop=True)

    return base


def get_data_quality_report(df: pd.DataFrame) -> Dict:
    """Génère un rapport de qualité des données."""
    report = {
        'n_rows': len(df),
        'n_geos': df['country'].nunique(),
        'n_weeks': df['date'].nunique(),
        'n_channels': df['channel_grouping'].nunique(),
        'date_range': f"{df['date'].min()} → {df['date'].max()}" if 'date' in df.columns else 'N/A',
        'total_revenue': df['revenue'].sum(),
        'total_cost': df['cost'].sum(),
        'total_transactions': df['transactions'].sum(),
        'missing_values': df.isnull().sum().to_dict(),
        'geos': sorted(df['country'].unique()),
        'channels': sorted(df['channel_grouping'].unique()),
    }

    # Checks de qualité
    checks = []
    n_datapoints = report['n_geos'] * report['n_weeks']
    n_paid = df[df['channel_grouping'].isin(
        [k for k, v in CHANNEL_CATEGORIES.items() if v['category'] == 'paid_media']
    )]['channel_grouping'].nunique()

    ratio = n_datapoints / max(n_paid * report['n_geos'] + 5, 1)

    checks.append({
        'name': 'Data points ratio',
        'value': f'{ratio:.1f}',
        'target': '> 8',
        'status': 'ok' if ratio > 8 else ('warning' if ratio > 5 else 'error'),
    })
    checks.append({
        'name': 'Historique',
        'value': f"{report['n_weeks']} semaines",
        'target': '≥ 104 (2 ans)',
        'status': 'ok' if report['n_weeks'] >= 104 else ('warning' if report['n_weeks'] >= 52 else 'error'),
    })
    checks.append({
        'name': 'Couverture géo',
        'value': f"{report['n_geos']} geos",
        'target': '≥ 5',
        'status': 'ok' if report['n_geos'] >= 5 else 'warning',
    })

    report['quality_checks'] = checks
    return report


def format_number(n: float, prefix: str = '', suffix: str = '') -> str:
    """Formate un nombre pour l'affichage."""
    if abs(n) >= 1_000_000:
        return f"{prefix}{n/1_000_000:,.1f}M{suffix}"
    elif abs(n) >= 1_000:
        return f"{prefix}{n/1_000:,.1f}K{suffix}"
    else:
        return f"{prefix}{n:,.0f}{suffix}"
