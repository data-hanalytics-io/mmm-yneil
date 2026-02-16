"""
Wrapper autour de Google Meridian pour simplifier l'intégration Streamlit.
Gère la configuration, l'entraînement et l'extraction des résultats.
"""
import pandas as pd
import numpy as np
import pickle
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict


@dataclass
class ModelConfig:
    """Configuration du modèle MMM."""
    kpi_column: str = 'revenue'
    kpi_type: str = 'revenue'
    max_lag: int = 8
    knots: int = 5
    n_chains: int = 4
    n_adapt: int = 500
    n_burnin: int = 100
    n_keep: int = 500
    hill_before_adstock: bool = False
    paid_media_prior_type: str = 'roi'
    media_effects_dist: str = 'log_normal'
    seed: int = 42

    # Canaux sélectionnés
    paid_channels: List[str] = field(default_factory=list)
    organic_channels: List[str] = field(default_factory=list)
    control_columns: List[str] = field(default_factory=list)

    # Priors personnalisés (optionnel)
    custom_roi_priors: Dict[str, Dict] = field(default_factory=dict)


@dataclass
class ModelResults:
    """Résultats du modèle (simulés si Meridian non disponible)."""
    is_simulated: bool = True
    r_squared: float = 0.0
    mape: float = 0.0
    wmape: float = 0.0

    # ROI par canal
    roi_by_channel: Dict[str, Dict] = field(default_factory=dict)

    # Contribution par canal
    contribution_by_channel: Dict[str, float] = field(default_factory=dict)

    # Adstock parameters
    adstock_params: Dict[str, Dict] = field(default_factory=dict)

    # Response curve data
    response_curves: Dict[str, pd.DataFrame] = field(default_factory=dict)

    # Time decomposition
    time_decomposition: Optional[pd.DataFrame] = None

    # Budget optimization
    optimized_budget: Optional[Dict] = None


def try_import_meridian():
    """Tente d'importer Meridian. Retourne None si non disponible."""
    try:
        from meridian.model import model as meridian_model
        from meridian.model import spec as model_spec
        from meridian.data import load
        from meridian.analysis import analyzer
        from meridian.analysis import visualizer
        from meridian.analysis import optimizer
        return {
            'model': meridian_model,
            'spec': model_spec,
            'load': load,
            'analyzer': analyzer,
            'visualizer': visualizer,
            'optimizer': optimizer,
        }
    except ImportError:
        return None


def build_meridian_input(wide_df: pd.DataFrame, config: ModelConfig, meridian_modules: dict):
    """Construit l'InputData Meridian depuis le DataFrame wide."""
    load = meridian_modules['load']

    media_cols = [c for c in wide_df.columns if c.endswith('_media')]
    spend_cols = [c for c in wide_df.columns if c.endswith('_spend')]
    organic_cols = [c for c in wide_df.columns
                    if c.endswith('_sessions')
                    and any(x in c for x in ['organic_search', 'organic_social',
                                              'emailing', 'push_notification'])]
    control_cols = [c for c in wide_df.columns
                    if c.endswith('_sessions')
                    and any(x in c for x in ['direct_traffic', 'referral_traffic'])]

    # Filtrer selon la config
    if config.paid_channels:
        media_cols = [c for c in media_cols
                      if c.replace('_media', '') in config.paid_channels]
        spend_cols = [c for c in spend_cols
                      if c.replace('_spend', '') in config.paid_channels]

    # S'assurer que media et spend sont alignés (mêmes canaux, même ordre)
    media_channels = [c.replace('_media', '') for c in media_cols]
    spend_channels = [c.replace('_spend', '') for c in spend_cols]

    # Ne garder que les canaux présents dans les deux listes
    common_channels = [ch for ch in media_channels if ch in spend_channels]
    media_cols = [f'{ch}_media' for ch in common_channels]
    spend_cols = [f'{ch}_spend' for ch in common_channels]

    media_to_channel = {f'{ch}_media': ch for ch in common_channels}
    spend_to_channel = {f'{ch}_spend': ch for ch in common_channels}

    coord_to_columns = load.CoordToColumns(
        time='time',
        geo='geo',
        controls=control_cols if control_cols else None,
        population='population',
        kpi=config.kpi_column,
        media=media_cols,
        media_spend=spend_cols,
        organic_media=organic_cols if organic_cols else None,
    )

    loader = load.DataFrameDataLoader(
        df=wide_df,
        kpi_type=config.kpi_type,
        coord_to_columns=coord_to_columns,
        media_to_channel=media_to_channel,
        media_spend_to_channel=spend_to_channel,
    )

    return loader.load()


def train_meridian_model(input_data, config: ModelConfig, meridian_modules: dict):
    """Configure et entraîne le modèle Meridian."""
    model_mod = meridian_modules['model']
    spec_mod = meridian_modules['spec']

    model_specification = spec_mod.ModelSpec(
        knots=config.knots,
        max_lag=config.max_lag,
        hill_before_adstock=config.hill_before_adstock,
        paid_media_prior_type=config.paid_media_prior_type,
        media_effects_dist=config.media_effects_dist,
    )

    mmm = model_mod.Meridian(
        input_data=input_data,
        model_spec=model_specification,
    )

    mmm.sample_posterior(
        n_chains=config.n_chains,
        n_adapt=config.n_adapt,
        n_burnin=config.n_burnin,
        n_keep=config.n_keep,
        seed=config.seed,
    )

    # sample_prior() requis par summary_metrics(), adstock_decay(), response_curves()
    mmm.sample_prior(n_draws=config.n_keep, seed=config.seed)

    return mmm


def extract_results(mmm, meridian_modules: dict) -> ModelResults:
    """Extrait les résultats du modèle entraîné (API Meridian v1.5)."""
    analyzer_mod = meridian_modules['analyzer']
    mmm_analyzer = analyzer_mod.Analyzer(meridian=mmm)

    # Model fit via predictive_accuracy()
    # Structure: value(metric, geo_granularity) avec metric=['R_Squared','MAPE','wMAPE']
    import math
    r_squared = mape = wmape = 0.0
    try:
        pa = mmm_analyzer.predictive_accuracy()
        geo_gran = list(pa.coords['geo_granularity'].values)[0]
        r_squared = float(pa.sel(metric='R_Squared', geo_granularity=geo_gran)['value'].values)
        mape_raw = float(pa.sel(metric='MAPE', geo_granularity=geo_gran)['value'].values)
        wmape = float(pa.sel(metric='wMAPE', geo_granularity=geo_gran)['value'].values)
        mape = 0.0 if (math.isinf(mape_raw) or math.isnan(mape_raw)) else mape_raw
        wmape = 0.0 if (math.isinf(wmape) or math.isnan(wmape)) else wmape
    except Exception:
        pass

    results = ModelResults(
        is_simulated=False,
        r_squared=r_squared,
        mape=mape,
        wmape=wmape,
    )

    # ROI & Contributions via summary_metrics()
    try:
        sm = mmm_analyzer.summary_metrics()
        coord_name = None
        for candidate in ['channel', 'media_channel', 'paid_media_channel']:
            if candidate in sm.dims:
                coord_name = candidate
                break

        if coord_name:
            channels = [str(c) for c in sm.coords[coord_name].values
                        if not str(c).lower().startswith('all ')]

            for ch in channels:
                ch_data = sm.sel({coord_name: ch})
                roi_mean = spend = contribution_pct = 0.0
                roi_ci_lo = roi_ci_hi = contribution_abs = 0.0

                for var in sm.data_vars:
                    try:
                        val = ch_data[var].values
                        v = float(val) if val.size == 1 else float(val.mean())
                    except Exception:
                        continue
                    vl = var.lower()
                    if 'roi' in vl and 'marginal' not in vl:
                        if 'mean' in vl or vl == 'roi' or 'median' in vl:
                            roi_mean = v
                        elif 'lo' in vl or 'lower' in vl:
                            roi_ci_lo = v
                        elif 'hi' in vl or 'upper' in vl:
                            roi_ci_hi = v
                    elif 'spend' in vl or 'cost' in vl:
                        spend = v
                    elif 'pct_of_contribution' in vl:
                        contribution_pct = v * 100 if v < 1 else v
                    elif 'contribution' in vl and 'pct' not in vl:
                        contribution_abs = v

                results.roi_by_channel[ch] = {
                    'roi_mean': round(roi_mean, 4),
                    'roi_ci_lo': round(roi_ci_lo, 4),
                    'roi_ci_hi': round(roi_ci_hi, 4),
                    'spend': round(spend, 2),
                    'contribution_pct': round(contribution_pct, 2),
                }
                results.contribution_by_channel[ch] = round(
                    contribution_abs if contribution_abs != 0 else spend * roi_mean, 2
                )
    except Exception:
        pass

    return results


# ══════════════════════════════════════════════════════════════════════
# SAVE / LOAD MODEL BUNDLE
# ══════════════════════════════════════════════════════════════════════

BUNDLE_VERSION = "1.0"


def compute_data_hash(df: pd.DataFrame) -> str:
    """Compute a SHA-256 hash of the DataFrame for traceability."""
    return "sha256:" + hashlib.sha256(
        pd.util.hash_pandas_object(df).values.tobytes()
    ).hexdigest()[:16]


def save_model_bundle(
    results: ModelResults,
    config: ModelConfig,
    filepath: str,
    training_duration: float = 0.0,
    n_geos: int = 0,
    n_weeks: int = 0,
    data_hash: str = "",
    meridian_version: str = "unknown",
) -> str:
    """Serialize ModelResults + ModelConfig into a single .pkl bundle."""
    bundle = {
        "meta": {
            "version": BUNDLE_VERSION,
            "created_at": datetime.now().isoformat(),
            "meridian_version": meridian_version,
            "training_duration_seconds": training_duration,
            "n_geos": n_geos,
            "n_weeks": n_weeks,
            "data_hash": data_hash,
        },
        "config": asdict(config),
        "results": {
            "is_simulated": results.is_simulated,
            "r_squared": results.r_squared,
            "mape": results.mape,
            "wmape": results.wmape,
            "roi_by_channel": results.roi_by_channel,
            "contribution_by_channel": results.contribution_by_channel,
            "adstock_params": results.adstock_params,
            "response_curves": results.response_curves,
            "time_decomposition": results.time_decomposition,
            "optimized_budget": results.optimized_budget,
        },
    }

    with open(filepath, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    return filepath


def load_model_bundle(filepath: str) -> Tuple[ModelResults, ModelConfig, Dict]:
    """
    Load a saved model bundle.
    Returns (ModelResults, ModelConfig, metadata_dict).
    """
    with open(filepath, "rb") as f:
        bundle = pickle.load(f)

    if not isinstance(bundle, dict):
        raise ValueError("Format de bundle invalide")
    for key in ("meta", "config", "results"):
        if key not in bundle:
            raise ValueError(f"Cle manquante dans le bundle: '{key}'")

    meta = bundle["meta"]

    config = ModelConfig(**bundle["config"])

    r = bundle["results"]
    results = ModelResults(
        is_simulated=r.get("is_simulated", False),
        r_squared=r.get("r_squared", 0.0),
        mape=r.get("mape", 0.0),
        wmape=r.get("wmape", 0.0),
        roi_by_channel=r.get("roi_by_channel", {}),
        contribution_by_channel=r.get("contribution_by_channel", {}),
        adstock_params=r.get("adstock_params", {}),
        response_curves=r.get("response_curves", {}),
        time_decomposition=r.get("time_decomposition"),
        optimized_budget=r.get("optimized_budget"),
    )

    return results, config, meta


# ══════════════════════════════════════════════════════════════════════
# SIMULATION (quand Meridian n'est pas installé)
# ══════════════════════════════════════════════════════════════════════

def generate_simulated_results(
    wide_df: pd.DataFrame,
    config: ModelConfig
) -> ModelResults:
    """
    Génère des résultats simulés réalistes pour la démo.
    Basé sur des heuristiques et les données réelles de spend/revenue.
    """
    np.random.seed(config.seed)

    # Identifier les colonnes
    media_cols = [c for c in wide_df.columns if c.endswith('_media')]
    spend_cols = [c for c in wide_df.columns if c.endswith('_spend')]
    channels = [c.replace('_media', '') for c in media_cols]

    total_revenue = wide_df['revenue'].sum()
    total_spend = sum(wide_df[c].sum() for c in spend_cols)

    # ── ROI par canal (simulé avec du bruit réaliste) ──
    roi_by_channel = {}
    contribution_by_channel = {}
    baseline_share = np.random.uniform(0.35, 0.50)
    remaining_share = 1 - baseline_share

    channel_spends = {}
    for ch, sc in zip(channels, spend_cols):
        channel_spends[ch] = wide_df[sc].sum()

    total_paid_spend = sum(channel_spends.values())

    for ch in channels:
        ch_spend = channel_spends[ch]
        spend_share = ch_spend / max(total_paid_spend, 1)

        # Simuler un ROI réaliste selon le type de canal
        if 'brand' in ch:
            base_roi = np.random.uniform(3.0, 8.0)
        elif 'non_brand' in ch:
            base_roi = np.random.uniform(1.5, 4.5)
        elif 'social' in ch:
            base_roi = np.random.uniform(0.3, 2.0)
        elif 'retarget' in ch:
            base_roi = np.random.uniform(2.0, 6.0)
        elif 'affil' in ch:
            base_roi = np.random.uniform(2.5, 7.0)
        else:
            base_roi = np.random.uniform(0.5, 3.0)

        ci_width = base_roi * np.random.uniform(0.15, 0.40)

        roi_by_channel[ch] = {
            'roi_mean': round(base_roi, 2),
            'roi_ci_lo': round(max(0, base_roi - ci_width), 2),
            'roi_ci_hi': round(base_roi + ci_width, 2),
            'spend': round(ch_spend, 0),
            'contribution_pct': round(spend_share * remaining_share * 100, 1),
        }
        contribution_by_channel[ch] = round(
            spend_share * remaining_share * total_revenue, 0
        )

    # ── Response curves (simulées) ──
    response_curves = {}
    for ch in channels:
        ch_spend = channel_spends[ch]
        if ch_spend == 0:
            continue
        x = np.linspace(0, ch_spend * 2.5, 100)
        roi = roi_by_channel[ch]['roi_mean']
        # Hill function simulée
        ec = ch_spend * np.random.uniform(0.6, 1.2)
        slope = np.random.uniform(1.2, 3.0)
        y = (roi * ch_spend) * (x / ec) ** slope / (1 + (x / ec) ** slope)
        response_curves[ch] = pd.DataFrame({
            'spend': x,
            'incremental_revenue': y,
            'marginal_roi': np.gradient(y, x),
        })

    # ── Time decomposition (simulée) ──
    weeks = sorted(wide_df['time'].unique())
    weekly_rev = wide_df.groupby('time')['revenue'].sum()

    decomp_data = {'time': weeks}
    decomp_data['baseline'] = [
        weekly_rev.get(w, 0) * baseline_share * np.random.uniform(0.9, 1.1)
        for w in weeks
    ]

    for ch in channels[:4]:  # Top 4 pour lisibilité
        share = roi_by_channel[ch]['contribution_pct'] / 100
        decomp_data[ch] = [
            weekly_rev.get(w, 0) * share * np.random.uniform(0.7, 1.3)
            for w in weeks
        ]

    time_decomp = pd.DataFrame(decomp_data)

    # ── Budget optimization (simulée) ──
    optimized = {}
    for ch in channels:
        current = channel_spends[ch]
        roi = roi_by_channel[ch]['roi_mean']
        # Channels à haut ROI reçoivent plus, bas ROI reçoivent moins
        avg_roi = np.mean([v['roi_mean'] for v in roi_by_channel.values()])
        factor = 1 + (roi - avg_roi) / avg_roi * 0.3
        factor = np.clip(factor, 0.5, 1.8)
        optimized[ch] = {
            'current_spend': round(current, 0),
            'optimized_spend': round(current * factor, 0),
            'change_pct': round((factor - 1) * 100, 1),
            'expected_roi': round(roi * np.random.uniform(0.95, 1.15), 2),
        }

    return ModelResults(
        is_simulated=True,
        r_squared=round(np.random.uniform(0.82, 0.95), 4),
        mape=round(np.random.uniform(0.06, 0.15), 4),
        wmape=round(np.random.uniform(0.04, 0.12), 4),
        roi_by_channel=roi_by_channel,
        contribution_by_channel=contribution_by_channel,
        adstock_params={
            ch: {
                'decay': round(np.random.uniform(0.3, 0.85), 2),
                'peak_lag': int(np.random.choice([0, 1, 2, 3])),
            }
            for ch in channels
        },
        response_curves=response_curves,
        time_decomposition=time_decomp,
        optimized_budget=optimized,
    )
