# 📊 MMM Studio — Marketing Mix Modeling App

Application Streamlit complète pour l'analyse Marketing Mix Modeling, propulsée par Google Meridian.

## Architecture

```
mmm_app/
├── app.py                          # Point d'entrée Streamlit (routing, CSS, sidebar)
├── requirements.txt                # Dépendances Python
├── pages/
│   ├── home.py                     # 🏠 Accueil — Overview pipeline & concepts
│   ├── data_import.py              # 📁 Import — Upload CSV, EDA, qualité des données
│   ├── channel_performance.py      # 📈 Canaux — Treemaps, ROAS, corrélations
│   ├── model_config.py             # ⚙️ Config — Sélection canaux, priors, hyperparams
│   ├── results.py                  # 🔬 Résultats — ROI, response curves, décomposition
│   └── optimization.py             # 💰 Optim — Réallocation budget, simulateur scénarios
└── utils/
    ├── data_processing.py          # Helpers de transformation et constantes
    └── meridian_wrapper.py         # Wrapper Meridian + mode simulation
```

## Pages

| Page | Fonctionnalités |
|---|---|
| **🏠 Accueil** | Pipeline overview, statut des étapes, concepts clés MMM |
| **📁 Import & Exploration** | Upload CSV, KPIs globaux, analyse par pays/canal/temps, heatmaps, qualité des données Meridian |
| **📈 Performance Canaux** | Treemaps spend/revenue, matrice ROAS × pays, analyse individuelle par canal, corrélations |
| **⚙️ Configuration Modèle** | Sélection canaux (paid/organic/control), KPI, adstock, knots, priors ROI, MCMC params |
| **🔬 Résultats & Diagnostics** | Model fit (R², MAPE), ROI incrémental avec IC, décomposition temporelle, response curves, adstock |
| **💰 Optimisation Budget** | Réallocation à budget fixe, simulateur de scénarios interactif, exports CSV |

## Installation & Lancement

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. (Optionnel) Installer Meridian pour les vrais résultats bayésiens
pip install google-meridian          # CPU
pip install google-meridian[and-cuda] # GPU (recommandé)

# 3. Lancer l'application
streamlit run app.py
```

## Modes de fonctionnement

### 🎭 Mode Simulation (sans Meridian)
L'app fonctionne parfaitement sans Meridian installé. Elle génère des résultats simulés réalistes basés sur vos données réelles (heuristiques sur les ROAS observés, simulation Hill/adstock). Idéal pour la démo et le prototypage.

### 🔬 Mode Meridian (avec Meridian)
Si Meridian est installé, l'app utilise le vrai moteur MCMC/NUTS bayésien. Nécessite Python 3.11-3.12 et un GPU est fortement recommandé.

## Format de données attendu

```csv
week,country,channel_grouping,sessions,revenue,new_customers,cost,transactions,impressions,clicks
52,"FR","Paid Search Brand",1208008,2347311,7492,34925,31598,2908736,1300895
```

| Colonne | Description |
|---|---|
| `week` | Numéro de semaine ISO |
| `country` | Code pays (FR, DE, UK...) |
| `channel_grouping` | Nom du canal marketing |
| `sessions` | Sessions web |
| `revenue` | Revenue généré |
| `cost` | Dépense média |
| `transactions` | Nombre de transactions |
| `impressions` | Impressions publicitaires |
| `clicks` | Clics |

## Mapping des canaux

L'app mappe automatiquement vos canaux vers les catégories Meridian :

- **Paid Media** (6 canaux) : Paid Search Brand/Non Brand, Paid Social, Retargeting, Affiliation, Display
- **Organic Media** (4 canaux) : Organic Search, Organic Social, Emailing, Push Notification
- **Controls** (2 canaux) : Direct, Referral (variables confondantes)
- **Exclus** : (Other), Not Tracked, Sms, Qr code, etc.

## Déploiement

### Streamlit Cloud
```bash
# Pusher le repo sur GitHub, puis connecter sur share.streamlit.io
```

### Docker
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

### Google Cloud Run
```bash
gcloud run deploy mmm-studio --source . --region europe-west1
```
