# HP AI Engine

**CNG Demand Forecasting & Optimisation Engine for HPCL**

A production-grade AI system that forecasts CNG demand across HPCL's nationwide station network and powers both day-to-day operational decisions and long-term strategic expansion planning.

---

## Architecture Overview

```
6 Data Inputs → TFT-GCN Prediction Engine → 3 Forecast Horizons
                                           ├── 0–6 hours  ──→ Operational Optimiser
                                           ├── 1–7 days   ──→ Operational Optimiser
                                           └── 1–6 months ──→ Location Intelligence
```

### Core Components

| Component | Purpose |
|---|---|
| **Prediction Engine** | TFT-GCN hybrid model producing multi-horizon demand forecasts with uncertainty quantification and SHAP interpretability |
| **Operational Optimiser** | Real-time dispenser management, tanker routing (OR-Tools VRPTW), and demand shifting |
| **Location Intelligence** | DBSCAN demand-zone clustering, multi-criteria site scoring, MDU vs permanent station decision |
| **Scalability** | PSI drift detection, automated retraining, federated learning, transfer learning for new stations |

---

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Generate synthetic training data
python -m hp_ai_engine.data.synthetic
```

---

## Configuration

All configuration lives in `configs/`:
- `model_config.yaml` — Model hyperparameters
- `training_config.yaml` — Training parameters
- `data_config.yaml` — Data paths and feature definitions
- `deployment_config.yaml` — Drift thresholds, federated learning, retraining schedule

---

## Project Structure

```
HP-AI-Engine/
├── pyproject.toml                          # Build config, dependencies
├── requirements.txt                        # Pinned dependencies
├── README.md                               # Overview and quick-start
├── PROJECT_STRUCTURE.md                    # Detailed structure doc
│
├── configs/
│   ├── model_config.yaml                   # GCN / TFT / attention hyperparameters
│   ├── training_config.yaml                # Loss weights, scheduler, checkpointing
│   ├── data_config.yaml                    # Feature lists, synthetic data params
│   └── deployment_config.yaml              # Drift thresholds, federated learning, retraining
│
├── hp_ai_engine/
│   ├── __init__.py                         # Package root
│   │
│   ├── utils/                              # Shared foundation
│   │   ├── __init__.py
│   │   ├── config.py                       # YAML loader with env-var overrides (HP_AI__*)
│   │   ├── logging.py                      # Structured JSON logging with context fields
│   │   ├── metrics.py                      # MAPE, RMSE, MAE, R², sMAPE, calibration
│   │   ├── geo.py                          # Haversine, OSM road network, distance matrices
│   │   └── time_utils.py                   # IST timezone, cyclical features, Indian holidays
│   │
│   ├── data/                               # Data layer
│   │   ├── __init__.py
│   │   ├── schemas.py                      # 7 Pydantic models (6 sources + merged record)
│   │   ├── ingestion.py                    # CSV/Parquet loader, schema validation, merge pipeline
│   │   ├── synthetic.py                    # Realistic synthetic data for all 6 sources
│   │   ├── feature_engineering.py          # Lags, rolling stats, cyclical encoding, normalisation
│   │   ├── graph_builder.py                # Dynamic GCN adjacency matrix (add/remove nodes)
│   │   └── dataset.py                      # PyTorch Dataset, collation, temporal train/val/test split
│   │
│   ├── models/                             # Prediction engine
│   │   ├── __init__.py
│   │   ├── gcn_encoder.py                  # Spatial GCN with residual connections + batch norm
│   │   ├── tft_model.py                    # Full TFT: GRN, VSN, LSTM enc/dec, attention, 3 heads
│   │   ├── context_attention.py            # Real-time override (z-score → bounded scaling)
│   │   ├── tft_gcn.py                      # Combined engine: GCN → TFT → Context Override
│   │   ├── uncertainty.py                  # MC Dropout (30 passes, percentile CIs)
│   │   └── explainability.py               # SHAP ablation + TFT-native attention extraction
│   │
│   ├── training/                           # Training pipeline
│   │   ├── __init__.py
│   │   ├── loss.py                         # Huber + volume-weighted MAPE, multi-horizon combiner
│   │   ├── trainer.py                      # Training loop: OneCycleLR, early stop, top-k checkpoints
│   │   ├── transfer.py                     # Warm-start new stations (freeze GCN → fine-tune)
│   │   └── federated.py                    # FedAvg: aggregator, cluster clients, rollback, coordinator
│   │
│   ├── optimiser/                          # Operational optimisation (0–6h forecast driven)
│   │   ├── __init__.py
│   │   ├── dispenser_rules.py              # 5 threshold rules → actionable instructions
│   │   ├── tanker_routing.py               # OR-Tools VRPTW solver + emergency rerouting
│   │   └── demand_shifting.py              # Non-price CRM: loyalty, fleet scheduling, redirects
│   │
│   ├── location/                           # Location intelligence (1–6mo forecast driven)
│   │   ├── __init__.py
│   │   ├── clustering.py                   # DBSCAN demand zones + underserved flagging
│   │   ├── site_scoring.py                 # 6-criteria MCDA (weighted, transparent breakdown)
│   │   └── mdu_decision.py                 # CV-based permanent vs MDU recommendation
│   │
│   └── scalability/                        # Production resilience
│       ├── __init__.py
│       ├── drift_detection.py              # PSI (covariate) + CUSUM (concept) drift monitoring
│       ├── retraining.py                   # Automated retrain: warm-start, validate, promote/rollback
│       └── onboarding.py                   # 5-step zero-downtime station onboarding pipeline
│
├── tests/
│   ├── __init__.py
│   ├── test_utils.py                       # Unit tests: metrics, geo, time, config
│   ├── test_data.py                        # Tests: schemas, synthetic gen, features, graph builder
│   └── test_engine.py                      # Tests: GCN, TFT, context, loss, rules, drift, MDU, scoring
│
├── platform/                               # (Future) API & dashboard layer
│   ├── api/
│   │   ├── main.py                         # FastAPI app
│   │   ├── routes/                         # REST endpoints per user role
│   │   └── middleware/                      # Auth, RBAC, rate limiting
│   └── dashboards/
│       ├── station_manager/                # Real-time view: 0-6h forecast, actions
│       ├── city_ops/                       # Cluster view: inter-station alerts, routing map
│       └── hpcl_executive/                 # Strategic: expansion heatmap, CAPEX simulator
│
└── infrastructure/                         # (Future) Deployment layer
    ├── docker/                             # Container definitions per tier
    ├── kubernetes/                         # K8s manifests (central / city / edge)
    ├── terraform/                          # Cloud infra provisioning
    └── monitoring/                         # Grafana dashboards, alerting rules
```

---

## Module Count

| Package       | Modules | Description |
|:-------------|:-------:|:-----------|
| utils         | 5       | Config, logging, metrics, geo, time |
| data          | 6       | Schemas, ingestion, synthetic, features, graph, dataset |
| models        | 6       | GCN, TFT, context attention, combined, uncertainty, SHAP |
| training      | 4       | Loss, trainer, transfer, federated |
| optimiser     | 3       | Dispenser rules, tanker routing, demand shifting |
| location      | 3       | Clustering, site scoring, MDU decisions |
| scalability   | 3       | Drift detection, retraining, onboarding |
| tests         | 3       | Utils, data, engine tests |
| **Total**     | **33**  | Production engine + test suite |

---

## License

Proprietary — HPCL Internal Use Only
