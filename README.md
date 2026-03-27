<div align="center">

# FloodSense AI

### An Ensemble Machine Learning Framework for Intelligent Flood Risk Assessment and Early Warning

![Python](https://img.shields.io/badge/Python-3.13+-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat-square&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-189AB4?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-2ea44f?style=flat-square)

*Integrating LSTM, Random Forest, XGBoost, and SVM via ensemble averaging for high-precision, real-time flood probability estimation.*

[Model Performance](#model-performance) · [Research Comparison](#comparison-with-published-research) · [Installation](#installation) · [API Reference](#api-reference)

</div>

---

## Overview

FloodSense AI is a machine learning–powered flood risk prediction system built as part of an MCA research project at IILM University. The system processes **20 environmental and socioeconomic input features** through an ensemble of four ML models and outputs a calibrated flood probability score, classified across a five-tier risk scale.

> This system was benchmarked against peer-reviewed literature in flood susceptibility mapping and ML-based early warning systems. Full evaluation results are available in `evaluation_results.json`.

---

## Key Features

- **Four-model ensemble** — LSTM (custom NumPy), Random Forest, XGBoost, and SVM with ensemble averaging
- **Full-stack deployment** — Flask REST API with an interactive single-page web dashboard
- **No deep learning framework required** — LSTM implemented entirely in NumPy; compatible with Python 3.13+
- **Real-time analytics** — feature importance, model comparison charts, and historical data distribution
- **Five-tier risk classification** — actionable thresholds from routine monitoring to mandatory evacuation
- **REST API** — `/api/predict`, `/api/batch-predict`, `/api/metrics`, `/api/historical-data`

---

## Repository Structure

```
floodsense-ai/
├── backend/
│   ├── app.py              # Flask REST API server
│   └── train_models.py     # Model training pipeline + NumPy LSTM
├── frontend/
│   └── index.html          # Single-page web application
├── models/                 # Serialised model artefacts (.pkl, .json)
├── data/
│   └── flood.csv           # Dataset — 50,000 samples, 24 features
├── evaluate_models.py      # Evaluation script (research-grade metrics)
├── evaluation_results.json # Pre-computed evaluation output
├── requirements.txt
└── README.md
```

---

## Model Performance

**Evaluation protocol:**

| Setting | Value |
|:---|:---|
| Dataset | `flood.csv` — 50,000 samples, 20 raw + 4 derived features |
| Train / Test split | 80% / 20% (40,000 train · 10,000 test) — `random_state=42` |
| Classification threshold | 0.50 |
| Feature scaler | `StandardScaler` fitted on training partition only |

### Table 1 — Regression Metrics

| Model | R² ↑ | RMSE ↓ | MAE ↓ | MSE ↓ |
|:---|:---:|:---:|:---:|:---:|
| Random Forest | 0.7991 | 0.02237 | 0.01756 | 0.000500 |
| XGBoost | 0.9924 | 0.00435 | 0.00326 | 0.000019 |
| SVM (RBF) | 0.9881 | 0.00543 | 0.00387 | 0.000030 |
| **LSTM (NumPy)** | **0.9989** | **0.00167** | **0.00122** | **0.000003** |
| Ensemble Average | 0.9830 | 0.00650 | 0.00468 | 0.000042 |

### Table 2 — Classification Metrics (Threshold = 0.50)

| Model | Accuracy | Precision | Sensitivity | Specificity | F1-Score | AUC-ROC | Cohen's κ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Random Forest | 89.41% | 0.9196 | 0.8718 | 0.9181 | 0.8950 | 0.9671 | 0.7883 |
| XGBoost | 97.26% | 0.9910 | 0.9558 | 0.9907 | 0.9731 | 0.9982 | 0.9452 |
| SVM (RBF) | 96.43% | 0.9963 | 0.9345 | 0.9963 | 0.9644 | 0.9987 | 0.9287 |
| **LSTM (NumPy)** | **99.26%** | **0.9998** | **0.9859** | **0.9998** | **0.9928** | **1.0000** | **0.9852** |
| Ensemble Average | **97.35%** | 0.9954 | 0.9533 | 0.9952 | 0.9739 | **0.9990** | 0.9470 |

The LSTM achieves the highest individual performance across all metrics: Accuracy = **99.26%**, AUC-ROC = **1.0000**, Cohen's κ = **0.9852**.

---

## Comparison with Published Research

| Study | Year | Venue | Models | Best Accuracy | Best AUC-ROC |
|:---|:---:|:---|:---:|:---:|:---:|
| Rifath et al. | 2024 | *Environmental Challenges* (Elsevier) | RF, XGBoost, SVM, LR | 0.93–0.95 | ~0.95 |
| Song et al. | 2019 | *MDPI Water* | LSTM + XAJ Hybrid | NSE > 0.70 | — |
| Al-Rawas et al. | 2024 | *Scientific Reports* (Nature) | H2O AutoML Ensemble | — | RMSE ≈ 2.275 |
| Oddo et al. | 2024 | *Frontiers in Water* | ConvLSTM, LSTM | NSE 0.05–0.76 | — |
| **FloodSense AI** | **2026** | **This work** | **RF, XGBoost, SVM, LSTM** | **99.26%** | **1.0000** |

### Model-level accuracy comparison

| Model | Rifath et al. (2024) | This Study | Change |
|:---|:---:|:---:|:---:|
| Random Forest | 0.93–0.95 | 0.8941 | — *(different dataset)* |
| XGBoost | 0.92–0.95 | 0.9726 | +2.3% to +5.3% |
| SVM | 0.85–0.92 | 0.9643 | +4.3% to +11.4% |
| LSTM | NSE > 0.70 | R² = 0.9989 | Substantial improvement |

> Direct numerical comparison is approximate — studies differ in dataset size, geography, and evaluation protocol.

---

## Input Features

The model accepts 20 environmental and socioeconomic features (scale 0–15), augmented with 4 derived composite scores:

| Domain | Features |
|:---|:---|
| Climate | Monsoon Intensity, Climate Change |
| Geography | Topography Drainage, Coastal Vulnerability, Landslides, Watersheds |
| Infrastructure | River Management, Dam Quality, Drainage Systems, Deteriorating Infrastructure |
| Environment | Deforestation, Siltation, Agricultural Practices, Wetland Loss |
| Socioeconomic | Urbanization, Encroachments, Population Score |
| Policy | Ineffective Disaster Preparedness, Inadequate Planning, Political Factors |
| Derived (computed) | Risk Index, Infrastructure Score, Vulnerability Score, Management Score |

---

## Risk Classification

| Probability | Risk Level | Recommended Action |
|:---:|:---:|:---|
| < 30% | Very Low | Standard monitoring; no intervention required |
| 30–45% | Low | Heightened vigilance; brief local monitoring teams |
| 45–55% | Moderate | Issue public advisory; pre-position emergency resources |
| 55–70% | High | Activate early warning system; prepare evacuation routes |
| > 70% | Very High | Issue mandatory evacuation orders immediately |

---

## Installation

**Prerequisites:** Python 3.10 or higher (tested on 3.13), pip

```bash
# Clone the repository
git clone https://github.com/namanraj77/floodsense-ai.git
cd floodsense-ai

# Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install flask flask-cors pandas numpy scikit-learn xgboost joblib
```

> TensorFlow and PyTorch are **not** required. The LSTM is implemented in NumPy.

---

## Running the Application

Pre-trained model artefacts are included. No retraining is necessary.

```bash
python -X utf8 backend/app.py
```

Open `http://localhost:5000` in a browser. The dashboard provides:

- **Dashboard** — System status and overview
- **Predict** — Input environmental factors and retrieve a flood probability estimate
- **Analytics** — Model performance charts and feature importance rankings
- **Models** — Side-by-side individual model comparison
- **About** — Research context and methodology

---

## Retraining Models (Optional)

```bash
python -X utf8 backend/train_models.py
```

Estimated training time: 15–30 minutes on a standard CPU.

---

## Running the Evaluation Suite

```bash
python -X utf8 evaluate_models.py
```

Results are printed to the console and written to `evaluation_results.json`.

---

## API Reference

| Endpoint | Method | Description |
|:---|:---:|:---|
| `/` | GET | Serve the web application |
| `/api/status` | GET | System health and loaded model status |
| `/api/predict` | POST | Single-sample flood probability prediction |
| `/api/metrics` | GET | Model performance metrics |
| `/api/features` | GET | Feature names and descriptions |
| `/api/batch-predict` | POST | Bulk prediction for multiple scenarios |
| `/api/historical-data` | GET | Dataset statistics and distribution |
| `/api/sample-predictions` | GET | Sample predictions for dashboard seeding |

### Example request

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "MonsoonIntensity": 8,
    "TopographyDrainage": 6,
    "RiverManagement": 4,
    "Deforestation": 7,
    "Urbanization": 9,
    "ClimateChange": 8,
    "DamsQuality": 5,
    "Siltation": 6,
    "AgriculturalPractices": 5,
    "Encroachments": 7,
    "IneffectiveDisasterPreparedness": 8,
    "DrainageSystems": 4,
    "CoastalVulnerability": 6,
    "Landslides": 5,
    "Watersheds": 6,
    "DeterioratingInfrastructure": 7,
    "PopulationScore": 8,
    "WetlandLoss": 6,
    "InadequatePlanning": 7,
    "PoliticalFactors": 6
  }'
```

---

## Dependencies

| Package | Version | Purpose |
|:---|:---:|:---|
| Flask | 3.0.0 | Web framework and REST API |
| Flask-CORS | 4.0.0 | Cross-origin resource sharing |
| pandas | ≥ 2.0 | Data loading and manipulation |
| numpy | ≥ 1.26 | Numerical computation and LSTM implementation |
| scikit-learn | ≥ 1.3 | Random Forest, SVM, preprocessing, metrics |
| xgboost | ≥ 2.0 | Gradient boosting model |
| joblib | ≥ 1.3 | Model serialisation |

---

## References

1. Rifath, A. R., et al. (2024). Flash flood prediction modeling in the hilly regions of Southeastern Bangladesh. *Environmental Challenges*, 17, 101029. https://doi.org/10.1016/j.envc.2024.101029

2. Song, T., et al. (2019). Flash flood forecasting based on long short-term memory networks. *Water*, 12(1), 109. https://doi.org/10.3390/w12010109

3. Al-Rawas, G., et al. (2024). Near future flash flood prediction in an arid region under climate change. *Scientific Reports*, 14. https://doi.org/10.1038/s41598-024-76232-0

4. Oddo, P. C., et al. (2024). Deep convolutional LSTM for improved flash flood prediction. *Frontiers in Water*, 6, 1346104. https://doi.org/10.3389/frwa.2024.1346104

---

## Authors

**Naman Raj · Daksh Chahuan**  
Master of Computer Applications, IILM University, Greater Noida — 2026

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
