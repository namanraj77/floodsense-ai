<div align="center">

# 🌊 FloodSense AI
### Intelligent Flood Risk Assessment & Early Warning System

[![Python](https://img.shields.io/badge/Python-3.13+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-189AB4?style=for-the-badge)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**Leveraging 4 advanced ML models — LSTM, Random Forest, XGBoost & SVM — to deliver precise, real-time flood probability predictions with ensemble averaging.**

[🚀 Live Demo](#-running-the-application) • [📊 Model Performance](#-model-performance) • [📚 Research Comparison](#-comparison-with-published-research) • [🛠 Installation](#-installation)

</div>

---

## 📖 Overview

FloodSense AI is a machine learning–powered flood prediction system developed as part of an MCA research project. It processes **20 environmental and socioeconomic risk factors** to predict flood probability using an ensemble of four ML models, served through a modern Flask web application.

> **Research Context:** This system was developed and evaluated against recent peer-reviewed literature in flood susceptibility mapping and ML-based early warning systems.

---

## ✨ Features

- 🤖 **4 ML Models** — LSTM (NumPy), Random Forest, XGBoost, SVM with ensemble averaging
- 🌐 **Full-Stack Web App** — Interactive dashboard served via Flask at `localhost:5000`
- 📊 **Real-Time Analytics** — Feature importance, model comparison, historical distribution charts
- 🗺️ **Risk Mapping** — 5-tier risk classification (Very Low → Very High) with actionable recommendations
- ⚡ **No TensorFlow Required** — Custom NumPy LSTM implementation, compatible with Python 3.13+
- 🔌 **REST API** — `/api/predict`, `/api/metrics`, `/api/batch-predict`, `/api/historical-data`

---

## 🏗️ Architecture

```
flood_prediction/
├── backend/
│   ├── app.py              # Flask REST API server
│   └── train_models.py     # Model training + NumPy LSTM implementation
├── frontend/
│   └── index.html          # Single-page web application
├── models/                 # Saved trained models (.pkl, .json)
├── data/
│   └── flood.csv           # Dataset (50,000 samples)
├── evaluate_models.py      # Evaluation script (research-grade metrics)
├── requirements.txt
└── README.md
```

---

## 📊 Model Performance

> **Evaluation Setup:**
> - Dataset: `flood.csv` — **50,000 samples**, 20 features + 4 derived = 24 total
> - Train/Test Split: **80% / 20%** (40,000 train | 10,000 test) — `random_state=42`
> - Classification Threshold: **0.50** (flood / no-flood)
> - Scaler: `StandardScaler` fitted on training data only

### Table 1 — Regression Metrics

| Model | R² ↑ | RMSE ↓ | MAE ↓ | MSE ↓ |
|:---|:---:|:---:|:---:|:---:|
| Random Forest | 0.7991 | 0.02237 | 0.01756 | 0.000500 |
| XGBoost | 0.9924 | 0.00435 | 0.00326 | 0.000019 |
| SVM (RBF) | 0.9881 | 0.00543 | 0.00387 | 0.000030 |
| **LSTM (NumPy)** | **0.9989** | **0.00167** | **0.00122** | **0.000003** |
| Ensemble Avg | 0.9830 | 0.00650 | 0.00468 | 0.000042 |

### Table 2 — Classification Metrics (Threshold = 0.50)

| Model | Accuracy | Precision | Sensitivity | Specificity | F1-Score | AUC-ROC | Cohen's κ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Random Forest | 89.41% | 0.9196 | 0.8718 | 0.9181 | 0.8950 | 0.9671 | 0.7883 |
| XGBoost | 97.26% | 0.9910 | 0.9558 | 0.9907 | 0.9731 | 0.9982 | 0.9452 |
| SVM (RBF) | 96.43% | 0.9963 | 0.9345 | 0.9963 | 0.9644 | 0.9987 | 0.9287 |
| **LSTM (NumPy)** | **99.26%** | **0.9998** | **0.9859** | **0.9998** | **0.9928** | **1.0000** | **0.9852** |
| **Ensemble Avg** | **97.35%** | 0.9954 | 0.9533 | 0.9952 | 0.9739 | **0.9990** | 0.9470 |

> 🏆 **LSTM achieves the highest performance** across all metrics: Accuracy = **99.26%**, AUC-ROC = **1.00**, Cohen's κ = **0.9852** (near-perfect agreement).

---

## 📚 Comparison with Published Research

The following table benchmarks FloodSense AI against recent peer-reviewed flood prediction studies:

| Study | Year | Journal | Models Used | Best Accuracy | Best AUC-ROC |
|:---|:---:|:---|:---:|:---:|:---:|
| Rifath et al. | 2024 | *Environmental Challenges* (Elsevier) | RF, XGBoost, SVM, LR | 0.93–0.95 | ~0.95 |
| Song et al. | 2019 | *MDPI Water* | LSTM, XAJ | NSE > 0.70 | — |
| Al-Rawas et al. | 2024 | *Scientific Reports* (Nature) | H2O AutoML Ensemble | — | RMSE ≈ 2.275 |
| Oddo et al. | 2024 | *Frontiers in Water* | ConvLSTM, LSTM | NSE 0.05–0.76 | — |
| **FloodSense AI** | **2025** | **This Work** | **RF, XGBoost, SVM, LSTM** | **99.26%** | **1.0000** |

### Model-Level Comparison (Accuracy)

| Model | Rifath et al. (2024) | **This Study** | Δ Improvement |
|:---|:---:|:---:|:---:|
| Random Forest | 0.93–0.95 | **0.8941** | — *(different dataset)* |
| XGBoost | 0.92–0.95 | **0.9726** | ↑ +2.3% to +5.3% |
| SVM | 0.85–0.92 | **0.9643** | ↑ +4.3% to +11.4% |
| LSTM | NSE > 0.70 | **R² = 0.9989** | ↑ Substantial |

> **Note:** Direct numerical comparison is approximate as studies use different datasets and geographic contexts. The key takeaway is that FloodSense AI achieves competitive or superior results, particularly in XGBoost and SVM performance.

---

## 🧠 Input Features

The model accepts **20 environmental and socioeconomic factors** (scale 0–15), plus 4 derived composite scores:

| Category | Features |
|:---|:---|
| 🌧️ **Climate** | Monsoon Intensity, Climate Change |
| 🏔️ **Geography** | Topography Drainage, Coastal Vulnerability, Landslides, Watersheds |
| 🏗️ **Infrastructure** | River Management, Dams Quality, Drainage Systems, Deteriorating Infrastructure |
| 🌿 **Environment** | Deforestation, Siltation, Agricultural Practices, Wetland Loss |
| 🏙️ **Socioeconomic** | Urbanization, Encroachments, Population Score |
| 📋 **Policy** | Ineffective Disaster Preparedness, Inadequate Planning, Political Factors |
| 🔢 **Derived** | Risk Index, Infrastructure Score, Vulnerability Score, Management Score |

---

## 🔁 Risk Classification

| Flood Probability | Risk Level | Action |
|:---:|:---:|:---|
| < 30% | 🟢 Very Low | Standard monitoring |
| 30–45% | 🟡 Low | Heightened vigilance |
| 45–55% | 🟠 Moderate | Alert residents, pre-position resources |
| 55–70% | 🔴 High | Activate early warning system |
| > 70% | 🚨 Very High | Issue evacuation orders immediately |

---

## 🛠 Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- pip

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/floodsense-ai.git
cd floodsense-ai

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install flask flask-cors pandas numpy scikit-learn xgboost joblib

# (Note: tensorflow is NOT required — we use a custom NumPy LSTM)
```

---

## 🚀 Running the Application

> ⚠️ **Models are pre-trained and included.** No retraining required.

```bash
# Start the Flask backend server
python -X utf8 backend/app.py
```

Then open your browser at **http://localhost:5000**

The web dashboard includes:
- 🏠 **Dashboard** — Overview & system status
- 🔮 **Predict** — Input environmental factors and get flood probability
- 📊 **Analytics** — Model performance charts & feature importance
- 🤖 **Models** — Individual model comparison
- ℹ️ **About** — Research context and methodology

---

## 🔁 Re-Training Models (Optional)

If you want to retrain models from scratch:

```bash
python -X utf8 backend/train_models.py
```

> ⏳ Estimated training time: 15–30 minutes on a standard CPU.

---

## 🧪 Running Evaluation

To reproduce the research paper metrics:

```bash
python -X utf8 evaluate_models.py
```

Results are printed to console and saved to `evaluation_results.json`.

---

## 🔌 API Reference

| Endpoint | Method | Description |
|:---|:---:|:---|
| `/` | GET | Serve the web application |
| `/api/status` | GET | Check system status & loaded models |
| `/api/predict` | POST | Get flood probability prediction |
| `/api/metrics` | GET | Get model performance metrics |
| `/api/features` | GET | Get feature descriptions |
| `/api/batch-predict` | POST | Batch prediction for multiple scenarios |
| `/api/historical-data` | GET | Dataset statistics & distribution |
| `/api/sample-predictions` | GET | Sample predictions for visualization |

### Example API Call

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

## 📦 Dependencies

| Package | Version | Purpose |
|:---|:---:|:---|
| Flask | 3.0.0 | Web framework & REST API |
| Flask-CORS | 4.0.0 | Cross-origin resource sharing |
| pandas | ≥ 2.0 | Data loading & manipulation |
| numpy | ≥ 1.26 | Numerical computation & LSTM |
| scikit-learn | ≥ 1.3 | RF, SVM, preprocessing, metrics |
| xgboost | ≥ 2.0 | Gradient boosting model |
| joblib | ≥ 1.3 | Model serialization |

> No TensorFlow or PyTorch required.

---

## 📄 References

1. **Rifath, A.R., et al.** (2024). *Flash flood prediction modeling in the hilly regions of Southeastern Bangladesh: A machine learning attempt on present and future climate scenarios.* Environmental Challenges, 17, 101029. https://doi.org/10.1016/j.envc.2024.101029

2. **Song, T., et al.** (2019). *Flash Flood Forecasting Based on Long Short-Term Memory Networks.* Water, 12(1), 109. https://doi.org/10.3390/w12010109

3. **Al-Rawas, G., et al.** (2024). *Near future flash flood prediction in an arid region under climate change.* Scientific Reports, 14, s41598-024-76232-0. https://doi.org/10.1038/s41598-024-76232-0

4. **Oddo, P.C., et al.** (2024). *Deep Convolutional LSTM for improved flash flood prediction.* Frontiers in Water, 6, 1346104. https://doi.org/10.3389/frwa.2024.1346104

---

## 👥 Authors

**Daksh** — MCA Research Project, IILM University

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ for flood early warning research

⭐ If this project helped your research, please give it a star!

</div>
