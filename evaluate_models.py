"""
Flood Prediction Model Evaluation Script
=========================================
Evaluates all 4 trained models on the held-out test set (80/20 split, seed=42).
Produces research-paper-ready metrics:
  Regression  : R², RMSE, MAE, MSE
  Classification (threshold=0.5): Accuracy, Precision, Recall, F1, AUC-ROC, Cohen's Kappa
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, cohen_kappa_score, confusion_matrix,
    classification_report
)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Add backend to path for NumPyLSTM
sys.path.insert(0, os.path.join(BASE_DIR, 'backend'))
from train_models import NumPyLSTM

# ── Feature lists (must match training) ───────────────────────────────────
FEATURES = [
    'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement',
    'Deforestation', 'Urbanization', 'ClimateChange', 'DamsQuality',
    'Siltation', 'AgriculturalPractices', 'Encroachments',
    'IneffectiveDisasterPreparedness', 'DrainageSystems',
    'CoastalVulnerability', 'Landslides', 'Watersheds',
    'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
    'InadequatePlanning', 'PoliticalFactors'
]
TARGET = 'FloodProbability'
THRESHOLD = 0.5   # flood / no-flood decision boundary

# ── Load & preprocess data (identical to training) ─────────────────────────
def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, 'flood.csv'))

    df['RiskIndex']           = (df['MonsoonIntensity'] + df['ClimateChange'] +
                                  df['Urbanization'] + df['Deforestation']) / 4
    df['InfrastructureScore'] = (df['RiverManagement'] + df['DamsQuality'] +
                                  df['DrainageSystems']) / 3
    df['VulnerabilityScore']  = (df['CoastalVulnerability'] + df['Landslides'] +
                                  df['Encroachments'] + df['PopulationScore']) / 4
    df['ManagementScore']     = (df['IneffectiveDisasterPreparedness'] +
                                  df['InadequatePlanning'] + df['PoliticalFactors']) / 3

    ext = FEATURES + ['RiskIndex', 'InfrastructureScore', 'VulnerabilityScore', 'ManagementScore']
    X = df[ext].values.astype(np.float64)
    y = df[TARGET].values.astype(np.float64)
    return X, y

# ── Metric computation ─────────────────────────────────────────────────────
def compute_all_metrics(y_true, y_pred):
    # Regression
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    # Classification (binary at threshold)
    y_true_bin = (y_true  >= THRESHOLD).astype(int)
    y_pred_bin = (y_pred  >= THRESHOLD).astype(int)

    acc     = accuracy_score(y_true_bin, y_pred_bin)
    prec    = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec     = recall_score(y_true_bin, y_pred_bin, zero_division=0)    # = Sensitivity
    f1      = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    kappa   = cohen_kappa_score(y_true_bin, y_pred_bin)

    # Specificity = TN / (TN + FP)
    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0,1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        spec = 0.0

    # AUC-ROC (use raw probabilities)
    try:
        auc = roc_auc_score(y_true_bin, y_pred)
    except Exception:
        auc = float('nan')

    return {
        'R2':           round(r2,   4),
        'RMSE':         round(rmse, 5),
        'MAE':          round(mae,  5),
        'MSE':          round(mse,  6),
        'Accuracy':     round(acc,  4),
        'Precision':    round(prec, 4),
        'Recall':       round(rec,  4),   # = Sensitivity
        'Sensitivity':  round(rec,  4),
        'Specificity':  round(spec, 4),
        'F1_Score':     round(f1,   4),
        'AUC_ROC':      round(auc,  4),
        'Kappa':        round(kappa,4),
    }

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  FLOOD PREDICTION — COMPREHENSIVE MODEL EVALUATION")
    print("  (Research Paper Grade | 80/20 Split | seed=42)")
    print("=" * 70)

    # 1. Reload data with identical preprocessing & split
    print("\n[1/3] Loading and splitting dataset...")
    X, y = load_data()
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"      Test set : {len(X_test)} samples")
    print(f"      Target   : min={y_test.min():.3f}  max={y_test.max():.3f}  mean={y_test.mean():.3f}")

    # 2. Scale (load the saved scaler)
    print("\n[2/3] Loading saved scaler and models...")
    scaler   = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    X_test_s = scaler.transform(X_test)

    # 3. Load models
    rf   = joblib.load(os.path.join(MODELS_DIR, 'random_forest.pkl'))
    xgb  = joblib.load(os.path.join(MODELS_DIR, 'xgboost.pkl'))
    svm  = joblib.load(os.path.join(MODELS_DIR, 'svm.pkl'))
    lstm = NumPyLSTM.load(os.path.join(MODELS_DIR, 'lstm_model.json'))
    print("      All 4 models loaded.")

    # 4. Predict
    print("\n[3/3] Running predictions on test set...")
    preds = {
        'Random Forest': np.clip(rf.predict(X_test), 0, 1),
        'XGBoost':       np.clip(xgb.predict(X_test), 0, 1),
        'SVM':           np.clip(svm.predict(X_test_s), 0, 1),
        'LSTM':          np.clip(lstm.predict(X_test_s), 0, 1),
    }
    preds['Ensemble'] = np.mean(list(preds.values()), axis=0)

    # 5. Compute metrics
    results = {}
    for name, pred in preds.items():
        results[name] = compute_all_metrics(y_test, pred)

    # ── Pretty print tables ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TABLE 1 — REGRESSION METRICS")
    print("=" * 70)
    header = f"{'Model':<18} {'R²':>8} {'RMSE':>10} {'MAE':>10} {'MSE':>12}"
    print(header)
    print("-" * 60)
    for name, m in results.items():
        print(f"{name:<18} {m['R2']:>8.4f} {m['RMSE']:>10.5f} {m['MAE']:>10.5f} {m['MSE']:>12.6f}")

    print("\n" + "=" * 70)
    print("  TABLE 2 — CLASSIFICATION METRICS  (threshold = 0.50)")
    print("=" * 70)
    header2 = f"{'Model':<18} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'Specificity':>12} {'F1':>8} {'AUC-ROC':>9} {'Kappa':>8}"
    print(header2)
    print("-" * 85)
    for name, m in results.items():
        print(f"{name:<18} {m['Accuracy']:>9.4f} {m['Precision']:>10.4f} {m['Recall']:>8.4f} "
              f"{m['Specificity']:>12.4f} {m['F1_Score']:>8.4f} {m['AUC_ROC']:>9.4f} {m['Kappa']:>8.4f}")

    # ── Save to JSON ───────────────────────────────────────────────────────
    out_path = os.path.join(BASE_DIR, 'evaluation_results.json')
    with open(out_path, 'w') as f:
        json.dump({
            'dataset_info': {
                'total_samples': len(X),
                'test_samples':  len(X_test),
                'split_ratio':   '80/20',
                'random_state':  42,
                'threshold':     THRESHOLD,
                'n_features':    X.shape[1],
            },
            'results': results
        }, f, indent=2)

    print(f"\n  Results saved to: {out_path}")
    print("=" * 70)
    print("  EVALUATION COMPLETE")
    print("=" * 70)

if __name__ == '__main__':
    main()
