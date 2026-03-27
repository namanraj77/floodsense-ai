"""
Flood Prediction Model Training Script
Trains LSTM (NumPy), Random Forest, XGBoost, and SVM models
Compatible with Python 3.14+ (no TensorFlow/PyTorch required)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

# Fix Windows console encoding for emoji/unicode
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

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


# ─────────────────────────────────────────────
#  NumPy LSTM Implementation (no deep-learning lib needed)
# ─────────────────────────────────────────────
class NumPyLSTM:
    """
    A single-layer LSTM + dense head built with NumPy.
    Architecture: LSTM(hidden) -> Dense(hidden//2) -> Dense(1)
    Training: Mini-batch SGD with Adam optimiser.
    """

    def __init__(self, input_size, hidden_size=64, lr=0.001, epochs=80,
                 batch_size=128, seed=42):
        np.random.seed(seed)
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.lr          = lr
        self.epochs      = epochs
        self.batch_size  = batch_size

        s = hidden_size
        n = input_size + hidden_size            # concat size for gate weights

        # LSTM weights  W[input+hidden -> 4 gates]
        self.Wf = np.random.randn(n, s) * 0.05
        self.bf = np.zeros(s)
        self.Wi = np.random.randn(n, s) * 0.05
        self.bi = np.zeros(s)
        self.Wo = np.random.randn(n, s) * 0.05
        self.bo = np.zeros(s)
        self.Wg = np.random.randn(n, s) * 0.05
        self.bg = np.zeros(s)

        # Dense 1  hidden -> hidden//2
        hs2 = max(hidden_size // 2, 1)
        self.W1 = np.random.randn(s, hs2) * 0.05
        self.b1 = np.zeros(hs2)

        # Dense 2  hidden//2 -> 1
        self.W2 = np.random.randn(hs2, 1) * 0.05
        self.b2 = np.zeros(1)

        # Adam moments
        self._params = ['Wf','bf','Wi','bi','Wo','bo','Wg','bg','W1','b1','W2','b2']
        self._m = {p: np.zeros_like(getattr(self, p)) for p in self._params}
        self._v = {p: np.zeros_like(getattr(self, p)) for p in self._params}
        self._t = 0

    # ── gates ──────────────────────────────────────────────
    @staticmethod
    def _sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    @staticmethod
    def _tanh(x): return np.tanh(np.clip(x, -30, 30))

    # ── forward for a single sample  x: (input_size,) ──────
    def _forward_one(self, x):
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)
        xh = np.concatenate([x, h])
        f  = self._sigmoid(xh @ self.Wf + self.bf)
        i  = self._sigmoid(xh @ self.Wi + self.bi)
        o  = self._sigmoid(xh @ self.Wo + self.bo)
        g  = self._tanh   (xh @ self.Wg + self.bg)
        c  = f * c + i * g
        h  = o * self._tanh(c)
        # dense layers
        d1 = np.maximum(0, h @ self.W1 + self.b1)   # ReLU
        y  = self._sigmoid(d1 @ self.W2 + self.b2)  # sigmoid -> [0,1]
        return float(y[0]), h, c, xh, f, i, o, g

    # ── batch forward (inference only) ─────────────────────
    def predict(self, X):
        return np.array([self._forward_one(x)[0] for x in X])

    # ── Adam parameter update ───────────────────────────────
    def _adam_update(self, grad, name):
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        self._t += 1
        self._m[name] = beta1 * self._m[name] + (1 - beta1) * grad
        self._v[name] = beta2 * self._v[name] + (1 - beta2) * grad ** 2
        m_hat = self._m[name] / (1 - beta1 ** self._t)
        v_hat = self._v[name] / (1 - beta2 ** self._t)
        update = self.lr * m_hat / (np.sqrt(v_hat) + eps)
        setattr(self, name, getattr(self, name) - update)

    # ── training (BPTT depth=1, single time-step) ───────────
    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        n = len(X_train)
        best_val_loss = np.inf
        patience_cnt  = 0
        patience      = 12
        best_params   = None

        for epoch in range(self.epochs):
            # Shuffle
            idx = np.random.permutation(n)
            Xs, ys = X_train[idx], y_train[idx]

            for start in range(0, n, self.batch_size):
                Xb = Xs[start:start + self.batch_size]
                yb = ys[start:start + self.batch_size]
                bs = len(Xb)

                # Accumulators
                dWf=np.zeros_like(self.Wf); dbf=np.zeros_like(self.bf)
                dWi=np.zeros_like(self.Wi); dbi=np.zeros_like(self.bi)
                dWo=np.zeros_like(self.Wo); dbo=np.zeros_like(self.bo)
                dWg=np.zeros_like(self.Wg); dbg=np.zeros_like(self.bg)
                dW1=np.zeros_like(self.W1); db1=np.zeros_like(self.b1)
                dW2=np.zeros_like(self.W2); db2=np.zeros_like(self.b2)

                for x, y_true in zip(Xb, yb):
                    h = np.zeros(self.hidden_size)
                    c = np.zeros(self.hidden_size)
                    xh = np.concatenate([x, h])
                    f  = self._sigmoid(xh @ self.Wf + self.bf)
                    i  = self._sigmoid(xh @ self.Wi + self.bi)
                    o  = self._sigmoid(xh @ self.Wo + self.bo)
                    g  = self._tanh   (xh @ self.Wg + self.bg)
                    c_new = f * c + i * g
                    tanh_c = self._tanh(c_new)
                    h_new  = o * tanh_c
                    d1_pre = h_new @ self.W1 + self.b1
                    d1     = np.maximum(0, d1_pre)
                    y_pre  = d1 @ self.W2 + self.b2
                    pred   = self._sigmoid(y_pre)

                    # Loss gradient (MSE)
                    dL_dy  = 2.0 * (pred - y_true) / bs       # (1,)
                    dy_dz  = pred * (1 - pred)                  # sigmoid'
                    delta2 = dL_dy * dy_dz                      # (1,)

                    dW2 += d1[:, None] @ delta2[None, :]        # (hs2,1)
                    db2 += delta2

                    # Backprop through dense-1 (ReLU)
                    dd1 = delta2 @ self.W2.T                    # (hs2,)
                    dd1_pre = dd1 * (d1_pre > 0)               # ReLU'

                    dW1 += h_new[:, None] @ dd1_pre[None, :]   # (hs,hs2)
                    db1 += dd1_pre

                    # Backprop through h_new -> LSTM weights
                    dh   = dd1_pre @ self.W1.T                  # (hs,)
                    do   = dh * tanh_c
                    dc   = dh * o * (1 - tanh_c**2)

                    do_pre = do * o * (1 - o)
                    dg_pre = dc * i * (1 - g**2)
                    di_pre = dc * g * i * (1 - i)
                    df_pre = dc * c * f * (1 - f)

                    dWf += xh[:, None] @ df_pre[None, :]
                    dbf += df_pre
                    dWi += xh[:, None] @ di_pre[None, :]
                    dbi += di_pre
                    dWo += xh[:, None] @ do_pre[None, :]
                    dbo += do_pre
                    dWg += xh[:, None] @ dg_pre[None, :]
                    dbg += dg_pre

                # Update params with Adam
                grads = dict(Wf=dWf, bf=dbf, Wi=dWi, bi=dbi,
                             Wo=dWo, bo=dbo, Wg=dWg, bg=dbg,
                             W1=dW1, b1=db1, W2=dW2, b2=db2)
                for name, grad in grads.items():
                    self._adam_update(grad, name)

            # Validation loss
            if X_val is not None:
                val_preds = self.predict(X_val)
                val_loss  = float(np.mean((val_preds - y_val)**2))
                if val_loss < best_val_loss - 1e-5:
                    best_val_loss = val_loss
                    patience_cnt  = 0
                    best_params   = {p: getattr(self, p).copy() for p in self._params}
                else:
                    patience_cnt += 1
                if patience_cnt >= patience:
                    if verbose:
                        print(f"   Early stopping at epoch {epoch+1}, val_loss={best_val_loss:.5f}")
                    break

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"   Epoch {epoch+1:3d}/{self.epochs} | val_loss={val_loss:.5f}")

        # Restore best weights
        if best_params:
            for p, v in best_params.items():
                setattr(self, p, v)
        return self

    def save(self, path):
        weights = {p: getattr(self, p).tolist() for p in self._params}
        meta = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'lr': self.lr,
            'epochs': self.epochs,
            'batch_size': self.batch_size
        }
        with open(path, 'w') as f:
            json.dump({'meta': meta, 'weights': weights}, f)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)
        m = data['meta']
        obj = cls(m['input_size'], m['hidden_size'], m['lr'], m['epochs'], m['batch_size'])
        for p, v in data['weights'].items():
            setattr(obj, p, np.array(v))
        return obj


# ─────────────────────────────────────────────
#  Data loading & feature engineering
# ─────────────────────────────────────────────
def load_and_preprocess_data():
    print("📊 Loading dataset...")
    df = pd.read_csv(os.path.join(DATA_DIR, 'flood.csv'))
    print(f"   Shape: {df.shape}")

    # Derived features
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
    print(f"   Target range: [{y.min():.3f}, {y.max():.3f}]")
    return X, y, ext, df


# ─────────────────────────────────────────────
#  Metrics helper
# ─────────────────────────────────────────────
def compute_metrics(y_true, y_pred, name):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    acc  = np.mean(np.abs(y_true - y_pred) <= 0.05) * 100
    print(f"   ✅ {name:<22} R²={r2:.4f}  RMSE={rmse:.5f}  MAE={mae:.5f}  Acc={acc:.1f}%")
    return {'mse': float(mse), 'rmse': float(rmse), 'mae': float(mae),
            'r2': float(r2), 'accuracy': float(acc)}


# ─────────────────────────────────────────────
#  Individual model trainers
# ─────────────────────────────────────────────
def train_random_forest(Xtr, Xte, ytr, yte):
    print("\n🌲 Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=300, max_depth=12, max_features='sqrt',
                               random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr)
    preds = rf.predict(Xte)
    m = compute_metrics(yte, preds, "Random Forest")
    joblib.dump(rf, os.path.join(MODELS_DIR, 'random_forest.pkl'))
    return rf, m, preds, rf.feature_importances_.tolist()


def train_xgboost(Xtr, Xte, ytr, yte):
    print("\n⚡ Training XGBoost...")
    xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                        subsample=0.8, colsample_bytree=0.8,
                        reg_alpha=0.1, reg_lambda=1.0, random_state=42,
                        n_jobs=-1, verbosity=0)
    xgb.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
    preds = xgb.predict(Xte)
    m = compute_metrics(yte, preds, "XGBoost")
    joblib.dump(xgb, os.path.join(MODELS_DIR, 'xgboost.pkl'))
    return xgb, m, preds, xgb.feature_importances_.tolist()


def train_svm(Xtr_s, Xte_s, ytr, yte):
    print("\n🔵 Training SVM...")
    svm = SVR(kernel='rbf', C=10.0, gamma='scale', epsilon=0.01)
    svm.fit(Xtr_s, ytr)
    preds = np.clip(svm.predict(Xte_s), 0, 1)
    m = compute_metrics(yte, preds, "SVM")
    joblib.dump(svm, os.path.join(MODELS_DIR, 'svm.pkl'))
    return svm, m, preds


def train_lstm(Xtr_s, Xte_s, ytr, yte):
    print("\n🔴 Training LSTM (NumPy)...")
    n_feat = Xtr_s.shape[1]

    # Split a small validation set
    split = int(0.85 * len(Xtr_s))
    Xtrain, Xval = Xtr_s[:split], Xtr_s[split:]
    ytrain, yval = ytr[:split], ytr[split:]

    lstm = NumPyLSTM(input_size=n_feat, hidden_size=64, lr=0.001,
                     epochs=80, batch_size=256, seed=42)
    lstm.fit(Xtrain, ytrain, Xval, yval, verbose=True)

    preds = np.clip(lstm.predict(Xte_s), 0, 1)
    m = compute_metrics(yte, preds, "LSTM (NumPy)")

    lstm.save(os.path.join(MODELS_DIR, 'lstm_model.json'))
    return lstm, m, preds


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main():
    print("=" * 62)
    print("🌊  FLOOD PREDICTION — MODEL TRAINING")
    print("=" * 62)

    X, y, feature_names, df = load_and_preprocess_data()

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\n📌 Train: {len(Xtr)}  |  Test: {len(Xte)}")

    scaler = StandardScaler()
    Xtr_s  = scaler.fit_transform(Xtr)
    Xte_s  = scaler.transform(Xte)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))

    rf,  rf_m,  rf_p,  rf_fi  = train_random_forest(Xtr, Xte, ytr, yte)
    xgb, xgb_m, xgb_p, xgb_fi = train_xgboost(Xtr, Xte, ytr, yte)
    svm, svm_m, svm_p          = train_svm(Xtr_s, Xte_s, ytr, yte)
    lstm,lstm_m, lstm_p        = train_lstm(Xtr_s, Xte_s, ytr, yte)

    # Ensemble
    ens_p = (rf_p + xgb_p + svm_p + lstm_p) / 4
    ens_m = compute_metrics(yte, ens_p, "Ensemble Avg")

    print("\n" + "=" * 62)
    print("📊  FINAL SUMMARY")
    print("=" * 62)
    for name, m in [('LSTM', lstm_m), ('Random Forest', rf_m),
                    ('XGBoost', xgb_m), ('SVM', svm_m), ('Ensemble', ens_m)]:
        print(f"  {name:<22} R²={m['r2']:.4f}  RMSE={m['rmse']:.5f}")

    # Feature importance
    min_len = min(len(rf_fi), len(xgb_fi), len(feature_names))
    fi_dict = {feature_names[i]: float((rf_fi[i] + xgb_fi[i]) / 2)
               for i in range(min_len)}
    fi_sorted = dict(sorted(fi_dict.items(), key=lambda x: x[1], reverse=True))

    # Sample predictions for frontend visualisation
    idx = np.random.choice(len(Xte), min(200, len(Xte)), replace=False)
    sample = {
        'actual':   yte[idx].tolist(),
        'rf':       rf_p[idx].tolist(),
        'xgb':      xgb_p[idx].tolist(),
        'svm':      svm_p[idx].tolist(),
        'lstm':     lstm_p[idx].tolist(),
        'ensemble': ens_p[idx].tolist()
    }

    meta = {
        'features':          feature_names,
        'original_features': FEATURES,
        'target':            TARGET,
        'n_samples':         int(len(X)),
        'n_features':        int(len(feature_names)),
        'target_min':        float(y.min()),
        'target_max':        float(y.max()),
        'target_mean':       float(y.mean()),
        'metrics': {
            'lstm':          lstm_m,
            'random_forest': rf_m,
            'xgboost':       xgb_m,
            'svm':           svm_m,
            'ensemble':      ens_m
        },
        'feature_importance':   fi_sorted,
        'sample_predictions':   sample,
        'training_complete':    True
    }

    with open(os.path.join(MODELS_DIR, 'model_metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ All models saved → {MODELS_DIR}")
    print("✅ Training complete!")
    print("=" * 62)


if __name__ == '__main__':
    main()
