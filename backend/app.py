"""
Flask Backend API for Flood Prediction System
Compatible with Python 3.14+ — uses NumPy LSTM (no TF/PyTorch)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Import our NumPy-based LSTM from the training module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_models import NumPyLSTM

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Global model storage
models = {}
scaler = None
metadata = {}

ORIGINAL_FEATURES = [
    'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement',
    'Deforestation', 'Urbanization', 'ClimateChange', 'DamsQuality',
    'Siltation', 'AgriculturalPractices', 'Encroachments',
    'IneffectiveDisasterPreparedness', 'DrainageSystems',
    'CoastalVulnerability', 'Landslides', 'Watersheds',
    'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
    'InadequatePlanning', 'PoliticalFactors'
]


def load_models():
    """Load all trained models"""
    global models, scaler, metadata
    
    print("📦 Loading models...")
    
    try:
        models['rf'] = joblib.load(os.path.join(MODELS_DIR, 'random_forest.pkl'))
        print("  ✅ Random Forest loaded")
    except Exception as e:
        print(f"  ❌ RF load failed: {e}")
    
    try:
        models['xgb'] = joblib.load(os.path.join(MODELS_DIR, 'xgboost.pkl'))
        print("  ✅ XGBoost loaded")
    except Exception as e:
        print(f"  ❌ XGB load failed: {e}")
    
    try:
        models['svm'] = joblib.load(os.path.join(MODELS_DIR, 'svm.pkl'))
        print("  ✅ SVM loaded")
    except Exception as e:
        print(f"  ❌ SVM load failed: {e}")
    
    try:
        models['lstm'] = NumPyLSTM.load(os.path.join(MODELS_DIR, 'lstm_model.json'))
        print("  ✅ LSTM (NumPy) loaded")
    except Exception as e:
        print(f"  ❌ LSTM load failed: {e}")
    
    try:
        scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
        print("  ✅ Scaler loaded")
    except Exception as e:
        print(f"  ❌ Scaler load failed: {e}")
    
    try:
        with open(os.path.join(MODELS_DIR, 'model_metadata.json'), 'r') as f:
            metadata = json.load(f)
        print("  ✅ Metadata loaded")
    except Exception as e:
        print(f"  ❌ Metadata load failed: {e}")
    
    print(f"📦 Loaded {len(models)} models successfully")


def engineer_features(input_dict):
    """Create all features including derived ones"""
    base = [float(input_dict.get(f, 5)) for f in ORIGINAL_FEATURES]
    
    monsoon, _, _, deforestation, urbanization, climate, _, _, _, _, _, _, coastal, landslides, _, _, population, _, planning, political = base
    _, _, river_mgmt, _, _, _, dams, _, _, encroachments, _, drainage, _, _, _, _, _, wetland, _, _ = base
    _, _, _, _, _, _, _, _, _, _, disaster_prep, _, _, _, _, _, _, _, _, _ = base
    
    risk_index = (base[0] + base[5] + base[4] + base[3]) / 4
    infrastructure_score = (base[2] + base[6] + base[11]) / 3
    vulnerability_score = (base[12] + base[13] + base[9] + base[16]) / 4
    management_score = (base[10] + base[18] + base[19]) / 3
    
    all_features = base + [risk_index, infrastructure_score, vulnerability_score, management_score]
    return np.array(all_features)


def get_risk_level(probability):
    """Get risk level from probability"""
    if probability < 0.3:
        return "Very Low", "#22c55e", "🟢"
    elif probability < 0.45:
        return "Low", "#86efac", "🟡"
    elif probability < 0.55:
        return "Moderate", "#fbbf24", "🟠"
    elif probability < 0.7:
        return "High", "#f97316", "🔴"
    else:
        return "Very High", "#ef4444", "🚨"


def get_recommendations(probability, input_data):
    """Generate recommendations based on prediction"""
    recommendations = []
    
    if probability >= 0.6:
        recommendations.append("🚨 Issue immediate flood warnings to local authorities")
        recommendations.append("🏃 Begin evacuation of flood-prone areas")
        recommendations.append("📦 Pre-position emergency supplies and relief kits")
    
    if input_data.get('MonsoonIntensity', 5) > 7:
        recommendations.append("🌧️ Monitor monsoon patterns closely with meteorological agencies")
    
    if input_data.get('RiverManagement', 5) < 4:
        recommendations.append("💧 Improve river embankment and flood control structures")
    
    if input_data.get('DrainageSystems', 5) < 4:
        recommendations.append("🚰 Upgrade urban drainage infrastructure immediately")
    
    if input_data.get('DamsQuality', 5) < 4:
        recommendations.append("🏗️ Inspect and repair aging dam infrastructure")
    
    if input_data.get('Deforestation', 5) > 6:
        recommendations.append("🌳 Implement reforestation programs in watershed areas")
    
    if input_data.get('Urbanization', 5) > 6:
        recommendations.append("🏙️ Enforce flood-resilient urban planning regulations")
    
    if input_data.get('IneffectiveDisasterPreparedness', 5) > 6:
        recommendations.append("📋 Strengthen disaster preparedness protocols and training")
    
    if probability >= 0.45:
        recommendations.append("📡 Activate early warning system and alert networks")
        recommendations.append("🏥 Put medical and rescue teams on standby")
    
    if len(recommendations) == 0:
        recommendations.append("✅ Maintain current flood monitoring protocols")
        recommendations.append("📊 Continue regular infrastructure maintenance")
    
    return recommendations[:6]


@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/status', methods=['GET'])
def status():
    """Check system status"""
    return jsonify({
        'status': 'online',
        'models_loaded': list(models.keys()),
        'models_count': len(models),
        'training_complete': metadata.get('training_complete', False)
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Make flood probability prediction using all 4 models"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Build feature vector
        features = engineer_features(data)
        features_2d = features.reshape(1, -1)
        
        # Scale features
        if scaler is not None:
            features_scaled = scaler.transform(features_2d)
        else:
            features_scaled = features_2d
        
        predictions = {}
        
        # Random Forest
        if 'rf' in models:
            pred = float(np.clip(models['rf'].predict(features_2d)[0], 0, 1))
            predictions['random_forest'] = pred
        
        # XGBoost
        if 'xgb' in models:
            pred = float(np.clip(models['xgb'].predict(features_2d)[0], 0, 1))
            predictions['xgboost'] = pred
        
        # SVM
        if 'svm' in models:
            pred = float(np.clip(models['svm'].predict(features_scaled)[0], 0, 1))
            predictions['svm'] = pred
        
        # LSTM (NumPy)
        if 'lstm' in models:
            pred = float(np.clip(models['lstm'].predict(features_scaled)[0], 0, 1))
            predictions['lstm'] = pred
        
        # Ensemble average
        if predictions:
            ensemble = float(np.mean(list(predictions.values())))
            predictions['ensemble'] = ensemble
        else:
            return jsonify({'error': 'No models available for prediction'}), 500
        
        # Risk assessment
        risk_level, risk_color, risk_icon = get_risk_level(ensemble)
        recommendations = get_recommendations(ensemble, data)
        
        # Confidence score (based on std dev of model predictions)
        pred_values = [v for k, v in predictions.items() if k != 'ensemble']
        confidence = float(1 - np.std(pred_values)) * 100 if len(pred_values) > 1 else 75.0
        confidence = max(0, min(100, confidence))
        
        return jsonify({
            'success': True,
            'predictions': {
                'individual': {
                    'LSTM': round(predictions.get('lstm', 0), 4),
                    'Random Forest': round(predictions.get('random_forest', 0), 4),
                    'XGBoost': round(predictions.get('xgboost', 0), 4),
                    'SVM': round(predictions.get('svm', 0), 4)
                },
                'ensemble_average': round(ensemble, 4),
                'flood_probability_percent': round(ensemble * 100, 2)
            },
            'risk_assessment': {
                'level': risk_level,
                'color': risk_color,
                'icon': risk_icon,
                'confidence': round(confidence, 1)
            },
            'recommendations': recommendations,
            'input_summary': {
                'monsoon_intensity': data.get('MonsoonIntensity'),
                'urbanization': data.get('Urbanization'),
                'river_management': data.get('RiverManagement')
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get model performance metrics"""
    if not metadata:
        return jsonify({'error': 'Models not trained yet'}), 404
    
    return jsonify({
        'metrics': metadata.get('metrics', {}),
        'feature_importance': metadata.get('feature_importance', {}),
        'dataset_info': {
            'n_samples': metadata.get('n_samples', 0),
            'n_features': metadata.get('n_features', 0),
            'target_min': metadata.get('target_min', 0),
            'target_max': metadata.get('target_max', 1),
            'target_mean': metadata.get('target_mean', 0.5)
        }
    })


@app.route('/api/sample-predictions', methods=['GET'])
def sample_predictions():
    """Get sample predictions for comparison chart"""
    if not metadata:
        return jsonify({'error': 'Models not trained yet'}), 404
    
    return jsonify({
        'sample_predictions': metadata.get('sample_predictions', {}),
        'metrics': metadata.get('metrics', {})
    })


@app.route('/api/features', methods=['GET'])
def get_features():
    """Get feature information"""
    feature_info = {
        'MonsoonIntensity': {'description': 'Monsoon rainfall intensity level', 'unit': 'scale 0-15', 'category': 'Climate'},
        'TopographyDrainage': {'description': 'Natural topographic drainage capacity', 'unit': 'scale 0-15', 'category': 'Geography'},
        'RiverManagement': {'description': 'Quality of river management systems', 'unit': 'scale 0-15', 'category': 'Infrastructure'},
        'Deforestation': {'description': 'Level of deforestation in watershed', 'unit': 'scale 0-15', 'category': 'Environment'},
        'Urbanization': {'description': 'Degree of urban development', 'unit': 'scale 0-15', 'category': 'Socioeconomic'},
        'ClimateChange': {'description': 'Impact of climate change indicators', 'unit': 'scale 0-15', 'category': 'Climate'},
        'DamsQuality': {'description': 'Quality and maintenance of dams', 'unit': 'scale 0-15', 'category': 'Infrastructure'},
        'Siltation': {'description': 'River/reservoir siltation level', 'unit': 'scale 0-15', 'category': 'Environment'},
        'AgriculturalPractices': {'description': 'Agricultural land use practices', 'unit': 'scale 0-15', 'category': 'Socioeconomic'},
        'Encroachments': {'description': 'Level of floodplain encroachments', 'unit': 'scale 0-15', 'category': 'Socioeconomic'},
        'IneffectiveDisasterPreparedness': {'description': 'Lack of disaster preparedness', 'unit': 'scale 0-15', 'category': 'Policy'},
        'DrainageSystems': {'description': 'Urban drainage system capacity', 'unit': 'scale 0-15', 'category': 'Infrastructure'},
        'CoastalVulnerability': {'description': 'Vulnerability of coastal areas', 'unit': 'scale 0-15', 'category': 'Geography'},
        'Landslides': {'description': 'Landslide risk and occurrence', 'unit': 'scale 0-15', 'category': 'Geography'},
        'Watersheds': {'description': 'Watershed management quality', 'unit': 'scale 0-15', 'category': 'Environment'},
        'DeterioratingInfrastructure': {'description': 'Infrastructure deterioration level', 'unit': 'scale 0-15', 'category': 'Infrastructure'},
        'PopulationScore': {'description': 'Population density risk score', 'unit': 'scale 0-15', 'category': 'Socioeconomic'},
        'WetlandLoss': {'description': 'Loss of natural wetlands', 'unit': 'scale 0-15', 'category': 'Environment'},
        'InadequatePlanning': {'description': 'Poor urban/rural planning level', 'unit': 'scale 0-15', 'category': 'Policy'},
        'PoliticalFactors': {'description': 'Political factors affecting management', 'unit': 'scale 0-15', 'category': 'Policy'},
    }
    
    return jsonify({'features': feature_info, 'feature_list': ORIGINAL_FEATURES})


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Batch prediction for multiple scenarios"""
    try:
        data = request.get_json()
        scenarios = data.get('scenarios', [])
        
        if not scenarios:
            return jsonify({'error': 'No scenarios provided'}), 400
        
        results = []
        for scenario in scenarios:
            features = engineer_features(scenario)
            features_2d = features.reshape(1, -1)
            
            if scaler is not None:
                features_scaled = scaler.transform(features_2d)
            else:
                features_scaled = features_2d
            
            preds = []
            if 'rf' in models:
                preds.append(float(np.clip(models['rf'].predict(features_2d)[0], 0, 1)))
            if 'xgb' in models:
                preds.append(float(np.clip(models['xgb'].predict(features_2d)[0], 0, 1)))
            if 'svm' in models:
                preds.append(float(np.clip(models['svm'].predict(features_scaled)[0], 0, 1)))
            if 'lstm' in models:
                preds.append(float(np.clip(models['lstm'].predict(features_scaled)[0], 0, 1)))
            
            ensemble = float(np.mean(preds)) if preds else 0
            risk_level, risk_color, _ = get_risk_level(ensemble)
            
            results.append({
                'scenario': scenario.get('name', 'Unknown'),
                'ensemble': round(ensemble, 4),
                'percent': round(ensemble * 100, 2),
                'risk_level': risk_level,
                'risk_color': risk_color
            })
        
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/historical-data', methods=['GET'])
def historical_data():
    """Return sample historical dataset statistics"""
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, 'flood.csv'))
        
        stats = {}
        for col in df.columns:
            stats[col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'median': float(df[col].median())
            }
        
        # Distribution of flood probability
        bins = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
        hist, edges = np.histogram(df['FloodProbability'], bins=bins)
        
        return jsonify({
            'statistics': stats,
            'distribution': {
                'counts': hist.tolist(),
                'bins': edges.tolist(),
                'labels': ['0-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-100%']
            },
            'total_records': len(df)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("🌊 FLOOD PREDICTION API SERVER")
    print("=" * 60)
    load_models()
    print("\n🚀 Starting server on http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
