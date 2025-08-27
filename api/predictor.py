# predictor.py (IMPROVED with XAI)
import pandas as pd
import joblib
import os
from datetime import datetime
import numpy as np
import shap # Import the SHAP library

# =============================
# Global Artifacts (Load Once)
# =============================

_artifacts = {}

def load_model():
    """
    Lazily load model, scaler, features, and create a SHAP explainer.
    """
    global _artifacts
    if _artifacts:
        return _artifacts

    # Find artifact files
    artifacts_dir = './artifacts/'
    if not os.path.exists(artifacts_dir):
        raise FileNotFoundError(f"Artifacts directory not found: '{artifacts_dir}'. Run model.py first.")

    model_file = os.path.join(artifacts_dir, 'fraud_detection_model.joblib')
    scaler_file = os.path.join(artifacts_dir, 'scaler.joblib')
    features_file = os.path.join(artifacts_dir, 'selected_features.joblib')

    # Load artifacts
    try:
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
        features = joblib.load(features_file)
        
        # --- EXPLAINABLE AI SETUP ---
        # Create a SHAP explainer for the model. This is used to understand predictions.
        explainer = shap.TreeExplainer(model)
        
        _artifacts = {
            'model': model,
            'scaler': scaler,
            'features': features,
            'explainer': explainer # Store the explainer
        }
        print("✅ Model, scaler, feature list, and SHAP explainer loaded successfully.")
        return _artifacts
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load artifacts or create explainer: {e}")


def predict_fraud(provider_data: pd.DataFrame):
    """
    Predict fraud risk and generate an explanation for the prediction.
    """
    artifacts = load_model()
    model = artifacts['model']
    scaler = artifacts['scaler']
    selected_features = artifacts['features']
    explainer = artifacts['explainer'] # Get the explainer

    # Validation and Preprocessing
    if not isinstance(provider_data, pd.DataFrame) or provider_data.empty:
        raise ValueError("Input must be a non-empty pandas DataFrame")
    
    missing = [f for f in selected_features if f not in provider_data.columns]
    if missing:
        raise ValueError(f"❌ Missing required features: {missing}")

    X = provider_data[selected_features].copy()
    numeric_cols = X.select_dtypes(include=np.number).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    X_scaled = scaler.transform(X)

    # Prediction
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]

    # --- EXPLAINABLE AI (XAI) CALCULATION ---
    shap_values = explainer.shap_values(X_scaled)
    
    # Create a DataFrame for easy analysis of SHAP values
    feature_impact = pd.DataFrame(
        list(zip(selected_features, shap_values[0])),
        columns=['feature', 'impact']
    ).sort_values(by='impact', ascending=False) # Sort to find top drivers
    
    # Get top 3 features pushing the score HIGHER (towards fraud)
    top_drivers = feature_impact.head(3)
    explanation = "Top fraud drivers: " + ", ".join([
        f"{row.feature} (impact: {row.impact:.2f})" for index, row in top_drivers.iterrows()
    ])

    # Risk categorization
    if prob > 0.8:
        risk = "High"
    elif prob > 0.5:
        risk = "Medium"
    else:
        risk = "Low"
        # For low-risk claims, it's more helpful to show what keeps the risk low
        bottom_drivers = feature_impact.tail(3)
        explanation = "Top non-fraud factors: " + ", ".join([
            f"{row.feature} (impact: {row.impact:.2f})" for index, row in bottom_drivers.iterrows()
        ])

    return {
        "prediction": "Fraud" if pred == 1 else "Not Fraud",
        "fraud_probability": round(float(prob), 4),
        "risk_level": risk,
        "recommended_action": {
            "High": "Flag for immediate investigation",
            "Medium": "Schedule audit within 30 days",
            "Low": "Approve claim automatically"
        }.get(risk),
        "explanation": explanation, # Add the explanation to the output
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }