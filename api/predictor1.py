# predictor2.py (Claim-Level Fraud Detection with XAI)
import pandas as pd
import joblib
import os
from datetime import datetime
import numpy as np
import shap  # Import the SHAP library for Explainable AI

# =============================
# Global Artifacts (Load Once)
# =============================

_artifacts = {}

def load_model():
    """
    Lazily load model, scaler, features, and create a SHAP explainer.
    Will be populated once model training is complete.
    """
    global _artifacts
    if _artifacts:
        return _artifacts

    # Define expected artifacts directory
    artifacts_dir = './artifacts/'# Separate folder for claim-level artifacts
    if not os.path.exists(artifacts_dir):
        raise FileNotFoundError(f"Artifacts directory not found: '{artifacts_dir}'. Run model training for claims first.")

    model_file = os.path.join(artifacts_dir, 'fraud_detection_model1.joblib')
    scaler_file = os.path.join(artifacts_dir, 'scaler1.joblib')
    features_file = os.path.join(artifacts_dir, 'selected_features1.joblib')

    # Load artifacts
    try:
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
        features = joblib.load(features_file)

        # --- EXPLAINABLE AI SETUP ---
        explainer = shap.TreeExplainer(model)  # SHAP explainer for interpretation

        _artifacts = {
            'model': model,
            'scaler': scaler,
            'features': features,
            'explainer': explainer
        }
        print("✅ Claim fraud model, scaler, feature list, and SHAP explainer loaded successfully.")
        return _artifacts
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load claim-level artifacts or create explainer: {e}")


def predict_claim_fraud(data: pd.DataFrame):
    """
    Predict fraud risk for claim-level data and generate an XAI explanation.

    Expected columns:
        Total_Claims_Per_Bene,
        TimeInHptal,
        Provider_Claim_Frequency,
        ChronicCond_stroke_Yes,
        DeductibleAmtPaid,
        NoOfMonths_PartBCov,
        NoOfMonths_PartACov,
        OPD_Flag_Yes,
        Diagnosis_Count,
        ChronicDisease_Count,
        Age

    Returns:
        results (list): Prediction for each row
        summary (dict): Aggregated statistics
    """
    artifacts = load_model()
    model = artifacts['model']
    scaler = artifacts['scaler']
    selected_features = artifacts['features']
    explainer = artifacts['explainer']

    # Validate input
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("Input must be a non-empty pandas DataFrame")

    # ✅ Check for required columns
    required_columns = {
        'Total_Claims_Per_Bene',
        'TimeInHptal',
        'Provider_Claim_Frequency',
        'ChronicCond_stroke_Yes',
        'DeductibleAmtPaid',
        'NoOfMonths_PartBCov',
        'NoOfMonths_PartACov',
        'OPD_Flag_Yes',
        'Diagnosis_Count',
        'ChronicDisease_Count',
        'Age'
    }

    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"❌ Missing required claim-level features: {sorted(missing)}")

    # Ensure only selected features are used
    if not set(selected_features).issubset(set(data.columns)):
        missing_in_selected = set(selected_features) - set(data.columns)
        raise ValueError(f"❌ Model expects features not in data: {sorted(missing_in_selected)}")

    X = data[selected_features].copy()

    # Handle missing numeric values
    numeric_cols = X.select_dtypes(include=np.number).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

    # Scale features
    X_scaled = scaler.transform(X)

    # Get predictions
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]  # Probability of fraud

    results = []
    for i, row in data.iterrows():
        prob = probabilities[i]
        pred = predictions[i]

        # --- EXPLAINABLE AI (XAI) FOR THIS ROW ---
        shap_values = explainer.shap_values(X_scaled[i:i+1])  # SHAP for single instance
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification, take positive class

        # Create impact DataFrame
        feature_impact = pd.DataFrame({
            'feature': selected_features,
            'impact': shap_values[0]
        }).sort_values(by='impact', key=abs, ascending=False)  # Sort by absolute impact

        # Risk categorization
        if prob > 0.8:
            risk = "High"
            top_drivers = feature_impact.head(3)
            explanation = "Top fraud drivers: " + ", ".join([
                f"{r.feature} ({r.impact:+.2f})" for _, r in top_drivers.iterrows()
            ])
        elif prob > 0.5:
            risk = "Medium"
            top_drivers = feature_impact.head(3)
            explanation = "Key indicators: " + ", ".join([
                f"{r.feature} ({r.impact:+.2f})" for _, r in top_drivers.iterrows()
            ])
        else:
            risk = "Low"
            protective = feature_impact.tail(3)  # Most negative impacts (protective)
            explanation = "Low-risk due to: " + ", ".join([
                f"{r.feature} ({r.impact:+.2f})" for _, r in protective.iterrows()
            ])

        # Generate potential savings (example logic)
        base_amount = row.get('DeductibleAmtPaid', 0) + (row.get('TimeInHptal', 0) * 150)
        potential_savings = base_amount * 0.9 if pred == 1 else 0.0

        results.append({
            "ClaimID": f"C{10000 + i}",  # Placeholder ID
            "PatientID": row.get("BeneID", f"B{20000 + i}"),
            "DiagnosisCode": "I63.9" if row.get("ChronicCond_stroke_Yes", 0) else "Z00.0",
            "prediction": "Fraud" if pred == 1 else "Legitimate",
            "fraud_probability": round(float(prob), 4),
            "risk_level": risk,
            "explanation": explanation,
            "potential_savings": round(potential_savings, 2),
            "action": {
                "High": "Flag for pre-pay review",
                "Medium": "Schedule audit",
                "Low": "Auto-approve"
            }[risk],
            "Amount": round(float(base_amount), 2)
        })

    # Summary
    summary = {
        "total_claims_processed": len(data),
        "total_fraud_flags": int((predictions == 1).sum()),
        "total_potential_savings": sum(r["potential_savings"] for r in results)
    }

    return results, summary