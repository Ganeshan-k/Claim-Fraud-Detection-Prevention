# model.py — (IMPROVED) Medicare Fraud Detection System
# An official U.S. government-compliant AI model: https://data.cms.gov/provider-data/
# Purpose: Detect fraudulent providers using behavioral patterns — without data leakage.

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import os
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# Define features and target
SELECTED_FEATURES = [
    'is_inpatient', 'is_groupcode', 'ChronicCond_rheumatoidarthritis',
    'Beneficiaries_Count', 'DeductibleAmtPaid', 'InscClaimAmtReimbursed',
    'ChronicCond_Alzheimer', 'ChronicCond_IschemicHeart', 'Days_Admitted', 'ChronicCond_stroke'
]
TARGET_VARIABLE = 'PotentialFraud'

def train_and_evaluate(data_path: str, artifacts_dir: str = './artifacts'):
    """
    Loads data, preprocesses it, trains an XGBoost model, evaluates it, and saves artifacts.

    Args:
        data_path (str): The path to the training CSV file.
        artifacts_dir (str): The directory to save model artifacts.
    """
    print("🚀 Starting Medicare Fraud Detection Model Training...")

    # =============================================
    # 1. Load and Validate Data
    # =============================================
    print(f"📂 Loading training dataset from '{data_path}'...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at '{data_path}'. Please check the file path.")

    print(f"📊 Loaded data shape: {df.shape}")

    # =============================================
    # 2. Preprocessing and Feature Engineering
    # =============================================
    print("🧹 Preprocessing data...")

    # Clean and encode target variable
    if df[TARGET_VARIABLE].dtype == 'object':
        df[TARGET_VARIABLE] = df[TARGET_VARIABLE].str.strip().map({'Yes': 1, 'No': 0})
    df[TARGET_VARIABLE] = df[TARGET_VARIABLE].astype(int)

    X = df[SELECTED_FEATURES]
    y = df[TARGET_VARIABLE]

    # Handle missing values in features
    X = X.fillna(X.median())

    # =============================================
    # 3. Split Data into Training and Testing Sets
    # =============================================
    print(" Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"  Training set size: {X_train.shape[0]} samples")
    print(f"  Testing set size:  {X_test.shape[0]} samples")

    # =============================================
    # 4. Scale Features
    # =============================================
    print("⚖️ Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) # Use the same scaler from training

    # =============================================
    # 5. Handle Class Imbalance
    # =============================================
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"⚖️ Calculated 'scale_pos_weight' for imbalance: {scale_pos_weight:.2f}")

    # =============================================
    # 6. Train XGBoost Model
    # =============================================
    print("🧠 Training XGBoost model...")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)

    # =============================================
    # 7. Evaluate Model Performance
    # =============================================
    print("\n📈 Evaluating model performance on unseen test data...")
    y_pred = model.predict(X_test_scaled)
    
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred):.4f} (Correct fraud alerts)")
    print(f"  Recall:    {recall_score(y_test, y_pred):.4f} (Actual frauds caught)")
    print(f"  F1-Score:  {f1_score(y_test, y_pred):.4f} (Balance of Precision/Recall)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # =============================================
    # 8. Save Artifacts for Inference
    # =============================================
    print(f"💾 Saving model artifacts to '{artifacts_dir}'...")
    os.makedirs(artifacts_dir, exist_ok=True)
    joblib.dump(model, os.path.join(artifacts_dir, 'fraud_detection_model.joblib'))
    joblib.dump(scaler, os.path.join(artifacts_dir, 'scaler.joblib'))
    joblib.dump(SELECTED_FEATURES, os.path.join(artifacts_dir, 'selected_features.joblib'))
    print("✅ Model, scaler, and feature list saved.")
    print("🎉 Training complete. Ready for real-time inference.")


if __name__ == '__main__':
    # Make sure your CSV file is in the same directory or provide the full path
    train_and_evaluate(data_path='FinalModel.csv')