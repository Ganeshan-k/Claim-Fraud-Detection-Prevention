import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
)
import joblib
import os
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# 🔍 Define features and target
SELECTED_FEATURES = [
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
]
TARGET_VARIABLE = 'PotentialFraud'  # Target: 1 = Fraud, 0 = Not Fraud

def train_and_evaluate(data_path: str, artifacts_dir: str = './artifacts'):
    """
    Loads data, preprocesses it, trains an XGBoost model, evaluates it, and saves artifacts.

    Args:
        data_path (str): Path to the training CSV file.
        artifacts_dir (str): Directory to save trained model and preprocessing objects.
    """
    print("🚀 Starting Medicare Fraud Detection Model Training...")
    print("🔐 An official U.S. government-compliant AI model: https://data.cms.gov/provider-data/")

    # =============================================
    # 1. Load and Validate Data
    # =============================================
    print(f"📂 Loading dataset from '{data_path}'...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at '{data_path}'. Please check the file path.")

    df = pd.read_csv(data_path)
    print(f"📊 Loaded data shape: {df.shape}")

    # Validate target column exists
    if TARGET_VARIABLE not in df.columns:
        raise ValueError(f"Target column '{TARGET_VARIABLE}' not found in dataset.")

    # =============================================
    # 2. Preprocessing and Feature Engineering
    # =============================================
    print("🧹 Preprocessing data...")

    # Ensure all selected features are present
    missing_features = [col for col in SELECTED_FEATURES if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features in data: {missing_features}")

    # Clean and encode target: 'Yes'/'No' → 1/0
    if df[TARGET_VARIABLE].dtype == 'object':
        df[TARGET_VARIABLE] = df[TARGET_VARIABLE].str.strip().map({'Yes': 1, 'No': 0})
    df[TARGET_VARIABLE] = df[TARGET_VARIABLE].astype(int)

    # Extract features and target
    X = df[SELECTED_FEATURES].copy()
    y = df[TARGET_VARIABLE]

    # Handle missing values: Use median for numerical features
    print("🔍 Checking for missing values...")
    if X.isnull().any().any():
        print("⚠️  Missing values detected. Filling with median...")
        X = X.fillna(X.median(numeric_only=True))
    else:
        print("✅ No missing values found.")

    # =============================================
    # 3. Split Data (Stratified to preserve fraud ratio)
    # =============================================
    print("🔄 Splitting data into training and testing sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y  # Ensures fraud ratio is preserved
    )
    print(f"  Training samples: {X_train.shape[0]} | Testing samples: {X_test.shape[0]}")
    print(f"  Fraud cases in training: {y_train.sum()} ({(y_train.mean()*100):.2f}%)")
    print(f"  Fraud cases in test:     {y_test.sum()} ({(y_test.mean()*100):.2f}%)")

    # =============================================
    # 4. Feature Scaling
    # =============================================
    print("⚖️ Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✅ Feature scaling complete.")

    # =============================================
    # 5. Handle Class Imbalance
    # =============================================
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"⚖️ Class imbalance detected. Using scale_pos_weight = {scale_pos_weight:.2f} to improve fraud detection.")

    # =============================================
    # 6. Train XGBoost Model
    # =============================================
    print("🧠 Training XGBoost classifier...")
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
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    print("✅ Model training complete.")

    # =============================================
    # 7. Evaluate Model (Focus on Fraud Class)
    # =============================================
    print("\n📈 Evaluating model on test set...")
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Key metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)  # High recall = catching more fraud
    f1 = f1_score(y_test, y_pred)

    print(f"✅ Accuracy:  {accuracy:.4f}")
    print(f"✅ Precision: {precision:.4f} (When we flag fraud, how often are we right?)")
    print(f"✅ Recall:    {recall:.4f} (What % of actual fraud did we catch?)")
    print(f"✅ F1-Score:  {f1:.4f} (Best balance of precision and recall)")

    # Detailed report
    print("\n📋 Classification Report (Detailed):")
    print(classification_report(y_test, y_pred, target_names=['Non-Fraud (0)', 'Fraud (1)']))

    # Confusion Matrix (Optional: Uncomment to display)
    # print("\n📊 Confusion Matrix:")
    # print(confusion_matrix(y_test, y_pred))

    # =============================================
    # 8. Save Artifacts for Deployment
    # =============================================
    print(f"\n💾 Saving model artifacts to '{artifacts_dir}'...")
    os.makedirs(artifacts_dir, exist_ok=True)

    joblib.dump(model, os.path.join(artifacts_dir, 'fraud_detection_model1.joblib'))
    joblib.dump(scaler, os.path.join(artifacts_dir, 'scaler1.joblib'))
    joblib.dump(SELECTED_FEATURES, os.path.join(artifacts_dir, 'selected_features1.joblib'))

    print("✅ Artifacts saved:")
    print(f"   - Model: {os.path.join(artifacts_dir, 'fraud_detection_model1.joblib')}")
    print(f"   - Scaler: {os.path.join(artifacts_dir, 'scaler1.joblib')}")
    print(f"   - Feature List: {os.path.join(artifacts_dir, 'selected_features1.joblib')}")
    print("🎉 Training pipeline complete. Ready for real-time inference.")

    # Optional: Save feature importance
    try:
        importance_df = pd.DataFrame({
            'Feature': SELECTED_FEATURES,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        importance_df.to_csv(os.path.join(artifacts_dir, 'feature_importance.csv'), index=False)
        print("📊 Feature importance saved.")
    except Exception as e:
        print(f"⚠️  Could not save feature importance: {e}")

if __name__ == "__main__":
    # ✅ Fixed: Use raw string for Windows path
    DATA_PATH = r'C:\Users\karth\OneDrive\Desktop\initdone\Stream\data_upload\final_claimmodel.csv'

    # Run the training pipeline
    train_and_evaluate(data_path=DATA_PATH, artifacts_dir='./artifacts')