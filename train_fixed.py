"""
Simple training script with explicit output
"""

import sys
import os
from pathlib import Path

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib

project_root = Path(__file__).parent

def main():
    print("\n" + "="*60, flush=True)
    print("BLOOD INFECTION (SEPSIS) PREDICTION SYSTEM", flush=True)
    print("="*60 + "\n", flush=True)
    
    # Step 1: Generate sample dataset
    print("STEP 1: GENERATING SAMPLE DATASET", flush=True)
    print("-" * 60, flush=True)
    
    np.random.seed(42)
    n_samples = 500
    
    data = pd.DataFrame({
        'wbc_count': np.clip(np.random.normal(7.5, 4, n_samples), 0, 100),
        'temperature': np.clip(np.random.normal(37.5, 1.2, n_samples), 35, 42),
        'heart_rate': np.clip(np.random.normal(85, 20, n_samples), 40, 200),
        'respiratory_rate': np.clip(np.random.normal(16, 4, n_samples), 10, 50),
        'lactate': np.clip(np.random.normal(2, 1.5, n_samples), 0, 20),
        'glucose': np.clip(np.random.normal(120, 40, n_samples), 40, 400),
        'platelet_count': np.clip(np.random.normal(250, 80, n_samples), 10, 500),
        'bilirubin': np.clip(np.random.normal(1.0, 0.8, n_samples), 0, 20)
    })
    
    sepsis = (
        (data['wbc_count'] > 11) & (data['temperature'] > 38.5) & (data['lactate'] > 2) |
        (data['heart_rate'] > 120) & (data['respiratory_rate'] > 20) & (data['lactate'] > 3)
    ).astype(int)
    
    sepsis[np.random.choice(n_samples, 50, replace=False)] = 1 - sepsis[np.random.choice(n_samples, 50, replace=False)]
    data['sepsis'] = sepsis
    
    data_path = project_root / 'data' / 'sepsis_dataset.csv'
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(data_path, index=False)
    
    print(f"✓ Dataset generated: {data_path}", flush=True)
    print(f"  Shape: {data.shape}", flush=True)
    print(f"  Sepsis cases: {sepsis.sum()}\n", flush=True)
    
    # Step 2: Preprocess
    print("STEP 2: PREPROCESSING DATA", flush=True)
    print("-" * 60, flush=True)
    
    X = data.drop('sepsis', axis=1)
    y = data['sepsis']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"✓ Features scaled: {X_scaled.shape}", flush=True)
    print(f"✓ Scaler expects: {scaler.n_features_in_} features\n", flush=True)
    
    # Step 3: Split
    print("STEP 3: SPLITTING DATA", flush=True)
    print("-" * 60, flush=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✓ Training set: {X_train.shape[0]} samples", flush=True)
    print(f"✓ Test set: {X_test.shape[0]} samples\n", flush=True)
    
    # Step 4: Train models
    print("STEP 4: TRAINING MODELS", flush=True)
    print("-" * 60, flush=True)
    
    print("\nTraining Random Forest...", flush=True)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_f1 = f1_score(y_test, rf_pred, zero_division=0)
    print(f"  Accuracy: {accuracy_score(y_test, rf_pred):.4f}", flush=True)
    print(f"  F1 Score: {rf_f1:.4f}", flush=True)
    
    print("\nTraining Logistic Regression...", flush=True)
    lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_f1 = f1_score(y_test, lr_pred, zero_division=0)
    print(f"  Accuracy: {accuracy_score(y_test, lr_pred):.4f}", flush=True)
    print(f"  F1 Score: {lr_f1:.4f}\n", flush=True)
    
    # Step 5: Save models
    print("STEP 5: SAVING PKL FILES", flush=True)
    print("-" * 60, flush=True)
    
    models_dir = project_root / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    if rf_f1 >= lr_f1:
        best_model = rf
        model_name = "Random Forest"
    else:
        best_model = lr
        model_name = "Logistic Regression"
    
    # Save scaler
    scaler_path = models_dir / 'scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"✓ Scaler saved: {scaler_path}", flush=True)
    print(f"  Size: {scaler_path.stat().st_size} bytes", flush=True)
    
    # Save model
    model_path = models_dir / 'infection_model.pkl'
    joblib.dump(best_model, model_path)
    print(f"✓ Model saved: {model_path}", flush=True)
    print(f"  Type: {model_name}", flush=True)
    print(f"  Size: {model_path.stat().st_size} bytes\n", flush=True)
    
    # Step 6: Verify
    print("STEP 6: VERIFYING FILES", flush=True)
    print("-" * 60, flush=True)
    
    if scaler_path.exists():
        test_scaler = joblib.load(scaler_path)
        print(f"✓ Scaler verified: {test_scaler.n_features_in_} features", flush=True)
    
    if model_path.exists():
        test_model = joblib.load(model_path)
        print(f"✓ Model verified: {type(test_model).__name__}", flush=True)
    
    print("\n" + "="*60, flush=True)
    print("✓ TRAINING COMPLETE!", flush=True)
    print("="*60, flush=True)
    print("\nNext step - Run Streamlit:", flush=True)
    print("  streamlit run app/app.py\n", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()