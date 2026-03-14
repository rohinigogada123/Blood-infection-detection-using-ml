"""
Simple training script without feature engineering
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib

project_root = Path(__file__).parent

print("\n" + "="*60)
print("BLOOD INFECTION (SEPSIS) PREDICTION SYSTEM")
print("="*60 + "\n")

# Step 1: Generate sample dataset (EXACTLY 8 FEATURES)
print("STEP 1: GENERATING SAMPLE DATASET")
print("-" * 60)

np.random.seed(42)
n_samples = 500

# Create EXACTLY 8 features - NO feature engineering
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

# Create target variable
sepsis = (
    (data['wbc_count'] > 11) & (data['temperature'] > 38.5) & (data['lactate'] > 2) |
    (data['heart_rate'] > 120) & (data['respiratory_rate'] > 20) & (data['lactate'] > 3)
).astype(int)

# Add some randomness
sepsis[np.random.choice(n_samples, 50, replace=False)] = 1 - sepsis[np.random.choice(n_samples, 50, replace=False)]
data['sepsis'] = sepsis

data_path = project_root / 'data' / 'sepsis_dataset.csv'
data_path.parent.mkdir(parents=True, exist_ok=True)
data.to_csv(data_path, index=False)

print(f"✓ Dataset generated: {data_path}")
print(f"  Shape: {data.shape}")
print(f"  Features: {list(data.columns)}")
print(f"  Number of features (excluding target): 8")
print(f"  Sepsis cases: {sepsis.sum()} ({sepsis.sum()/len(sepsis)*100:.1f}%)\n")

# Step 2: Preprocess (NO FEATURE ENGINEERING)
print("STEP 2: PREPROCESSING DATA")
print("-" * 60)

X = data.drop('sepsis', axis=1)
y = data['sepsis']

print(f"Features: {list(X.columns)}")
print(f"Number of features: {X.shape[1]}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"✓ Data preprocessed")
print(f"  Features scaled: {X_scaled.shape}")
print(f"  Scaler expects: {scaler.n_features_in_} features\n")

# Step 3: Split
print("STEP 3: SPLITTING DATA")
print("-" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples\n")

# Step 4: Train models
print("STEP 4: TRAINING MODELS")
print("-" * 60)

print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_f1 = f1_score(y_test, rf_pred, zero_division=0)
print(f"  Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
print(f"  F1 Score: {rf_f1:.4f}")

print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_f1 = f1_score(y_test, lr_pred, zero_division=0)
print(f"  Accuracy: {accuracy_score(y_test, lr_pred):.4f}")
print(f"  F1 Score: {lr_f1:.4f}\n")

# Step 5: Save models
print("STEP 5: SAVING PKL FILES")
print("-" * 60)

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
print(f"✓ Scaler saved: {scaler_path}")
print(f"  Scaler features: {scaler.n_features_in_}")

# Save model
model_path = models_dir / 'infection_model.pkl'
joblib.dump(best_model, model_path)
print(f"✓ Model saved: {model_path}")
print(f"  Model type: {model_name}\n")

# Step 6: Verify
print("STEP 6: VERIFYING FILES")
print("-" * 60)

if scaler_path.exists():
    size = scaler_path.stat().st_size
    print(f"✓ Scaler PKL: {size:,} bytes")
    # Verify scaler
    test_scaler = joblib.load(scaler_path)
    print(f"  Scaler expects: {test_scaler.n_features_in_} features")

if model_path.exists():
    size = model_path.stat().st_size
    print(f"✓ Model PKL: {size:,} bytes")

print("\n" + "="*60)
print("✓ TRAINING COMPLETE!")
print("="*60)
print("\nReady to use Streamlit app:")
print("  streamlit run app/app.py\n")