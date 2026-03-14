"""
Blood Infection (Sepsis) Prediction System
Main entry point for training and evaluation
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

# Direct imports without src package
import importlib.util

# Load modules directly
spec_preprocessing = importlib.util.spec_from_file_location(
    "data_preprocessing", 
    project_root / "src" / "data_preprocessing.py"
)
data_preprocessing = importlib.util.module_from_spec(spec_preprocessing)
spec_preprocessing.loader.exec_module(data_preprocessing)
DataPreprocessor = data_preprocessing.DataPreprocessor

spec_train = importlib.util.spec_from_file_location(
    "train_model",
    project_root / "src" / "train_model.py"
)
train_model = importlib.util.module_from_spec(spec_train)
spec_train.loader.exec_module(train_model)
ModelTrainer = train_model.ModelTrainer

spec_eval = importlib.util.spec_from_file_location(
    "evaluation",
    project_root / "src" / "evaluation.py"
)
evaluation = importlib.util.module_from_spec(spec_eval)
spec_eval.loader.exec_module(evaluation)
ModelEvaluator = evaluation.ModelEvaluator

def generate_sample_dataset():
    """Generate synthetic sepsis dataset for demonstration"""
    print("Generating sample sepsis dataset...")
    
    np.random.seed(42)
    n_samples = 500
    
    # Features
    wbc_count = np.random.normal(7.5, 4, n_samples)
    temperature = np.random.normal(37.5, 1.2, n_samples)
    heart_rate = np.random.normal(85, 20, n_samples)
    respiratory_rate = np.random.normal(16, 4, n_samples)
    lactate = np.random.normal(2, 1.5, n_samples)
    glucose = np.random.normal(120, 40, n_samples)
    platelet_count = np.random.normal(250, 80, n_samples)
    bilirubin = np.random.normal(1.0, 0.8, n_samples)
    
    # Create sepsis label based on features
    sepsis = (
        (wbc_count > 11) & (temperature > 38.5) & (lactate > 2) |
        (heart_rate > 120) & (respiratory_rate > 20) & (lactate > 3)
    ).astype(int)
    
    # Add some randomness
    sepsis[np.random.choice(n_samples, 50, replace=False)] = 1 - sepsis[np.random.choice(n_samples, 50, replace=False)]
    
    # Create DataFrame
    data = pd.DataFrame({
        'wbc_count': np.clip(wbc_count, 0, 100),
        'temperature': np.clip(temperature, 35, 42),
        'heart_rate': np.clip(heart_rate, 40, 200),
        'respiratory_rate': np.clip(respiratory_rate, 10, 50),
        'lactate': np.clip(lactate, 0, 20),
        'glucose': np.clip(glucose, 40, 400),
        'platelet_count': np.clip(platelet_count, 10, 500),
        'bilirubin': np.clip(bilirubin, 0, 20),
        'sepsis': sepsis
    })
    
    # Save dataset
    data_path = project_root / 'data' / 'sepsis_dataset.csv'
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(data_path, index=False)
    
    print(f"✓ Sample dataset generated: {data_path}")
    print(f"  Shape: {data.shape}")
    print(f"  Sepsis cases: {sepsis.sum()} ({sepsis.sum()/len(sepsis)*100:.1f}%)")
    
    return data_path

def main():
    """Main pipeline: preprocess → train → evaluate"""
    
    print("\n" + "="*60)
    print("BLOOD INFECTION (SEPSIS) PREDICTION SYSTEM")
    print("="*60 + "\n")
    
    # Step 1: Prepare data
    print("STEP 1: DATA PREPARATION")
    print("-" * 60)
    
    dataset_path = project_root / 'data' / 'sepsis_dataset.csv'
    if not dataset_path.exists():
        dataset_path = generate_sample_dataset()
    else:
        print(f"Using existing dataset: {dataset_path}")
    
    # Step 2: Preprocess data
    print("\n\nSTEP 2: DATA PREPROCESSING")
    print("-" * 60)
    
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data(str(dataset_path))
    
    X, y, features = preprocessor.preprocess(
        df, 
        target_column='sepsis',
        fit_scaler=True,
        scaler_path=str(project_root / 'models' / 'scaler.pkl')
    )
    
    # Step 3: Split data
    print("\n\nSTEP 3: DATA SPLITTING")
    print("-" * 60)
    
    trainer = ModelTrainer(random_state=42)
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(X, y)
    
    # Step 4: Train models
    print("\n\nSTEP 4: MODEL TRAINING")
    print("-" * 60)
    
    rf_model = trainer.train_random_forest(n_estimators=100, max_depth=10)
    lr_model = trainer.train_logistic_regression(C=1.0, max_iter=1000)
    
    # Step 5: Evaluate models
    print("\n\nSTEP 5: MODEL EVALUATION")
    print("-" * 60)
    
    evaluator = ModelEvaluator()
    
    rf_results = evaluator.evaluate(rf_model, X_test, y_test, 'Random Forest')
    lr_results = evaluator.evaluate(lr_model, X_test, y_test, 'Logistic Regression')
    
    # Step 6: Compare models
    print("\n\nSTEP 6: MODEL COMPARISON")
    print("-" * 60)
    
    comparison_df = evaluator.compare_models()
    
    # Step 7: Save best model
    print("\n\nSTEP 7: MODEL SAVING")
    print("-" * 60)
    
    # Select best model based on F1 score
    if rf_results['f1'] >= lr_results['f1']:
        print("\n✓ Random Forest selected as best model")
        trainer.save_model('random_forest', rf_model, 
                          str(project_root / 'models' / 'infection_model.pkl'))
    else:
        print("\n✓ Logistic Regression selected as best model")
        trainer.save_model('logistic_regression', lr_model,
                          str(project_root / 'models' / 'infection_model.pkl'))
    
    # Step 8: Feature importance
    print("\n\nSTEP 8: FEATURE IMPORTANCE")
    print("-" * 60)
    
    importances = trainer.get_feature_importance('random_forest')
    if importances is not None:
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print("\nTop Features:")
        print(importance_df.to_string(index=False))
    
    print("\n\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print(f"  ✓ models/scaler.pkl")
    print(f"  ✓ models/infection_model.pkl")
    print("\nTo run the Streamlit app, use:")
    print("  streamlit run app/app.py")
    print("\n")

if __name__ == "__main__":
    main()