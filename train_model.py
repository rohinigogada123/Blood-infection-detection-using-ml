import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """Train and evaluate ML models for sepsis prediction"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def split_data(self, X, y, test_size=0.2, val_size=0.1):
        """Split data into train, validation, and test sets"""
        print(f"\nSplitting data...")
        print(f"  Test size: {test_size}")
        print(f"  Validation size: {val_size}")
        
        # First split: Train+Val vs Test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Second split: Train vs Val
        val_split = val_size / (1 - test_size)
        self.X_train, X_val, self.y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_split, 
            random_state=self.random_state, stratify=y_temp
        )
        
        print(f"✓ Data split completed:")
        print(f"  Training set: {self.X_train.shape[0]} samples")
        print(f"  Validation set: {X_val.shape[0]} samples")
        print(f"  Test set: {self.X_test.shape[0]} samples")
        
        return self.X_train, X_val, self.X_test, self.y_train, y_val, self.y_test
    
    def train_random_forest(self, n_estimators=100, max_depth=10):
        """Train Random Forest classifier"""
        print(f"\n{'='*50}")
        print("TRAINING RANDOM FOREST CLASSIFIER")
        print(f"{'='*50}")
        print(f"Parameters: n_estimators={n_estimators}, max_depth={max_depth}")
        
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1
        )
        
        rf_model.fit(self.X_train, self.y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(rf_model, self.X_train, self.y_train, cv=5, scoring='f1_weighted')
        print(f"\nCross-validation F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        self.models['random_forest'] = rf_model
        print("✓ Random Forest training completed")
        
        return rf_model
    
    def train_logistic_regression(self, C=1.0, max_iter=1000):
        """Train Logistic Regression classifier"""
        print(f"\n{'='*50}")
        print("TRAINING LOGISTIC REGRESSION CLASSIFIER")
        print(f"{'='*50}")
        print(f"Parameters: C={C}, max_iter={max_iter}")
        
        lr_model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=self.random_state,
            n_jobs=-1,
            solver='lbfgs'
        )
        
        lr_model.fit(self.X_train, self.y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(lr_model, self.X_train, self.y_train, cv=5, scoring='f1_weighted')
        print(f"\nCross-validation F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        self.models['logistic_regression'] = lr_model
        print("✓ Logistic Regression training completed")
        
        return lr_model
    
    def save_model(self, model_name, model, filepath='models/infection_model.pkl'):
        """Save trained model"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath='models/infection_model.pkl'):
        """Load trained model"""
        model = joblib.load(filepath)
        print(f"✓ Model loaded from {filepath}")
        return model
    
    def get_feature_importance(self, model_name='random_forest'):
        """Get feature importance from tree-based models"""
        if model_name not in self.models:
            print(f"Model '{model_name}' not found")
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            return importances
        else:
            print(f"Model '{model_name}' does not have feature importance")
            return None