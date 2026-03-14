import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

class DataPreprocessor:
    """Handle data cleaning, feature engineering, and scaling"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = 'sepsis'
        
    def load_data(self, filepath):
        """Load CSV data"""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Data shape: {df.shape}")
        return df
    
    def handle_missing_values(self, df, strategy='mean'):
        """Handle missing values"""
        print("Handling missing values...")
        
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"\nMissing values:\n{missing[missing > 0]}")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].mean(), inplace=True)
            
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].mode()[0], inplace=True)
        
        print("✓ Missing values handled")
        return df
    
    def remove_duplicates(self, df):
        """Remove duplicate rows"""
        print(f"Removing duplicates... (Initial: {len(df)} rows)")
        df = df.drop_duplicates()
        print(f"✓ Duplicates removed (Final: {len(df)} rows)")
        return df
    
    def feature_engineering(self, df):
        """Create new features from existing ones"""
        print("Feature engineering...")
        
        # IMPORTANT: Only create features if the required columns exist
        if 'wbc_count' in df.columns and 'temperature' in df.columns:
            df['wbc_temp_interaction'] = df['wbc_count'] * df['temperature']
        
        if 'lactate' in df.columns and 'glucose' in df.columns:
            df['lactate_glucose_ratio'] = df['lactate'] / (df['glucose'] + 1)
        
        print("✓ Feature engineering completed")
        return df
    
    def remove_outliers(self, df, columns=None, z_threshold=3):
        """Remove outliers using Z-score"""
        print("Removing outliers...")
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        initial_shape = df.shape[0]
        
        for col in columns:
            if col != self.target_column:  # Don't remove based on target
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < z_threshold]
        
        print(f"✓ Outliers removed ({initial_shape} → {df.shape[0]} rows)")
        return df
    
    def encode_categorical(self, df):
        """Encode categorical variables"""
        print("Encoding categorical variables...")
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if self.target_column in categorical_cols:
            categorical_cols.remove(self.target_column)
        
        for col in categorical_cols:
            df[col] = pd.Categorical(df[col]).codes
            print(f"  Encoded: {col}")
        
        print("✓ Categorical encoding completed")
        return df
    
    def scale_features(self, X, fit=True, scaler_path='models/scaler.pkl'):
        """Scale features using StandardScaler"""
        print("Scaling features...")
        
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.scaler, scaler_path)
            print(f"✓ Scaler fitted and saved to {scaler_path}")
        else:
            X_scaled = self.scaler.transform(X)
            print("✓ Features scaled")
        
        return X_scaled
    
    def load_scaler(self, scaler_path='models/scaler.pkl'):
        """Load pre-fitted scaler"""
        print(f"Loading scaler from {scaler_path}...")
        self.scaler = joblib.load(scaler_path)
        print("✓ Scaler loaded")
        return self.scaler
    
    def preprocess(self, df, target_column='sepsis', fit_scaler=True, scaler_path='models/scaler.pkl'):
        """Complete preprocessing pipeline"""
        print("\n" + "="*50)
        print("STARTING DATA PREPROCESSING PIPELINE")
        print("="*50 + "\n")
        
        self.target_column = target_column
        
        # Step 1: Remove duplicates
        df = self.remove_duplicates(df)
        
        # Step 2: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 3: Remove outliers (before feature engineering)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)
        df = self.remove_outliers(df, columns=numeric_cols)
        
        # Step 4: Feature engineering
        df = self.feature_engineering(df)
        
        # Step 5: Encode categorical variables
        df = self.encode_categorical(df)
        
        # Step 6: Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        self.feature_columns = X.columns.tolist()
        
        print(f"\nFeatures before scaling: {self.feature_columns}")
        print(f"Number of features: {len(self.feature_columns)}")
        
        # Step 7: Scale features
        X_scaled = self.scale_features(X, fit=fit_scaler, scaler_path=scaler_path)
        
        print("\n" + "="*50)
        print("PREPROCESSING COMPLETE")
        print("="*50)
        print(f"Final shape - X: {X_scaled.shape}, y: {y.shape}")
        print(f"Features: {self.feature_columns}\n")
        
        return X_scaled, y, self.feature_columns