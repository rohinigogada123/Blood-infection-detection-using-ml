from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import numpy as np
import pandas as pd

class ModelEvaluator:
    """Evaluate model performance with comprehensive metrics"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate(self, model, X_test, y_test, model_name='model'):
        """Comprehensive model evaluation"""
        print(f"\n{'='*50}")
        print(f"EVALUATING {model_name.upper()}")
        print(f"{'='*50}\n")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            has_proba = True
        else:
            y_pred_proba = None
            has_proba = False
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print("CLASSIFICATION METRICS:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        
        # ROC-AUC Score
        if has_proba and len(np.unique(y_test)) == 2:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            print(f"  ROC-AUC:   {roc_auc:.4f}")
        else:
            roc_auc = None
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nCONFUSION MATRIX:")
        print(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        # Sensitivity and Specificity
        sensitivity = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
        specificity = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
        
        print(f"\nCLINICAL METRICS:")
        print(f"  Sensitivity: {sensitivity:.4f} (True Positive Rate)")
        print(f"  Specificity: {specificity:.4f} (True Negative Rate)")
        
        # Classification Report
        print(f"\nDETAILED CLASSIFICATION REPORT:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'confusion_matrix': cm
        }
    
    def get_results_dataframe(self):
        """Get results as DataFrame"""
        results_data = []
        
        for model_name, metrics in self.results.items():
            row = {
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1'],
                'ROC-AUC': metrics['roc_auc'],
                'Sensitivity': metrics['sensitivity'],
                'Specificity': metrics['specificity']
            }
            results_data.append(row)
        
        return pd.DataFrame(results_data)
    
    def compare_models(self):
        """Compare all evaluated models"""
        if not self.results:
            print("No models evaluated yet")
            return None
        
        print(f"\n{'='*80}")
        print("MODEL COMPARISON")
        print(f"{'='*80}\n")
        
        df = self.get_results_dataframe()
        print(df.to_string(index=False))
        
        return df