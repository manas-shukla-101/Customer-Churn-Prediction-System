"""
Model Training and Evaluation Module
Handles XGBoost model training, evaluation, and prediction.
"""

import xgboost as xgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report, roc_curve, auc)
import numpy as np
import joblib
import os


class ModelTrainer:
    """Trains and evaluates XGBoost model for churn prediction."""
    
    def __init__(self, random_state=42):
        self.model = None
        self.random_state = random_state
        self.metrics = {}
        self.feature_importance = None
        self.feature_names = None
        
    def train(self, X_train, y_train, params=None):
        """Train XGBoost model with optimal parameters."""
        if params is None:
            params = {
                'objective': 'binary:logistic',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.5,
                'reg_lambda': 1.0,
                'random_state': self.random_state,
                'eval_metric': 'logloss',
                'verbose': 0
            }
        
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train)],
            verbose=False
        )
        
        self.feature_names = X_train.columns.tolist()
        print("✓ Model trained successfully")
        return self.model
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance on test data."""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # ROC curve data
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        self.metrics['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
        
        print("✓ Model evaluation completed")
        return self.metrics
    
    def get_feature_importance(self):
        """Get feature importance scores."""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        importance_scores = self.model.feature_importances_
        feature_importance_df = {
            'feature': self.feature_names,
            'importance': importance_scores.tolist()
        }
        
        # Sort by importance
        sorted_idx = np.argsort(importance_scores)[::-1]
        self.feature_importance = {
            'feature': [self.feature_names[i] for i in sorted_idx],
            'importance': [importance_scores[i] for i in sorted_idx]
        }
        
        return self.feature_importance
    
    def predict(self, X):
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'churn_probability': probabilities[:, 1]
        }
    
    def predict_single(self, X_single):
        """Predict for a single customer."""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        pred = self.model.predict(X_single)[0]
        proba = self.model.predict_proba(X_single)[0]
        
        return {
            'will_churn': bool(pred),
            'churn_probability': float(proba[1]),
            'retain_probability': float(proba[0])
        }
    
    def save_model(self, filepath):
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save!")
        
        joblib.dump(self.model, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load pre-trained model from disk."""
        if os.path.exists(filepath):
            self.model = joblib.load(filepath)
            print(f"✓ Model loaded from {filepath}")
            return self.model
        else:
            print(f"✗ Model file not found: {filepath}")
            return None
    
    def get_metrics(self):
        """Return model evaluation metrics."""
        return self.metrics
