"""
Feature Analysis Module
Analyzes feature importance, generates insights, and provides retention recommendations.
"""

import pandas as pd
import numpy as np


class FeatureAnalyzer:
    """Analyzes features and provides business insights."""
    
    def __init__(self, feature_importance, feature_names, X_train):
        self.feature_importance = feature_importance
        self.feature_names = feature_names
        self.X_train = X_train
        
    def get_top_features(self, top_n=10):
        """Get top N most important features."""
        top_features = self.feature_importance['feature'][:top_n]
        top_scores = self.feature_importance['importance'][:top_n]
        
        return list(zip(top_features, top_scores))
    
    def analyze_feature_impact(self):
        """Analyze impact of top features on churn."""
        analysis = []
        
        for i, (feature, score) in enumerate(zip(
            self.feature_importance['feature'][:10],
            self.feature_importance['importance'][:10]
        )):
            impact = {
                'rank': i + 1,
                'feature': feature,
                'importance_score': round(score, 4),
                'impact_level': self._categorize_impact(score)
            }
            analysis.append(impact)
        
        return analysis
    
    def _categorize_impact(self, score):
        """Categorize impact level based on score."""
        if score > 0.1:
            return "Critical"
        elif score > 0.05:
            return "High"
        elif score > 0.02:
            return "Medium"
        else:
            return "Low"
    
    def generate_retention_strategies(self, feature_importance_list):
        """Generate business-friendly retention strategies."""
        strategies = []
        
        top_features = [f[0].lower() for f in feature_importance_list[:5]]
        
        # Map features to business strategies
        strategy_map = {
            'contract_length': {
                'strategy': 'Contract Extension Program',
                'description': 'Offer incentives for longer contract terms to increase customer commitment',
                'expected_impact': 'High'
            },
            'monthly_charges': {
                'strategy': 'Price Optimization',
                'description': 'Review pricing structure and offer discounts for loyal customers',
                'expected_impact': 'High'
            },
            'total_charges': {
                'strategy': 'Value Enhancement',
                'description': 'Bundle services to increase perceived value',
                'expected_impact': 'Medium'
            },
            'tenure': {
                'strategy': 'Early Engagement Program',
                'description': 'Focus on customer engagement in first 6 months',
                'expected_impact': 'Critical'
            },
            'internet_service': {
                'strategy': 'Service Quality Improvement',
                'description': 'Improve service reliability and performance',
                'expected_impact': 'High'
            },
            'tech_support': {
                'strategy': 'Support Enhancement',
                'description': 'Ensure 24/7 technical support availability',
                'expected_impact': 'Medium'
            }
        }
        
        for feature in top_features:
            for key, strategy in strategy_map.items():
                if key in feature or feature in key:
                    strategies.append(strategy)
                    break
        
        return strategies[:5]
    
    def get_customer_segments(self, predictions, data):
        """Segment customers by churn risk."""
        data = data.copy()
        data['churn_probability'] = predictions['churn_probability']
        
        segments = {
            'safe': len(data[data['churn_probability'] < 0.3]),
            'at_risk': len(data[(data['churn_probability'] >= 0.3) & (data['churn_probability'] < 0.7)]),
            'critical': len(data[data['churn_probability'] >= 0.7])
        }
        
        return segments
    
    def get_actionable_insights(self, model, X_test, feature_names):
        """Generate actionable insights from model predictions."""
        predictions = model.predict_proba(X_test)[:, 1]
        
        # Identify high-risk customers
        high_risk_idx = np.where(predictions >= 0.7)[0]
        
        insights = {
            'total_customers': len(X_test),
            'high_risk_count': len(high_risk_idx),
            'high_risk_percentage': round((len(high_risk_idx) / len(X_test)) * 100, 2),
            'average_churn_prob': round(predictions.mean(), 4),
            'recommendation': f'Focus retention efforts on {len(high_risk_idx)} high-risk customers'
        }
        
        return insights
