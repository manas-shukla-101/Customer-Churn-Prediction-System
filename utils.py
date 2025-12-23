"""
Utility Functions Module
Common utilities for visualization, data handling, and reporting.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import io
import base64
from datetime import datetime


class Visualizer:
    """Create various visualizations for churn analysis."""
    
    @staticmethod
    def plot_feature_importance(feature_importance, top_n=15):
        """Create feature importance bar plot."""
        features = feature_importance['feature'][:top_n]
        scores = feature_importance['importance'][:top_n]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(features, scores, color='steelblue')
        ax.set_xlabel('Importance Score')
        ax.set_title('Top Features Affecting Customer Churn')
        ax.invert_yaxis()
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_confusion_matrix(conf_matrix):
        """Create confusion matrix heatmap."""
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'])
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        ax.set_title('Confusion Matrix')
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_roc_curve(roc_data, roc_auc):
        """Create ROC curve plot."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(roc_data['fpr'], roc_data['tpr'], color='darkorange', 
                lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc="lower right")
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_churn_distribution(y_data, title="Churn Distribution"):
        """Create churn distribution pie chart."""
        fig, ax = plt.subplots(figsize=(8, 6))
        churn_counts = pd.Series(y_data).value_counts()
        colors = ['#2ecc71', '#e74c3c']
        ax.pie(churn_counts.values, labels=['No Churn', 'Churn'], autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax.set_title(title)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_customer_segments(segments):
        """Create customer segment visualization."""
        fig, ax = plt.subplots(figsize=(8, 6))
        categories = list(segments.keys())
        values = list(segments.values())
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        
        ax.bar(categories, values, color=colors)
        ax.set_ylabel('Number of Customers')
        ax.set_title('Customer Risk Segments')
        
        # Add value labels on bars
        for i, v in enumerate(values):
            ax.text(i, v + max(values)*0.01, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_metrics_comparison(metrics):
        """Create model metrics comparison chart."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
        metric_values = [
            metrics.get('accuracy', 0),
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            metrics.get('f1', 0),
            metrics.get('roc_auc', 0)
        ]
        
        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
        bars = ax.bar(metric_names, metric_values, color=colors)
        
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Metrics')
        ax.set_ylim([0, 1])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig


class DataAnalyzer:
    """Analyze data patterns and distributions."""
    
    @staticmethod
    def get_data_summary(df):
        """Get summary statistics of the dataset."""
        summary = {
            'total_records': len(df),
            'total_features': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'numeric_features': len(df.select_dtypes(include=['number']).columns),
            'categorical_features': len(df.select_dtypes(include=['object']).columns)
        }
        return summary
    
    @staticmethod
    def get_numeric_stats(df, numeric_columns):
        """Get statistics for numeric columns."""
        stats = df[numeric_columns].describe().to_dict()
        return stats
    
    @staticmethod
    def get_categorical_distribution(df, categorical_columns):
        """Get value counts for categorical columns."""
        distribution = {}
        for col in categorical_columns:
            if col in df.columns:
                distribution[col] = df[col].value_counts().head(10).to_dict()
        return distribution


class ReportGenerator:
    """Generate reports and summaries."""
    
    @staticmethod
    def generate_model_report(metrics, feature_importance, insights):
        """Generate a comprehensive model report."""
        report = f"""
        MODEL PERFORMANCE REPORT
        {'='*50}
        
        ACCURACY METRICS:
        - Accuracy: {metrics.get('accuracy', 0):.4f}
        - Precision: {metrics.get('precision', 0):.4f}
        - Recall: {metrics.get('recall', 0):.4f}
        - F1-Score: {metrics.get('f1', 0):.4f}
        - ROC-AUC: {metrics.get('roc_auc', 0):.4f}
        
        CUSTOMER INSIGHTS:
        - Total Customers: {insights['total_customers']}
        - High-Risk Customers: {insights['high_risk_count']} ({insights['high_risk_percentage']}%)
        - Average Churn Probability: {insights['average_churn_prob']:.4f}
        
        TOP FACTORS AFFECTING CHURN:
        """
        
        for i, (feature, score) in enumerate(zip(
            feature_importance['feature'][:5],
            feature_importance['importance'][:5]
        ), 1):
            report += f"\n        {i}. {feature}: {score:.4f}"
        
        report += f"\n\n        RECOMMENDATION:\n        {insights['recommendation']}"
        return report
    
    @staticmethod
    def generate_customer_report(customer_id, prediction, features):
        """Generate individual customer churn report."""
        report = f"""
        CUSTOMER CHURN ASSESSMENT REPORT
        {'='*50}
        Customer ID: {customer_id}
        Assessment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        CHURN PREDICTION:
        - Churn Probability: {prediction['churn_probability']:.2%}
        - Retention Probability: {prediction['retain_probability']:.2%}
        - Status: {'HIGH RISK' if prediction['churn_probability'] > 0.7 else 'AT RISK' if prediction['churn_probability'] > 0.3 else 'SAFE'}
        
        RECOMMENDED ACTIONS:
        """
        
        if prediction['churn_probability'] > 0.7:
            report += "\n        1. URGENT: Contact customer for retention discussion"
            report += "\n        2. Offer special incentives/discounts"
            report += "\n        3. Provide dedicated account manager"
        elif prediction['churn_probability'] > 0.3:
            report += "\n        1. Monitor customer activity closely"
            report += "\n        2. Proactive customer engagement"
            report += "\n        3. Offer value-added services"
        else:
            report += "\n        1. Maintain standard service quality"
            report += "\n        2. Regular check-ins with customer"
        
        return report
