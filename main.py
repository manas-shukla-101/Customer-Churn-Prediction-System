"""
Customer Churn Prediction System
This module contains the implementation of a customer churn prediction system using machine learning techniques.
It includes data preprocessing, model training, evaluation, and prediction functionalities.
It uses popular libraries such as pandas, scikit-learn, streamlit, and XGBoost.
Streamlit is used for building an interactive web application for users to input customer data and receive churn predictions.
XGBoost is employed as the machine learning model for its efficiency and performance in classification tasks.
In this, you understand why customers leave a service not just how to predict future churn.
So, you can implement strategies to retain customers and improve business outcomes.
"""


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

from data_processor import DataProcessor
from model_trainer import ModelTrainer
from feature_analyzer import FeatureAnalyzer
from utils import Visualizer, DataAnalyzer, ReportGenerator

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.sidebar.markdown("---")
st.logo("icon.png")
st.markdown(
    """
    ---
    <style>
        [alt=Logo] {
            height: 6rem; /* Adjust this value */
            margin-top: 3rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Custom CSS
st.markdown("""
    <style>
    .main-header { font-size: 2.5em; font-weight: bold; color: #2c3e50; }
    .metric-card { background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0; }
    .success-text { color: #27ae60; font-weight: bold; }
    .warning-text { color: #e74c3c; font-weight: bold; }
    .info-text { color: #3498db; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'model_trainer' not in st.session_state:
    st.session_state.model_trainer = ModelTrainer()
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False


def main():
    """Main application function."""
    st.markdown('<div class="main-header">🎯 Customer Churn Prediction System</div>', unsafe_allow_html=True)
    st.markdown("*Predict customer churn, understand why, and implement retention strategies*")
    st.divider()
    
    # Sidebar navigation
    with st.sidebar:
        st.title("📋 Navigation")
        page = st.radio(
            "Select a page:",
            ["📤 Upload Data", "🔍 Data Analysis", "🤖 Model Training", 
             "📊 Model Evaluation", "💡 Predictions", "📈 Analytics Dashboard"]
        )
    
    # Route to different pages
    if page == "📤 Upload Data":
        page_upload_data()
    elif page == "🔍 Data Analysis":
        page_data_analysis()
    elif page == "🤖 Model Training":
        page_model_training()
    elif page == "📊 Model Evaluation":
        page_model_evaluation()
    elif page == "💡 Predictions":
        page_predictions()
    elif page == "📈 Analytics Dashboard":
        page_analytics_dashboard()


def page_upload_data():
    """Page for uploading and exploring data."""
    st.header("📤 Data Upload & Exploration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload your CSV dataset", type=['csv'])
    
    with col2:
        sample_download = st.checkbox("Download sample dataset")
    
    if sample_download:
        # Generate sample dataset
        sample_data = {
            'customer_id': range(1, 101),
            'tenure': np.random.randint(1, 73, 100),
            'monthly_charges': np.random.uniform(20, 120, 100),
            'total_charges': np.random.uniform(100, 8000, 100),
            'contract_length': np.random.choice(['Month-to-month', 'One year', 'Two year'], 100),
            'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], 100),
            'tech_support': np.random.choice(['Yes', 'No'], 100),
            'churn': np.random.choice(['Yes', 'No'], 100)
        }
        sample_df = pd.DataFrame(sample_data)
        st.download_button(
            label="Download Sample CSV",
            data=sample_df.to_csv(index=False),
            file_name="sample_churn_data.csv",
            mime="text/csv"
        )
    
    if uploaded_file is not None:
        # Load data
        st.session_state.df_original = st.session_state.data_processor.load_data(uploaded_file)
        
        if st.session_state.df_original is not None:
            st.success("✓ Data loaded successfully!")
            
            # Detect target column
            target = st.session_state.data_processor.detect_target_column(st.session_state.df_original)
            
            if target is None:
                st.warning("⚠️ Could not auto-detect target column. Please select it:")
                all_cols = st.session_state.df_original.columns.tolist()
                target = st.selectbox("Select target column (churn column):", all_cols)
                st.session_state.data_processor.target_column = target
            else:
                st.info(f"✓ Target column detected: **{target}**")
            
            # Identify column types
            st.session_state.data_processor.identify_column_types(st.session_state.df_original)
            
            # Display data info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", len(st.session_state.df_original))
            with col2:
                st.metric("Total Features", len(st.session_state.df_original.columns))
            with col3:
                st.metric("Numeric Features", len(st.session_state.data_processor.numeric_columns))
            with col4:
                st.metric("Categorical Features", len(st.session_state.data_processor.categorical_columns))
            
            # Display first few rows
            st.subheader("Dataset Preview")
            st.dataframe(st.session_state.df_original.head(10), use_container_width=True)
            
            # Display statistics
            st.subheader("Dataset Statistics")
            stats = st.session_state.data_processor.get_statistics(st.session_state.df_original)
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Missing Values:**", f"{stats['missing_values']} total missing values")
            with col2:
                st.write("**Target Distribution:**")
                if stats['target_distribution']:
                    for label, count in stats['target_distribution'].items():
                        st.write(f"  - {label}: {count} ({count/len(st.session_state.df_original)*100:.1f}%)")


def page_data_analysis():
    """Page for data analysis and exploration."""
    st.header("🔍 Data Analysis & Insights")
    
    if st.session_state.df_original is None:
        st.warning("⚠️ Please upload data first on the 'Upload Data' page")
        return
    
    df = st.session_state.df_original
    
    # Data Summary
    st.subheader("📈 Data Summary")
    summary = DataAnalyzer.get_data_summary(df)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", summary['total_records'])
    with col2:
        st.metric("Total Features", summary['total_features'])
    with col3:
        st.metric("Numeric Features", summary['numeric_features'])
    with col4:
        st.metric("Categorical Features", summary['categorical_features'])
    
    st.divider()
    
    # Numeric columns analysis
    if st.session_state.data_processor.numeric_columns:
        st.subheader("📊 Numeric Features Analysis")
        numeric_stats = DataAnalyzer.get_numeric_stats(df, st.session_state.data_processor.numeric_columns)
        st.dataframe(pd.DataFrame(numeric_stats).round(2), use_container_width=True)
    
    # Categorical columns analysis
    if st.session_state.data_processor.categorical_columns:
        st.subheader("🏷️ Categorical Features Analysis")
        cat_dist = DataAnalyzer.get_categorical_distribution(df, st.session_state.data_processor.categorical_columns)
        
        col_select = st.selectbox("Select categorical feature to analyze:", 
                                  list(cat_dist.keys()) if cat_dist else [])
        
        if col_select and col_select in cat_dist:
            dist_data = cat_dist[col_select]
            st.bar_chart(pd.Series(dist_data))
    
    # Churn distribution
    if st.session_state.data_processor.target_column:
        st.subheader("🎯 Target Variable Distribution")
        target_col = st.session_state.data_processor.target_column
        
        fig = Visualizer.plot_churn_distribution(df[target_col], f"Distribution of {target_col}")
        st.pyplot(fig)
        plt.close()


def page_model_training():
    """Page for model training."""
    st.header("🤖 Model Training")
    
    if st.session_state.df_original is None:
        st.warning("⚠️ Please upload data first")
        return
    
    st.info("ℹ️ This page preprocesses data and trains the XGBoost model")
    
    # Preprocessing options
    st.subheader("🔧 Preprocessing Options")
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2)
    
    with col2:
        random_state = st.number_input("Random State", 0, 1000, 42)
    
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Processing data..."):
            # Preprocess data
            df_processed = st.session_state.data_processor.preprocess(st.session_state.df_original)
            st.session_state.df_processed = df_processed
            
            # Prepare for training
            X, y = st.session_state.data_processor.prepare_for_training(df_processed)
            
            # Split data
            X_train, X_test, y_train, y_test = st.session_state.data_processor.split_data(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Store test data in session
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            
            st.success("✓ Data preprocessing completed")
        
        with st.spinner("Training XGBoost model..."):
            # Train model
            st.session_state.model_trainer.train(X_train, y_train)
            st.session_state.model_trained = True
            st.success("✓ Model trained successfully!")
        
        # Display training info
        st.subheader("📊 Training Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Samples", len(X_train))
        with col2:
            st.metric("Testing Samples", len(X_test))
        with col3:
            st.metric("Features Used", X_train.shape[1])


def page_model_evaluation():
    """Page for model evaluation."""
    st.header("📊 Model Evaluation")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first on the 'Model Training' page")
        return
    
    with st.spinner("Evaluating model..."):
        # Evaluate model
        metrics = st.session_state.model_trainer.evaluate(
            st.session_state.X_test, 
            st.session_state.y_test
        )
    
    # Display metrics
    st.subheader("📈 Performance Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    with col2:
        st.metric("Precision", f"{metrics['precision']:.4f}")
    with col3:
        st.metric("Recall", f"{metrics['recall']:.4f}")
    with col4:
        st.metric("F1-Score", f"{metrics['f1']:.4f}")
    with col5:
        st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
    
    st.divider()
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        fig = Visualizer.plot_confusion_matrix(metrics['confusion_matrix'])
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("ROC Curve")
        fig = Visualizer.plot_roc_curve(metrics['roc_curve'], metrics['roc_auc'])
        st.pyplot(fig)
        plt.close()
    
    # Feature Importance
    st.divider()
    st.subheader("🎯 Feature Importance Analysis")
    
    feature_importance = st.session_state.model_trainer.get_feature_importance()
    
    fig = Visualizer.plot_feature_importance(feature_importance)
    st.pyplot(fig)
    plt.close()
    
    # Feature analysis
    analyzer = FeatureAnalyzer(feature_importance, 
                              st.session_state.data_processor.get_feature_names(), 
                              st.session_state.X_test)
    
    impact_analysis = analyzer.analyze_feature_impact()
    
    st.subheader("📊 Impact Analysis")
    impact_df = pd.DataFrame(impact_analysis)
    st.dataframe(impact_df, use_container_width=True)


def page_predictions():
    """Page for making predictions."""
    st.header("💡 Customer Churn Predictions")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first")
        return
    
    # Prediction type selection
    pred_type = st.radio("Select prediction type:", ["Single Customer", "Batch Predictions"])
    
    if pred_type == "Single Customer":
        st.subheader("🔍 Single Customer Prediction")
        st.info("Enter customer data to predict churn probability")
        
        # Create input fields dynamically
        customer_data = {}
        
        # Get numeric features
        numeric_features = st.session_state.data_processor.numeric_columns
        categorical_features = st.session_state.data_processor.categorical_columns
        
        col1, col2 = st.columns(2)
        
        # Numeric inputs
        for i, feature in enumerate(numeric_features):
            if i % 2 == 0:
                customer_data[feature] = col1.number_input(f"{feature}", value=0.0)
            else:
                customer_data[feature] = col2.number_input(f"{feature}", value=0.0)
        
        # Categorical inputs (not used in single prediction yet)
        
        if st.button("🔮 Predict Churn", type="primary"):
            # Prepare data
            input_df = pd.DataFrame([customer_data])
            
            # Get prediction
            prediction = st.session_state.model_trainer.predict_single(input_df)
            
            # Display results
            st.divider()
            st.subheader("📊 Prediction Result")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                status = "⚠️ HIGH RISK" if prediction['churn_probability'] > 0.7 else \
                        "⚡ AT RISK" if prediction['churn_probability'] > 0.3 else "✓ SAFE"
                st.metric("Status", status)
            
            with col2:
                st.metric("Churn Probability", f"{prediction['churn_probability']:.2%}")
            
            with col3:
                st.metric("Retain Probability", f"{prediction['retain_probability']:.2%}")
            
            # Recommendations
            st.subheader("💡 Recommended Actions")
            report = ReportGenerator.generate_customer_report("N/A", prediction, customer_data)
            st.write(report)
    
    else:  # Batch Predictions
        st.subheader("📊 Batch Predictions")
        st.info("Upload a CSV file with customer data to get predictions for all customers")
        
        uploaded_file = st.file_uploader("Upload customer data", type=['csv'])
        
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            
            if st.button("🔮 Predict All Customers", type="primary"):
                with st.spinner("Making predictions..."):
                    # Preprocess the batch data using the same processor and scaler from training
                    batch_df_processed = st.session_state.data_processor.prepare_for_prediction(batch_df)
                    
                    predictions = st.session_state.model_trainer.predict(batch_df_processed)
                    
                    # Add predictions to dataframe
                    batch_df['churn_probability'] = predictions['churn_probability']
                    batch_df['prediction'] = ['Churn' if p > 0.5 else 'No Churn' 
                                             for p in predictions['churn_probability']]
                    
                    st.success("✓ Predictions completed!")
                    
                    # Display results
                    st.subheader("📋 Prediction Results")
                    st.dataframe(batch_df, use_container_width=True)
                    
                    # Download results
                    csv = batch_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name="churn_predictions.csv",
                        mime="text/csv"
                    )


def page_analytics_dashboard():
    """Page for analytics dashboard."""
    st.header("📈 Analytics Dashboard")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first")
        return
    
    st.subheader("📊 Model Performance Overview")
    
    # Get metrics
    metrics = st.session_state.model_trainer.get_metrics()
    
    # Performance metrics chart
    fig = Visualizer.plot_metrics_comparison(metrics)
    st.pyplot(fig)
    plt.close()
    
    st.divider()
    
    # Customer Segments
    st.subheader("👥 Customer Risk Segmentation")
    
    predictions = st.session_state.model_trainer.predict(st.session_state.X_test)
    segments = {
        'safe': np.sum(predictions['churn_probability'] < 0.3),
        'at_risk': np.sum((predictions['churn_probability'] >= 0.3) & (predictions['churn_probability'] < 0.7)),
        'critical': np.sum(predictions['churn_probability'] >= 0.7)
    }
    
    fig = Visualizer.plot_customer_segments(segments)
    st.pyplot(fig)
    plt.close()
    
    # Insights
    st.divider()
    st.subheader("💡 Business Insights")
    
    feature_importance = st.session_state.model_trainer.get_feature_importance()
    analyzer = FeatureAnalyzer(feature_importance,
                              st.session_state.data_processor.get_feature_names(),
                              st.session_state.X_test)
    
    insights = analyzer.get_actionable_insights(
        st.session_state.model_trainer.model,
        st.session_state.X_test,
        st.session_state.data_processor.get_feature_names()
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Customers", insights['total_customers'])
    with col2:
        st.metric("High Risk Count", insights['high_risk_count'])
    with col3:
        st.metric("High Risk %", f"{insights['high_risk_percentage']:.1f}%")
    
    # Retention Strategies
    st.subheader("🎯 Recommended Retention Strategies")
    
    top_features = [(f, s) for f, s in zip(feature_importance['feature'][:10], 
                                           feature_importance['importance'][:10])]
    strategies = analyzer.generate_retention_strategies(top_features)
    
    for i, strategy in enumerate(strategies, 1):
        with st.expander(f"{i}. {strategy['strategy']}"):
            st.write(f"**Description:** {strategy['description']}")
            st.write(f"**Expected Impact:** {strategy['expected_impact']}")


if __name__ == "__main__":
    main()













