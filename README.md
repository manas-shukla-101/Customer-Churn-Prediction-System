# Customer Churn Prediction System

An intelligent machine learning application that predicts customer churn, analyzes key drivers, and recommends retention strategies. Built with XGBoost and Streamlit for an interactive user experience.

## 📋 Overview

Customer churn prediction is critical for business retention. This system helps identify at-risk customers before they leave, understand the factors driving churn, and implement targeted retention strategies. With 8.1% high-risk customer identification rate, businesses can proactively reduce churn and improve profitability.

## ✨ Key Features

- **Automated Data Processing**: Intelligent preprocessing with automatic target detection, missing value handling, and feature scaling
- **XGBoost Model Training**: High-performance machine learning model with optimized hyperparameters
- **Comprehensive Analytics**: Feature importance analysis, customer segmentation, and actionable insights
- **Interactive Dashboard**: Streamlit-based web interface for data exploration and predictions
- **Retention Strategies**: AI-generated business recommendations based on churn drivers
- **Multi-page Application**:
  - Data Upload & Exploration
  - Data Analysis & Statistics
  - Model Training & Tuning
  - Model Evaluation & Metrics
  - Single & Batch Predictions
  - Analytics Dashboard with Business Insights

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/manas-shukla-101/Customer-Churn-Prediction-System.git
cd churn-prediction

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Run the Streamlit application
streamlit run main.py
```

Open your browser and navigate to `http://localhost:8501`

## 📁 Project Structure

```
├── main.py                  # Streamlit web application
├── data_processor.py        # Data preprocessing & feature engineering
├── model_trainer.py         # XGBoost model training & evaluation
├── feature_analyzer.py      # Feature importance & business insights
├── utils.py                 # Visualization & reporting utilities
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```

## 🎯 How It Works

1. **Data Upload**: Load your customer dataset (CSV format)
2. **Data Preprocessing**: Automatic encoding, scaling, and missing value handling
3. **Model Training**: Train XGBoost classifier with stratified splitting
4. **Analysis**: View feature importance and customer risk segments
5. **Predictions**: Predict churn for new customers with probability scores
6. **Recommendations**: Get actionable retention strategies based on top churn drivers

## 📊 Model Performance

- **Accuracy**: Varies by dataset
- **Metrics**: Precision, Recall, F1-Score, ROC-AUC
- **Visualization**: Confusion Matrix, ROC Curve, Feature Importance Charts

## 🛠️ Technologies Used

- **Machine Learning**: XGBoost, scikit-learn
- **Data Processing**: pandas, NumPy
- **Web Framework**: Streamlit
- **Visualization**: Matplotlib, Seaborn
- **Model Persistence**: joblib

## 📝 Input Data Format

CSV file with customer features:
- Numeric columns (charges, minutes, calls, etc.)
- Categorical columns (service types, contract types, etc.)
- Target column: 'Churn', 'Target', 'Churned', or similar

## 💡 Example Workflow

1. Upload customer dataset
2. System auto-detects target column
3. Review data statistics and distributions
4. Train model on 80% of data
5. Evaluate performance on 20% test set
6. Make predictions on new customers
7. Review retention recommendations

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## 📄 License

MIT License - feel free to use this project for personal and commercial purposes.

## 👨‍💼 Author

Created for customer retention analytics and churn mitigation strategies.

---

**Ready to predict and prevent customer churn? Start using the system today!**

---
---


_Created with 💗 by Manas Shukla_
