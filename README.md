# Customer Churn Prediction System – End-to-End ML Case Study

## 📌 Business Problem
Customer churn directly impacts revenue and long-term growth. Acquiring new customers is significantly more expensive than retaining existing ones, making early churn detection critical for business success.

This project focuses on identifying **customers with a high probability of churn**, understanding the **key factors driving churn**, and providing **actionable insights** to support retention strategies for business and marketing teams.

---

## 📊 Dataset Overview
- **Source**: Public Kaggle customer churn dataset  
- **Records**: Several thousand customer entries  
- **Features**: ~10–12 customer attributes  
- **Target Variable**: `Churn` (binary classification)

### Data Challenges
- Missing values across multiple columns  
- Noisy and inconsistent customer behavior data  
- Imbalanced churn vs non-churn classes  

These challenges required careful preprocessing and model evaluation beyond accuracy alone.

---

## 🧠 Approach & Key Decisions

### 1️⃣ Data Preprocessing
- Missing value handling and data cleaning
- Automatic encoding of categorical variables
- Feature scaling for numerical stability
- Stratified train-test split to preserve churn distribution

### 2️⃣ Model Selection
- **XGBoost Classifier** was selected due to:
  - Strong performance on tabular data
  - Ability to handle non-linear relationships
  - Built-in feature importance for explainability

### 3️⃣ Evaluation Strategy
Instead of relying only on accuracy, the model was evaluated using:
- **Precision** – to reduce false churn alerts
- **ROC-AUC** – to measure overall discrimination ability

This aligns with real-world churn scenarios where incorrect predictions can lead to unnecessary retention costs.

---

## 📈 Model Performance
- **ROC-AUC**: ~0.80–0.90 (dataset-dependent)
- **Precision**: Optimized to ensure reliable churn identification
- Visual diagnostics:
  - Confusion Matrix
  - ROC Curve
  - Feature Importance plots

---

## 🔍 Business Insights
Key churn drivers identified include:
- Contract type and tenure
- Monthly charges
- Usage patterns and service subscriptions

These insights enable teams to:
- Identify high-risk customer segments
- Prioritize retention efforts
- Design targeted marketing campaigns

---

## 🏢 How Businesses Can Use This System
1. Upload historical customer data
2. Train churn model on internal datasets
3. Identify customers with high churn probability
4. Focus retention strategies (offers, support, engagement)
5. Reduce revenue loss through proactive intervention

**End Users**:
- Management teams
- Marketing & CRM teams
- Customer success teams

---

## 🖥️ Application Features
- Interactive Streamlit dashboard
- Automated data processing
- Model training & evaluation
- Single and batch churn predictions
- Business-focused analytics & insights

---

## 🛠️ Tech Stack
- **Python**
- **Machine Learning**: XGBoost, scikit-learn
- **Data Processing**: pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Web App**: Streamlit
- **Model Persistence**: joblib

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/manas-shukla-101/Customer-Churn-Prediction-System.git
cd Customer-Churn-Prediction-System

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run main.py
```
---

## 📌 Future Improvements
- Compare XGBoost with Logistic Regression & Random Forest
- Add SHAP for advanced explainability
- Deploy model using FastAPI
- Integrate SQL-based data ingestion

---

## 👨‍💻 Author
**Manas Shukla**
_Data Analytics | Machine Learning | Business Insights_
