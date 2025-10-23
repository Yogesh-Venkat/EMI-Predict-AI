# EMIPredict_AI_README.md

# EMIPredict AI - Intelligent Financial Risk Assessment Platform

## Project Overview
**EMIPredict AI** is a comprehensive financial risk assessment platform that predicts **EMI eligibility** and **maximum monthly EMI** using machine learning. The system integrates MLflow for experiment tracking and a multi-page **Streamlit web application** for interactive real-time predictions.

This project leverages **classification and regression models** with advanced feature engineering and provides actionable insights for financial institutions, FinTech platforms, and individual users.

---

## Skills and Technologies
- **Programming & Analytics:** Python, Pandas, NumPy, Scikit-learn, XGBoost  
- **Web App & Deployment:** Streamlit, Streamlit Cloud, Multi-page app structure  
- **ML & Experiment Tracking:** Logistic Regression, Random Forest, XGBoost, MLflow  
- **Data Handling:** Feature engineering, ratio calculations, categorical encoding, train-test-validation splits  
- **Visualization:** Matplotlib, Seaborn, interactive dashboards  

---

## Problem Statement
Many people struggle to pay EMIs due to poor financial planning and inadequate risk assessment. This project addresses this problem by providing **data-driven predictions and insights** on EMI eligibility and affordability using a dataset of 400,000 realistic financial profiles across 5 EMI scenarios.

**Goals:**
- Dual ML problems: Classification (EMI eligibility) and Regression (Maximum EMI amount)  
- Real-time risk assessment for loan applications  
- Comprehensive feature engineering and ratio analysis  
- MLflow integration for experiment tracking and model versioning  
- Streamlit Cloud deployment for easy access  

---

## Dataset
**Dataset:** `EMI_dataset.csv`  

- **Records:** 400,000 financial profiles  
- **Input Features (22 variables):**
  - Personal: `age`, `gender`, `marital_status`, `education`
  - Employment & Income: `monthly_salary`, `employment_type`, `years_of_employment`, `company_type`
  - Housing & Family: `house_type`, `monthly_rent`, `family_size`, `dependents`
  - Expenses: `school_fees`, `college_fees`, `travel_expenses`, `groceries_utilities`, `other_monthly_expenses`
  - Financial Status: `existing_loans`, `current_emi_amount`, `credit_score`, `bank_balance`, `emergency_fund`
  - Loan Application: `emi_scenario`, `requested_amount`, `requested_tenure`
- **Target Variables:**
  - Classification: `emi_eligibility` (Eligible, High_Risk, Not_Eligible)  
  - Regression: `max_monthly_emi` (safe EMI amount)  
- **EMI Scenarios:** E-commerce Shopping, Home Appliances, Vehicle, Personal Loan, Education  

---

## Architecture & Data Flow

```
Dataset (400K Records)
        ↓
Data Cleaning & Validation
        ↓
Feature Engineering (Ratios, Encodings)
        ↓
Train-Test-Validation Split
        ↓
ML Model Training (Classification & Regression)
        ↓
MLflow Logging & Model Registry
        ↓
Streamlit Multi-page Application
        ↓
Cloud Deployment (Streamlit Cloud)
```

**Components:**
1. **Data Layer:** Structured dataset with 22 features  
2. **Processing Layer:** Preprocessing, feature engineering, EDA  
3. **Model Layer:** Classification & regression models, MLflow tracking  
4. **Application Layer:** Multi-page Streamlit app with real-time predictions  
5. **Deployment Layer:** Streamlit Cloud hosting, GitHub integration  

---

## Machine Learning Models
**Classification Models (EMI Eligibility):**
- Logistic Regression  
- Random Forest Classifier  
- XGBoost Classifier  

**Regression Models (Maximum EMI):**
- Linear Regression  
- Random Forest Regressor  
- XGBoost Regressor  

**Metrics Logged (via MLflow):**
- Classification: Accuracy, Precision, Recall, F1-Score  
- Regression: RMSE, MAE, R2, MAPE  

---

## Streamlit Web Application
**Pages:**
1. **Home 🏠** – Overview, key features, EMI distribution  
2. **Prediction 💰** – Real-time EMI eligibility and max EMI prediction  
3. **Analysis 📊** – Visualizations, correlations, dataset exploration  
4. **Model Performance 📈** – Compare model metrics and logs from MLflow  
5. **Recommendations 🛠** – Actionable insights for financial risk reduction  
6. **Admin ⚙️** – Data management, model retraining  

**Inputs:** Age, gender, salary, dependents, housing type, monthly expenses, loan request details, etc.  
**Outputs:** EMI eligibility badge (🟢 Eligible / 🟠 High Risk / 🔴 Not Eligible), predicted max EMI  

---

## Installation & Setup

1. Clone repository:
```bash
git clone https://github.com/username/EMIPredict_AI.git
cd EMIPredict_AI
```

2. Create virtual environment & install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate      # Linux / Mac
.venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

3. Run Streamlit app locally:
```bash
streamlit run streamlit_app/app.py
```

4. MLflow logging is enabled; runs saved in `/mlruns`.  

---

## Usage
- Fill applicant financial details on the **Prediction** page.  
- View EMI eligibility and maximum EMI instantly.  
- Explore **Analysis** for data insights and **Model Performance** for ML metrics.  
- Follow recommendations for high-risk applicants.  

---

## Expected Results
- Classification F1-score > 0.95  
- Regression RMSE < 2000 INR  
- MLflow experiment tracking and model registry integrated  
- Streamlit Cloud deployment with multi-page interactive app  

---

## Business Impact
- Automated financial risk assessment, reducing manual processing time by ~80%  
- Standardized loan eligibility assessment across multiple EMI scenarios  
- Scalable platform for high-volume loan applications  
- Actionable recommendations to improve financial health  

---

## Technical Tags
Python, MLflow, Streamlit Cloud, Feature Engineering, Classification, Regression, XGBoost, Random Forest, Financial Analytics, Risk Assessment, Big Data Processing  

---

## References
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)  
- [Streamlit Documentation](https://docs.streamlit.io/)  
- [Scikit-learn](https://scikit-learn.org/stable/)  
- [XGBoost](https://xgboost.readthedocs.io/en/stable/)

