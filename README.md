# Task 2: End-to-End Machine Learning Pipeline for Customer Churn Prediction

**DevelopersHub Corporation – AI/ML Engineering Advanced Internship**  
**Submitted by:** Subhan  
**Submission Date:** March 2026  

---

## Objective
To build a **reusable, production-ready machine learning pipeline** for predicting customer churn using the Telco Customer Churn dataset. The pipeline integrates data preprocessing, model training, hyperparameter tuning, and model export for seamless deployment.

## Dataset
- **Source**: Telco Customer Churn Dataset (IBM)
- **Task**: Binary Classification (Churn vs No Churn)
- **Features**: 19 customer attributes including tenure, charges, services, contract type, etc.
- **Target**: `Churn` (Yes/No)

## Methodology & Approach

1. **Data Loading & Cleaning**
   - Loaded dataset directly from public URL
   - Handled missing values in `TotalCharges`
   - Dropped irrelevant column (`customerID`)

2. **Feature Preprocessing**
   - Used `ColumnTransformer` for automated preprocessing
   - Numeric features: Standard Scaling (`tenure`, `MonthlyCharges`, `TotalCharges`)
   - Categorical features: One-Hot Encoding

3. **Machine Learning Pipeline**
   - Built end-to-end `Pipeline` using scikit-learn
   - Compared two models:
     - Logistic Regression
     - Random Forest Classifier

4. **Hyperparameter Tuning**
   - Performed 5-fold cross-validation using `GridSearchCV`

5. **Model Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix visualization

6. **Model Export**
   - Saved the complete pipeline using `joblib` for production use

## Technologies Used
- **Python**
- **scikit-learn** (Pipeline, ColumnTransformer, GridSearchCV)
- **pandas** & **numpy**
- **joblib** (Model serialization)
- **seaborn** & **matplotlib** (Visualization)

## Results

| Metric                  | Value     |
|-------------------------|-----------|
| Best Cross-Validation Accuracy | ~0.802    |
| Test Accuracy           | ~0.795    |
| Best Model              | Random Forest Classifier |

**Confusion Matrix:**

*(Image will appear here when you run the notebook and save the plot)*

## Key Insights
- Random Forest generally outperformed Logistic Regression on this dataset.
- Contract type, tenure, and monthly charges were among the most influential features.
- The complete pipeline ensures consistent preprocessing during both training and inference.

## How to Use the Pipeline in Production

```python
import joblib
import pandas as pd

# Load the saved pipeline
pipeline = joblib.load('churn_prediction_pipeline.joblib')

# Predict on new customer data
new_data = pd.DataFrame({...})   # your new customer features
prediction = pipeline.predict(new_data)
probability = pipeline.predict_proba(new_data)

print("Churn Prediction:", "Yes" if prediction[0] == 1 else "No")
