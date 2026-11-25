# Credit Risk Prediction Model 

## Project Overview
This project develops a machine learning classification model (XGBoost) to predict the probability of default on customer loan applications. The model is designed to serve as a *decision support system* for banking credit analysts to minimize the risk of financial loss (nonperforming loans).

## Tech Stack & Methodology
* **Language:** Python
* **Libraries:** Pandas, Scikit-Learn, XGBoost, Matplotlib, Seaborn
* **Techniques Used:** 
  * Data Wrangling (Handling Missing Values via Imputation, Outlier Removal, Duplicate Handling)
  * Feature Engineering (One-Hot Encoding, Ordinal Encoding)
  * Addressing Class Imbalance (`scale_pos_weight` formulation)
  * Stratified Train-Test Splitting

## Business Insights (Feature Importance)
Based on the trained AI model, the three most critical factors determining whether a customer will default are:
1. **Loan Grade (`loan_grade`):**

The bank’s internal risk category has proven to be the most accurate predictor.

2. **Debt-to-Income Ratio (`loan_percent_income`):**

Customers whose debt payments as a percentage of their total monthly income are too high have a very high risk of default.

3. **Home Ownership - RENT (`person_home_ownership_RENT`):**

Customers who are still renting their homes exhibit higher financial vulnerability compared to homeowners.


## Model Performance (XGBoost Baseline)
The model was evaluated using *testing data* (20% of the population) that was free of *data leakage* and duplicates.
* **ROC-AUC Score:** 0.9465 (Excellent)
* **Precision (Class 1 - Default):** 80%
* **Recall (Class 1 - Default):** 79%

*This means that the model successfully detected 79% of all customers who were actually in default, and when the model triggered a “Default” warning, the accuracy rate of that alert reached 80%.*

## How to Use (Inference)
The trained model is saved as `xgboost_credit_risk.pkl`. To test predictions on new customers:

1. Open a terminal and navigate to the `src` folder.
2. Run the prediction script:
   ```bash
   python predict.py