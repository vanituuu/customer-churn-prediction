# Customer Churn Analysis & Prediction (EDA + Multi-Model ML)
**Project Overview**

Customer churn is a critical business problem where companies lose customers due to dissatisfaction or better alternatives.
This project focuses on analyzing customer behavior, identifying churn drivers, and building multiple machine learning models to predict churn effectively.

# Objective

• Analyze customer data to understand churn patterns

• Perform Exploratory Data Analysis (EDA)

• Engineer meaningful features

• Build and compare multiple ML models

• Prioritize recall to avoid missing potential churn customers

 # Exploratory Data Analysis (EDA)

• Churn distribution analysis

• Missing value & duplicate check

• Correlation heatmap

• Tenure vs churn

• Monthly usage vs churn

• Key Observations:

• New customers have a higher churn rate

• Customers with high usage and frequent support calls are more likely to churn


# Feature Engineering

Additional features created to improve model performance:

new_customer → tenure ≤ 6 months

heavy_user → monthly usage > 220

high_support → support calls ≥ 3

These features helped capture customer behavior patterns more effectively.

# Machine Learning Models Used

The following models were trained and compared:

# Model	Purpose
Logistic Regression	Baseline & interpretability
Decision Tree	Rule-based understanding
Random Forest	Capture non-linear patterns
Gradient Boosting	Performance optimization

# Model Evaluation

Accuracy

Precision

Recall 

F1-score
Why Recall?

In churn prediction, missing a churn customer is costlier than falsely flagging a loyal one.
Therefore, recall was prioritized during model selection.

# Model Comparison

All models were evaluated on the same test set, and results were compared in a single table.
The best model was selected based on recall score.

# Feature Importance

Random Forest feature importance was used to identify top churn drivers

Key Drivers of Churn:

Low tenure

High monthly usage

Frequent support calls

Month-to-month contracts

# Business Insights & Recommendations

Improve onboarding experience for new customers

Monitor high-usage customers closely for service issues

Offer incentives to move customers from monthly to yearly contracts

These actions can significantly reduce churn risk.

🧠 Skills Demonstrated

Exploratory Data Analysis (EDA)

Feature Engineering

Classification Modeling

Model Comparison

Business-oriented ML thinking


🧑‍💻 Tech Stack

Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

