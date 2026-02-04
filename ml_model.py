import pandas as pd 
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
# Load data
df = pd.read_csv("customer_churn.csv")

# Drop ID column
df.drop(columns=['customer_id'], inplace=True)
print(df.drop)

# feature engineering
df['new_customer'] = (df['tenure'] <= 6)
df['heavy_user'] = (df['monthly_usage'] > 220)
df['high_support'] = (df['support_calls'] >= 3)

# Feature & target
x = df[['age','income','tenure','monthly_usage',
        'support_calls','contract_type',
        'new_customer','heavy_user','high_support']]

y = df['churn']

# Train -test split
x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=0.2,random_state = 42)

#Feature scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# user input
user_input= np.array([[36,56000,13,340,10,1,0,1,1]])
user_input_scaled = scaler.transform(user_input)

# Model Training
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
    "Random Forest": RandomForestClassifier(n_estimators=200),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Model Training evaluate


Recall = {}



for name, model in models.items():
    
    if name == "Logistic Regression":
        model.fit(x_train_scaled, y_train)
        y_pred = model.predict(x_test_scaled)
        user_pred = model.predict(user_input_scaled)[0]

    else:
        model.fit(x_train_scaled, y_train)
        y_pred = model.predict(x_test_scaled)
        user_pred = model.predict(user_input)[0]

    Recall[name] = recall_score(y_test, y_pred)


    print("\n==============================")
    print(f" Model: {name}")
    print("User Churn Prediction:", "Churn" if user_pred == 1 else "No Churn")
    print("Accuracy",accuracy_score(y_test,y_pred))
    print("Recall_Score",recall_score(y_test,y_pred))
    print("Precision",precision_score(y_test,y_pred))
    print("F1-score",f1_score(y_test,y_pred))
    print("\n Confusion Matrix:")
    print(confusion_matrix(y_test,y_pred))

    print("\n Classification Report:")
    print(classification_report(y_test,y_pred))


# Select Best Model

for model, score in Recall.items():
    print(model, "Recall", score)

best_model_name = max(Recall, key=Recall.get)
print("\nBest Model is:", best_model_name)

# FEATURE IMPORTANCE

# Random Forest
rf_importance = pd.DataFrame({
    "Feature": x.columns,
    "Importance": models["Random Forest"].feature_importances_
}).sort_values(by="Importance", ascending=False)

print(rf_importance)

