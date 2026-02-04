import pandas as pd 

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
# Load data
df = pd.read_csv("data/customers.csv")

# Feature & target
x = df[['age','income','tenure','monthly_usage']]
y= df['churn']

# Train -test split
x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=0.2,random_state = 42)

#Feature scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
# Model Training
model = LogisticRegression()
model.fit(x_train_scaled,y_train)

# Prediction
y_pred = model.predict(x_test_scaled)

# Evaluation 
print("Accuracy",accuracy_score(y_test,y_pred))
print("Recall_Score",recall_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))
print("F1-score",f1_score(y_test,y_pred))
print("\n Confusion Matrix:")
print(confusion_matrix(y_test,y_pred))

print("\n Classification Report:")

print(classification_report(y_test,y_pred))

# FEATURE IMPORTANCE

importance = pd.DataFrame({
    "Feature": x.columns,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

print("\nFeature Importance:")
print(importance)
