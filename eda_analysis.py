import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("customers_churn_2000.csv")

# 1. Basic info
print("Shape:", df.shape)
print("\nData Info:")
print(df.info())

# 2. Missing values
print("\nMissing values:")
print(df.isnull().sum())

# 3. Duplicate rows
print("\nDuplicate rows:", df.duplicated().sum())

# 4. Statistical summary
print("\nStatistical summary:")
print(df.describe())

# 5. Churn distribution
sns.countplot(x="churn", data=df)
plt.title("Churn Distribution")
plt.show()

# 6. Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 7. Boxplot
sns.boxplot(x='churn', y='tenure', data=df)
plt.title("Tenure vs Churn")
plt.show()

sns.boxplot(x='churn', y='monthly_usage', data=df)
plt.title("Monthly Usage vs Churn")
plt.show()




