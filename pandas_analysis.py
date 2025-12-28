import pandas as pd

# Load data
df = pd.read_csv("data/customers.csv")
print(df)

print("Shape:", df.shape)
print("\nInfo:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\n Duplicate values:")
print(df.duplicated().sum())

print("\nChurn distribution:")
print(df['churn'].value_counts())

print("\nChurn percentage:")
print(df["churn"].value_counts(normalize=True) * 100)

print("\nAverage values by churn:")
print(df.groupby("churn").mean(numeric_only=True))

print(df.corr(numeric_only=True)['churn'].sort_values(ascending =False))

high_risk = df[
    (df["tenure"]<6)&
    (df["monthly_usage"]>200)
]

print("\nHigh-risk customers:")
print(high_risk)

