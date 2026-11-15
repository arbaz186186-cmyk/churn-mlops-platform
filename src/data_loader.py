import pandas as pd

data = pd.read_csv("data/raw/customer_churn.csv")

print("Dataset Loaded Successfully")
print(data.head())

print("Dataset Shape:", data.shape)
print("Column Names:", data.columns.tolist())

print("Missing Values")
print(data.isnull().sum())

print("TotalCharges Data Type:", data["TotalCharges"].dtype)

data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"], errors="coerce"
)

print("TotalCharges Missing After Conversion:",
      data["TotalCharges"].isnull().sum())

data["TotalCharges"] = data["TotalCharges"].fillna(
    data["TotalCharges"].median()
)

print("Missing Values After Cleaning:",
      data["TotalCharges"].isnull().sum())

print("Customer Churn Count:")
print(data["Churn"].value_counts())

print("Churn Percentage:")
print(data["Churn"].value_counts(normalize=True) * 100)

data.to_csv("data/cleaned_customer_churn.csv", index=False)

print("Cleaned dataset saved successfully")

print("Numerical Columns:")
print(data.select_dtypes(include="number").columns.tolist())

print("Categorical Columns:")
print(data.select_dtypes(include=["str", "object"]).columns.tolist())
data = data.drop("customerID", axis=1)

print("Customer ID Removed")
print("New Dataset Shape:", data.shape)

# Separate target column
y = data["Churn"].map({"Yes": 1, "No": 0})

# Separate input features
X = data.drop("Churn", axis=1)

# Convert text columns into numerical columns
X = pd.get_dummies(X, drop_first=True, dtype=int)

print("Encoded Features Shape:", X.shape)
print("Target Shape:", y.shape)
print(X.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)
print("Training Labels:", y_train.shape)
print("Testing Labels:", y_test.shape)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

print("Model Training Successful!")

from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy * 100, "%")

print("Classification Report:")
print(classification_report(y_test, y_pred))


import joblib

joblib.dump(
    {
        "model": model,
        "feature_names": X.columns.tolist()
    },
    "models/churn_model.pkl"
)

print("Model Saved Successfully!")
