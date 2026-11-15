import joblib
import pandas as pd

# Load saved model
saved_data = joblib.load("models/churn_model.pkl")

model = saved_data["model"]

print("Saved Model Loaded Successfully!")

# Test one customer
customer = pd.read_csv("data/cleaned_customer_churn.csv")

customer = customer.drop("customerID", axis=1)
customer = customer.drop("Churn", axis=1)

customer = pd.get_dummies(customer, drop_first=True, dtype=int)

# Match training features
customer = customer.reindex(
    columns=saved_data["feature_names"],
    fill_value=0
)

prediction = model.predict(customer.iloc[[0]])

if prediction[0] == 1:
    print("Prediction: Customer may leave")
else:
    print("Prediction: Customer may stay")
