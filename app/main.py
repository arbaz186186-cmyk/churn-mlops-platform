import joblib
import pandas as pd
from fastapi import FastAPI

app = FastAPI(title="Customer Churn Prediction API")

# Load saved ML model
saved_data = joblib.load("models/churn_model.pkl")
model = saved_data["model"]
feature_names = saved_data["feature_names"]

# Load training dataset for consistent encoding
training_data = pd.read_csv("data/cleaned_customer_churn.csv")
training_data = training_data.drop(
    columns=["customerID", "Churn"],
    errors="ignore"
)

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}


@app.post("/predict")
def predict_churn(customer: dict):

    customer_data = pd.DataFrame([customer])

    # Apply same encoding using training categories
    combined_data = pd.concat(
        [training_data, customer_data],
        ignore_index=True
    )

    combined_data = pd.get_dummies(
        combined_data,
        drop_first=True,
        dtype=int
    )

    # Extract only the new customer's row
    data = combined_data.iloc[[-1]]

    # Match training feature columns
    data = data.reindex(
        columns=feature_names,
        fill_value=0
    )

    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "Customer may leave"
    else:
        result = "Customer may stay"

    return {"prediction": result}
