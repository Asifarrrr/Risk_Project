import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ============================= FastAPI Application =============================

# Initialize the FastAPI application.
app = FastAPI(title='Risk API')


# ============================= Model Loading =============================

# Load the previously trained Random Forest model.
model = joblib.load('risk_model.pkl')

# Load the feature names and their original order used during model training.
model_columns = joblib.load('model_columns.pkl')


# ============================= Input Data Validation =============================

# Define the structure and data types expected from incoming API requests.
# Pydantic automatically validates the received input.
class PersonInput(BaseModel):
    person_age: int
    person_income: float | int
    person_emp_length: float | int
    loan_amnt: int
    loan_int_rate: float | int
    loan_percent_income: float | int
    cb_person_default_on_file: int
    cb_person_cred_hist_length: int

    # Home ownership features (One-Hot Encoded).
    person_home_ownership_mortgage: int
    person_home_ownership_other: int
    person_home_ownership_own: int
    person_home_ownership_rent: int

    # Loan intent features (One-Hot Encoded).
    loan_intent_debtconsolidation: int
    loan_intent_education: int
    loan_intent_homeimprovement: int
    loan_intent_medical: int
    loan_intent_personal: int
    loan_intent_venture: int

    # Loan grade features (One-Hot Encoded).
    loan_grade_a: int
    loan_grade_b: int
    loan_grade_c: int
    loan_grade_d: int
    loan_grade_e: int
    loan_grade_f: int
    loan_grade_g: int


# ============================= Prediction Endpoint =============================

# Define the POST endpoint that receives input features and returns a prediction.
@app.post('/predict')
def Loan_Prediction(data: PersonInput):

    # Convert the validated input data into a Pandas DataFrame.
    df = pd.DataFrame([data.model_dump()])

    # Reorder the input columns to match the model's training feature order.
    # Missing columns, if any, are filled with zeros.
    df = df.reindex(columns=model_columns, fill_value=0)

    # Generate the predicted loan status.
    prediction = int(model.predict(df)[0])

    # Calculate the probability associated with class 1 (positive loan status).
    probability = float(model.predict_proba(df)[0][1])

    # Return the prediction and probability as a JSON response.
    return {
        'Prediction': prediction,
        'Probability': probability
    }