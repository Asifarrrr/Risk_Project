# Credit Risk Prediction

A Machine Learning project that predicts loan status based on personal, financial, and credit-related information using a Random Forest Classifier. The project combines data preprocessing, model training, FastAPI backend development, and an interactive Streamlit application for generating predictions and exploring credit risk data.

## Project Overview

The objective of this project is to build an end-to-end Machine Learning application that analyzes applicant information and predicts loan status based on patterns learned from historical credit data.

The application is divided into three main components:

- **Data Processing and Model Training:** Cleaning the dataset, preparing features, and training the classification model.
- **FastAPI Backend:** Handling prediction requests, validating input data, and communicating with the trained model.
- **Streamlit Frontend:** Providing an interactive interface for making predictions and visualizing dataset information.

## Technologies & Libraries

- **Python:** Main programming language.
- **Pandas:** Data manipulation and preprocessing.
- **Scikit-learn:** Machine Learning model development and training.
- **Random Forest Classifier:** Classification algorithm used for loan status prediction.
- **FastAPI:** Backend API for handling prediction requests.
- **Pydantic:** Input validation and data structure management.
- **Streamlit:** Interactive web application.
- **Plotly:** Interactive data visualization.
- **Joblib:** Saving and loading trained models and processed datasets.

## Data Preprocessing

Several preprocessing operations are performed to prepare the dataset for model training:

- Removing duplicate records.
- Cleaning and standardizing categorical values.
- Handling missing values.
- Replacing unrealistic age and employment length values.
- Calculating the loan-to-income ratio.
- Converting categorical values into numerical representations.
- Applying One-Hot Encoding to categorical features.
- Filtering records based on income values.

## Machine Learning Model

The project uses a **Random Forest Classifier** from Scikit-learn.

The selected hyperparameters are:

| Parameter | Value |
|---|---:|
| Number of estimators | 200 |
| Maximum depth | 8 |
| Minimum samples per leaf | 15 |
| Minimum samples split | 20 |
| Class weight | Balanced |
| Random state | 42 |

The dataset is divided into training and testing sets:

- Training data: 85%
- Testing data: 15%

The model is trained using the processed applicant features to predict the target variable, `loan_status`.

## Application Features

### Loan Status Prediction

The Streamlit interface allows users to enter applicant information, including:

- Age and personal income.
- Employment length.
- Loan amount and interest rate.
- Loan-to-income ratio.
- Credit history length.
- Previous default status.
- Home ownership.
- Loan intent.
- Loan grade.

The submitted information is sent to the FastAPI backend, where the trained model generates a prediction and its associated probability.

### Interactive Data Visualization Dashboard

The application also provides a dashboard for exploring the processed dataset.

Available features include:

- Income range filtering.
- Loan grade filtering.
- Loan intent filtering.
- Age vs. income scatter plot, categorized by loan status.
- Loan grade distribution pie chart for records with a positive loan status.

All visualizations are created using Plotly and integrated into Streamlit.

## Backend API

FastAPI connects the frontend to the trained Machine Learning model.

**Endpoint:**

`POST /predict`

The API performs the following operations:

1. Receives applicant information in JSON format.
2. Validates the input using Pydantic.
3. Converts the input into a Pandas DataFrame.
4. Aligns the feature columns with the model's original training structure.
5. Generates a prediction using the Random Forest Classifier.
6. Returns the prediction and class 1 probability as a JSON response.

## Project Architecture

```text
User
 |
 v
Streamlit Frontend
 |
 | JSON Request
 v
FastAPI Backend
 |
 v
Input Validation
 |
 v
Feature Alignment
 |
 v
Random Forest Model
 |
 v
Prediction + Probability
 |
 v
Streamlit Results
```

## Future Improvements

- Add comprehensive model evaluation and performance metrics.
- Experiment with different classification algorithms.
- Improve data preprocessing and validation.
- Expand the dashboard with additional visualizations.
- Deploy the application online.
- Implement automated testing.

## Disclaimer

This project is developed for educational and Machine Learning practice purposes. Predictions are not professional financial advice and should not be used as the sole basis for real-world lending decisions.
