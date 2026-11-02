# Credit Risk Prediction

A Machine Learning project that predicts loan status based on personal, financial, and credit-related information. The project uses a Random Forest Classifier to analyze applicant features and generate predictions through an interactive web application.

## Project Overview

The main goal of this project is to build an end-to-end Machine Learning application that combines data preprocessing, model training, API development, and data visualization.

Users can enter personal and financial information through a Streamlit interface and receive a prediction from a trained Random Forest model. The application also includes an interactive dashboard for exploring and visualizing the credit risk dataset.

## Technologies Used

- **Python:** Main programming language.
- **Pandas:** Data manipulation and preprocessing.
- **Scikit-learn:** Machine Learning model development and training.
- **Random Forest Classifier:** Classification algorithm used for loan status prediction.
- **FastAPI:** Backend API for handling prediction requests.
- **Pydantic:** Input data validation.
- **Streamlit:** Interactive web application and user interface.
- **Plotly:** Interactive data visualizations.
- **Joblib:** Saving and loading trained models and processed data.

## Project Structure

```text
Credit-Risk-Prediction/
│
├── data_cleaning.py       # Data preprocessing and model training
├── api.py                 # FastAPI backend and prediction endpoint
├── app.py                 # Streamlit frontend and dashboard
│
├── credit_risk_dataset.csv
├── data_frame.pkl         # Generated processed dataset
├── risk_model.pkl         # Generated trained model
├── model_columns.pkl      # Generated model feature names
│
├── .gitignore
├── requirements.txt
└── README.md
```

The generated `.pkl` files and original dataset are not included in the Git repository. Run the preprocessing and training script to generate the required model files.

## Data Preprocessing

Before training the model, several preprocessing operations are performed:

- Removing duplicate records.
- Cleaning and standardizing categorical values.
- Handling missing values.
- Replacing unrealistic age and employment length values.
- Calculating the loan-to-income percentage.
- Converting categorical values into numerical representations.
- Applying One-Hot Encoding to categorical features.
- Removing records with income values above the selected threshold.

## Machine Learning Model

The project uses a Random Forest Classifier from Scikit-learn.

Selected model parameters:

- Number of estimators: 200
- Maximum depth: 8
- Minimum samples per leaf: 15
- Minimum samples split: 20
- Class weight: Balanced
- Random state: 42

The dataset is divided into training and testing sets, with 85% used for training and 15% reserved for testing.

## Backend API

FastAPI handles communication between the frontend and the trained Machine Learning model.

The API provides a prediction endpoint:

`POST /predict`

It receives applicant information as JSON, validates the input using Pydantic, prepares the features in the correct order, and passes them to the trained model.

The response contains:

- **Prediction:** The predicted loan status.
- **Probability:** The model's estimated probability for class 1.

## Streamlit Application

The frontend provides two main functionalities:

### Loan Status Prediction

Users can enter information such as:

- Age and annual income.
- Employment length.
- Loan amount and interest rate.
- Credit history.
- Previous default status.
- Home ownership.
- Loan intent.
- Loan grade.

The application sends these features to the FastAPI backend and displays the prediction and its associated probability.

### Data Visualization Dashboard

The dashboard allows users to explore the dataset using interactive filters and charts.

It includes:

- Income range filtering.
- Loan grade filtering.
- Loan intent filtering.
- Age vs. income scatter plot.
- Loan grade distribution pie chart.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Credit-Risk-Prediction.git
cd Credit-Risk-Prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset and train the model

Place the original dataset in the project directory and run:

```bash
python data_cleaning.py
```

This generates the required model and processed dataset files.

### 4. Start the FastAPI server

```bash
uvicorn api:app --reload
```

The API will be available at:

`http://127.0.0.1:8000`

FastAPI documentation:

`http://127.0.0.1:8000/docs`

### 5. Start the Streamlit application

Open another terminal and run:

```bash
streamlit run app.py
```

The application will open in your browser.

## Future Improvements

- Add model evaluation metrics and performance analysis.
- Improve error handling and input validation.
- Experiment with other Machine Learning algorithms.
- Improve the dashboard with additional visualizations.
- Deploy the API and Streamlit application online.
- Add automated testing.

## Disclaimer

This project is developed for educational and Machine Learning practice purposes. Predictions should not be considered professional financial advice or used as the sole basis for actual lending decisions.

## Author

Developed as a personal Machine Learning project to practice data preprocessing, classification, API development, and interactive data visualization.
