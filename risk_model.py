import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ============================= Data Loading =============================

df = None

try:
    # Load the original credit risk dataset.
    df = pd.read_csv(r"C:\Users\pc\PycharmProjects\WelcomeScreen\Mini Projects\Risk_Project\credit_risk_dataset.csv")

except FileNotFoundError:
    pass


# ============================= Data Cleaning =============================

# Remove duplicate records from the dataset.
df = df.drop_duplicates()

# Standardize categorical text values by removing extra spaces
# and converting all characters to lowercase.
text = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

for col in text:
    df[col] = df[col].str.strip().str.lower()


# Replace unrealistic age values with missing values.
df.loc[df['person_age'] > 100, 'person_age'] = None

# Fill missing age values using the column mean and convert them to integers.
df['person_age'] = df['person_age'].fillna(df['person_age'].mean())
df['person_age'] = df['person_age'].astype(int)


# Replace unrealistic employment length values with missing values.
df.loc[df['person_emp_length'] > 42, 'person_emp_length'] = None

# Remove records that have missing employment length values.
df = df.dropna(subset='person_emp_length')


# Fill missing loan interest rates using the column mean.
# Round the calculated mean to two decimal places.
df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].mean().round(2))


# Calculate the loan amount as a percentage of the person's income.
# This feature represents the proportion of income required to cover the loan.
df['loan_percent_income'] = (((df['loan_amnt'] * 100) / df['person_income']) / 100).round(2)


# Convert the default status from categorical values into binary values.
# Y = 1 (Default), N = 0 (No Default).
df['cb_person_default_on_file'] = df['cb_person_default_on_file'].replace({
    'y': 1,
    'n': 0
})

# Ensure that the default status column contains numeric integer values.
df['cb_person_default_on_file'] = pd.to_numeric(
    df['cb_person_default_on_file'],
    errors='coerce'
).astype(int)


# Apply One-Hot Encoding to categorical features.
# This converts categorical values into separate numerical columns.
df = pd.get_dummies(
    df,
    columns=['person_home_ownership', 'loan_intent', 'loan_grade'],
    dtype=int
)


# Remove records with income values above the selected threshold.
df = df[df['person_income'] <= 480000]


# ============================= Feature Selection =============================

# Separate the input features (X) from the target variable (y).
# loan_status is the value the model will learn to predict.
x = df.drop(columns=['loan_status'])
y = df['loan_status']


# ============================= Train-Test Split =============================

# Split the dataset into training and testing sets.
# 85% of the data is used for training and 15% for testing.
# random_state ensures reproducibility of the split.
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.15,
    random_state=42
)


# ============================= Random Forest Model =============================

# Initialize the Random Forest Classifier with the selected hyperparameters.
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=15,
    min_samples_split=20,
    class_weight='balanced',
    random_state=42
)

# Train the model using the training dataset.
model.fit(x_train, y_train)


# ============================= Model and Data Export =============================

# Save the processed dataset for the Streamlit dashboard.
joblib.dump(df, 'data_frame.pkl')

# Save the trained Random Forest model for future predictions.
joblib.dump(model, 'risk_model.pkl')

# Save the feature names and their order to ensure that future prediction
# inputs follow the same structure expected by the trained model.
joblib.dump(list(x.columns), 'model_columns.pkl')