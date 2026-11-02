import joblib
import requests
import streamlit as st
import plotly.express as px

# ============================= Streamlit UI Configuration =============================

st.set_page_config(page_title='Risk Model', layout='wide')
st.title('Loan Status Prediction AI Model')
st.caption('Powered by Random Forest Classifier')

# ============================= Dataset Loading =============================

df_preview = None

try:
    # Load the preprocessed dataset used for the dashboard.
    df_preview = joblib.load('data_frame.pkl')

    # Convert one-hot encoded loan grade columns back into a single categorical column.
    grade_cols = ['loan_grade_a','loan_grade_b','loan_grade_c','loan_grade_d','loan_grade_e'
                 ,'loan_grade_f','loan_grade_g']
    df_preview['loan_grade'] = df_preview[grade_cols].idxmax(axis=1).str.replace('loan_grade_', '')

    # Convert one-hot encoded loan intent columns back into a single categorical column.
    intent_cols = ['loan_intent_debtconsolidation','loan_intent_education','loan_intent_homeimprovement'
                   ,'loan_intent_medical','loan_intent_personal','loan_intent_venture']
    df_preview['loan_intent'] = df_preview[intent_cols].idxmax(axis=1).str.replace('loan_intent_', '')

    # Display the first 100 rows of the dataset inside an expandable section.
    with st.expander('First 100 rows of the dataset'):
        st.dataframe(df_preview.head(100))

except FileNotFoundError:
    pass

st.divider()

# ============================= Prediction Interface =============================

# Divide the main page into two columns: user inputs and prediction results.
left_col, right_col = st.columns(2)

with left_col:
    st.subheader('Person Features')
    st.write('Enter the features bellow')

    # Collect the numerical and categorical features required by the model.
    age = st.number_input('Age', min_value=20, max_value=94, step=1)
    person_income = st.number_input('Person Income', 4000.0, 800000.0, step=100.0)
    person_emp_len = st.number_input('Person Employment Length', min_value=0, max_value=41)
    loan_amount = st.number_input('Loan Amount', min_value=500, max_value=35000, step=10)
    loan_rate = st.number_input('Loan Rate', min_value=5.42, max_value=23.22, step=0.2)
    loan_income = st.number_input('Loan Income %', min_value=0.0, max_value=0.83, step=0.1)
    cb_cred_hist = st.number_input('Person Credit History', min_value=2, max_value=30, step=1)

    cb_def = st.selectbox('Person Default On File', options=['Yes', 'No'])
    own = st.selectbox('Home Ownership', options=['Mortgage', 'Other', 'Own', 'Rent'])
    loan_intent = st.selectbox('Loan Intent', options=['Debt Consolidation', 'Education', 'Home Improvement'
                                                            ,'Medical', 'Personal', 'Venture'])
    grade = st.selectbox('Loan Grade', options=['a', 'b', 'c', 'd', 'e', 'f', 'g'])

with right_col:
    st.subheader('AI Prediction')
    st.write('Click Predict to start the operation')

    if st.button('Predict'):

        # ============================= Categorical Encoding =============================

        # Convert the default status into a binary value.
        cb = 1 if cb_def == 'Yes' else 0

        # One-hot encode the selected home ownership category.
        own_m = 1 if own == 'Mortgage' else 0
        own_o = 1 if own == 'Other' else 0
        own_own = 1 if own == 'Own' else 0
        own_r = 1 if own == 'Rent' else 0

        # One-hot encode the selected loan intent.
        loan_d = 1 if loan_intent == 'Debt Consolidation' else 0
        loan_e = 1 if loan_intent == 'Education' else 0
        loan_h = 1 if loan_intent == 'Home Improvement' else 0
        loan_m = 1 if loan_intent == 'Medical' else 0
        loan_p = 1 if loan_intent == 'Personal' else 0
        loan_v = 1 if loan_intent == 'Venture' else 0

        # One-hot encode the selected loan grade.
        grade_a = 1 if grade == 'a' else 0
        grade_b = 1 if grade == 'b' else 0
        grade_c = 1 if grade == 'c' else 0
        grade_d = 1 if grade == 'd' else 0
        grade_e = 1 if grade == 'e' else 0
        grade_f = 1 if grade == 'f' else 0
        grade_g = 1 if grade == 'g' else 0

        # ============================= API Request Payload =============================

        # Organize all input features into a dictionary.
        # The keys must match the feature names expected by the API/model.
        payload = {
            'person_age': age,
            'person_income': person_income,
            'person_emp_length': person_emp_len,
            'loan_amnt': loan_amount,
            'loan_int_rate': loan_rate,
            'loan_percent_income': loan_income,
            'cb_person_default_on_file': cb,
            'cb_person_cred_hist_length': cb_cred_hist,
            'person_home_ownership_mortgage': own_m,
            'person_home_ownership_other': own_o,
            'person_home_ownership_own': own_own,
            'person_home_ownership_rent': own_r,
            'loan_intent_debtconsolidation': loan_d,
            'loan_intent_education': loan_e,
            'loan_intent_homeimprovement': loan_h,
            'loan_intent_medical': loan_m,
            'loan_intent_personal': loan_p,
            'loan_intent_venture': loan_v,
            'loan_grade_a': grade_a,
            'loan_grade_b': grade_b,
            'loan_grade_c': grade_c,
            'loan_grade_d': grade_d,
            'loan_grade_e': grade_e,
            'loan_grade_f': grade_f,
            'loan_grade_g': grade_g
        }

        # ============================= Model Prediction =============================

        try:
            # Send the input features to the FastAPI prediction endpoint.
            response = requests.post('http://127.0.0.1:8000/predict', json=payload)

            if response.status_code == 200:

                # Extract the prediction and its probability from the API response.
                result = response.json()
                prediction = result['Prediction']
                probability = result['Probability']

            # Display the prediction result.
            if prediction == 1:
                st.success(f'Result: **Yes**, Probability: {round(probability * 100, 2)}%')
            else:
                st.error(f'Result: **No**, Probability: {round(probability * 100, 2)}%')

        except requests.exceptions.ConnectionError:
            # Display an error if the FastAPI server is not running or reachable.
            st.error('Could not connect to API server!')

    else:
        st.info('Fill the features before clicking (Predict)')

st.divider()

# ============================= Dashboard Filters =============================

with st.sidebar:
    st.subheader('Dashboard Features')

    # Get the minimum and maximum income values for the income range slider.
    min_value_income, max_value_income = df_preview['person_income'].min(), df_preview['person_income'].max()

    # Define the available loan grades and intents.
    loan_grade_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    loan_intent_list = ['debtconsolidation', 'education', 'homeimprovement', 'medical', 'personal', 'venture']

    # Collect filter selections from the sidebar.
    income = st.slider('Loan Income', min_value_income, max_value_income, (min_value_income, max_value_income))
    loan_grade = st.selectbox('Loan Grade', options=loan_grade_list)
    loan_int = st.selectbox('Loan Intent', options=loan_intent_list)

    # Filter the dataset according to the selected income range, grade, and intent.
    filtered_data = df_preview[
        (df_preview['person_income'] >= income[0]) &
        (df_preview['person_income'] <= income[1]) &
        (df_preview['loan_grade'] == loan_grade) &
        (df_preview['loan_intent'] == loan_int)
    ]

# ============================= Data Visualization Dashboard =============================

st.subheader('Data Visualization Dashboard')

if not filtered_data.empty:

    # Display the dashboard charts side by side.
    col1, col2 = st.columns(2)

    with col1:
        # Scatter plot showing the relationship between age and income.
        # Colors represent the loan status of each person.
        fig1 = px.scatter(
            filtered_data,
            x='person_age',
            y='person_income',
            color='loan_status',
            hover_data=['person_emp_length', 'loan_percent_income', 'loan_grade', 'loan_intent'],
            title='Age vs Income (Colored By Loan Status)'
        )

        fig1.update_layout(template='plotly_dark')
        st.plotly_chart(fig1, width='stretch')

    with col2:
        # Select records with a positive loan status for the selected loan intent.
        valid_status = df_preview[
            (df_preview['loan_status'] == 1) &
            (df_preview['loan_intent'] == loan_int)
        ]

        # Pie chart showing the distribution of loan grades among these records.
        fig2 = px.pie(
            valid_status,
            names='loan_grade',
            title='Loan Status Proportion On Loan Grade'
        )

        fig2.update_layout(template='plotly_dark')
        st.plotly_chart(fig2, width='stretch')

else:
    # Inform the user when no records match the selected dashboard filters.
    st.warning('Filled features did not match any person, try other features!')