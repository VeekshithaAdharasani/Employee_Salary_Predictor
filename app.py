import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open("best_model.pkl", "rb") as file:
    model = pickle.load(file)

# Define encoders (same label mappings as in your notebook)
workclass_mapping = {
    'Private': 4, 'Local-gov': 2,
    'State-gov': 6, 'Federal-gov': 1,
    'Others': 0
}

occupation_mapping = {
    'Prof-specialty': 9, 'Craft-repair': 1, 'Exec-managerial': 3,
    'Adm-clerical': 0, 'Sales': 11, 'Other-service': 8,
    'Machine-op-inspct': 6, 'Transport-moving': 13, 'Handlers-cleaners': 5,
    'Tech-support': 12, 'Farming-fishing': 4, 'Protective-serv': 10,
    'Priv-house-serv': 7, 'Armed-Forces': 2, 'Others': 14
}

gender_mapping = {'Male': 1, 'Female': 0, 'Other': 2}
marital_mapping = {
    'Never-married': 2, 'Married-civ-spouse': 1, 'Divorced': 0,
    'Separated': 3, 'Widowed': 4, 'Married-spouse-absent': 5
}

# Streamlit UI
st.title("💼 Employee Salary Predictor")

# Collect user input
age = st.number_input("🎂 Age", min_value=18, max_value=75, value=30)
educational_num = st.slider("🎓 Education Level (5-16)", 5, 16, 13)
workclass = st.selectbox("🏢 Workclass", list(workclass_mapping.keys()))
occupation = st.selectbox("🛠️ Occupation", list(occupation_mapping.keys()))
marital_status = st.selectbox("💍 Marital Status", list(marital_mapping.keys()))
gender = st.selectbox("🧑 Gender", list(gender_mapping.keys()))
capital_gain = st.number_input("💰 Capital Gain", value=0)
hours_per_week = st.slider("🕒 Hours per Week", 1, 99, 40)

# Predict button
if st.button("🔮 Predict Salary"):
    # Encode categorical variables
    encoded_input = pd.DataFrame([{
        'age': age,
        'educational-num': educational_num,
        'workclass': workclass_mapping[workclass],
        'occupation': occupation_mapping[occupation],
        'marital-status': marital_mapping[marital_status],
        'gender': gender_mapping[gender],
        'capital-gain': capital_gain,
        'hours-per-week': hours_per_week
    }])

    # Prediction
    prediction = model.predict(encoded_input)[0]
    st.success(f"💰 Predicted Salary: ${prediction:,.2f}")
