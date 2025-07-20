import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from datetime import datetime

# Streamlit settings
st.set_page_config(page_title="💼 Salary Prediction App", layout="wide")

@st.cache_resource
def load_models():
    classifier = joblib.load("salary_prediction_pipeline.pkl")      # classification model
    regressor = joblib.load("salary_regression_pipeline.pkl")       # regression model
    return classifier, regressor

classifier, regressor = load_models()

def validate_inputs(age, experience):
    if not (18 <= age <= 100):
        return False, "Age must be between 18 and 100"
    if experience < 0 or experience > (age - 16):
        return False, "Experience must be realistic"
    return True, ""

def get_user_input():
    st.sidebar.title("⚙️ Settings")
    mode = st.sidebar.radio("Choose Prediction Mode", ["Classification (<=50K or >50K)", "Regression (Exact Salary)"])
    currency = st.sidebar.selectbox("Currency", ["₹ (INR)", "$ (USD)", "€ (EUR)", "£ (GBP)"])

    st.title("💼 Salary Prediction App")
    age = st.number_input("Age", 18, 100, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    education = st.selectbox("Education", ["10th", "12th", "Bachelors", "Masters", "Doctorate", "HS-grad", "Some-college"])
    occupation = st.selectbox("Occupation", ["Exec-managerial", "Craft-repair", "Sales", "Adm-clerical", "Prof-specialty", "Tech-support", "Other-service", "Transport-moving", "Handlers-cleaners", "Farming-fishing", "Machine-op-inspct","Protective-serv", "Priv-house-serv", "Armed-Forces"])
    experience = st.slider("Years of Experience", 0, 50, 5)
    workclass = st.selectbox("Workclass", ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay"])
    marital_status = st.selectbox("Marital Status", ["Never-married", "Married-civ-spouse", "Divorced", "Separated", "Widowed"])
    relationship = st.selectbox("Relationship", ["Not-in-family", "Husband", "Wife", "Own-child", "Unmarried", "Other-relative"])
    native_country = st.selectbox("Country", ["United-States", "India", "Mexico", "Philippines", "Germany", "Canada", "Cuba", "England"])

    return {
        "mode": mode,
        "currency": currency,
        "features": {
            "age": age,
            "workclass": workclass,
            "fnlwgt": 200000,
            "education": education,
            "educational-num": 12,
            "marital-status": marital_status,
            "occupation": occupation,
            "relationship": relationship,
            "race": "White",
            "gender": gender,
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": native_country
        },
        "experience": experience
    }

def make_prediction(mode, features):
    input_df = pd.DataFrame([features])
    if mode.startswith("Classification"):
        label = classifier.predict(input_df)[0]
        return ">50K" if label == 1 else "<=50K"
    else:
        salary = regressor.predict(input_df)[0]
        return salary

def display_result(mode, prediction, currency):
    symbol = currency.split()[0]
    st.subheader("🔍 Prediction Result")

    if mode.startswith("Classification"):
        st.success(f"Predicted Income Category: **{prediction}**")
    else:
        converted_salary = prediction * {
            "₹ (INR)": 1,
            "$ (USD)": 0.012,
            "€ (EUR)": 0.011,
            "£ (GBP)": 0.0095
        }[currency]

        st.markdown(f"""
        <div style='background:linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding:20px; border-radius:15px; text-align:center;'>
            <p style='font-size:1.2rem; color:white;'>💰 Predicted Annual Salary</p>
            <div style='font-size:2.5rem; font-weight:700; color:white;'>{symbol} {converted_salary:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Monthly Salary", f"{symbol} {converted_salary / 12:,.2f}")
        col2.metric("Hourly Rate", f"{symbol} {converted_salary / (40 * 52):.2f}")
        col3.metric("Daily Earning", f"{symbol} {converted_salary / 365:.2f}")

def main():
    user_input = get_user_input()
    valid, error_msg = validate_inputs(user_input["features"]["age"], user_input["experience"])

    if not valid:
        st.error(error_msg)
        return

    if st.button("🔮 Predict"):
        with st.spinner("Analyzing your profile..."):
            time.sleep(1)
            result = make_prediction(user_input["mode"], user_input["features"])
            display_result(user_input["mode"], result, user_input["currency"])

if __name__ == "__main__":
    main()
