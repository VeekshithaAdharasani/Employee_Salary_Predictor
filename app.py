import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import time
import plotly.express as px

# Page config
st.set_page_config(
    page_title="💼 Salary Prediction App",
    page_icon="💰",
    layout="wide"
)

# Load both models
@st.cache_resource
def load_models():
    classifier = joblib.load("xgb_classifier.pkl")
    regressor = joblib.load("xgb_regressor.pkl")
    return classifier, regressor

classifier, regressor = load_models()

st.title("💼 AI-Powered Salary Prediction App")

# Sidebar options
with st.sidebar:
    st.header("⚙️ Prediction Settings")
    mode = st.radio("Select Prediction Mode", ["Classification", "Regression"])
    currency = st.selectbox("Currency", ["₹ (INR)", "$ (USD)", "€ (EUR)", "£ (GBP)"])

# Main input section
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 100, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    workclass = st.selectbox("Workclass", ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay"])
    marital_status = st.selectbox("Marital Status", ["Never-married", "Married-civ-spouse", "Divorced", "Separated", "Widowed", "Married-spouse-absent"])
    relationship = st.selectbox("Relationship", ["Not-in-family", "Husband", "Wife", "Own-child", "Unmarried", "Other-relative"])
    native_country = st.selectbox("Country", ["United-States", "Mexico", "India", "Philippines", "Canada"])

with col2:
    education_level = st.selectbox("Education Level", ["10th", "11th", "12th", "HS-grad", "Some-college", "Bachelors", "Masters", "Doctorate"])
    occupation = st.selectbox("Occupation", ["Exec-managerial", "Craft-repair", "Sales", "Prof-specialty", "Tech-support", "Other-service"])
    hours_per_week = st.slider("Hours per Week", 1, 80, 40)
    capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
    capital_loss = st.number_input("Capital Loss", 0, 100000, 0)
    fnlwgt = st.number_input("Fnlwgt", 10000, 1000000, 200000)

st.markdown("---")

if st.button("🔮 Predict My Salary"):
    with st.spinner("Analyzing your profile..."):
        time.sleep(1)

        input_df = pd.DataFrame({
            "age": [age],
            "workclass": [workclass],
            "fnlwgt": [fnlwgt],
            "education": [education_level],
            "educational-num": [13],
            "marital-status": [marital_status],
            "occupation": [occupation],
            "relationship": [relationship],
            "race": ["White"],
            "gender": [gender],
            "capital-gain": [capital_gain],
            "capital-loss": [capital_loss],
            "hours-per-week": [hours_per_week],
            "native-country": [native_country]
        })

        symbol = currency.split()[0]
        currency_multipliers = {"₹ (INR)": 1, "$ (USD)": 0.012, "€ (EUR)": 0.011, "£ (GBP)": 0.0095}

        try:
            if mode == "Classification":
                prediction = classifier.predict(input_df)[0]
                income_class = ">50K" if prediction == 1 else "<=50K"
                st.success(f"🏷️ Predicted Income Class: {income_class}")
            else:
                salary = regressor.predict(input_df)[0]
                converted_salary = salary * currency_multipliers[currency]
                st.markdown(f"""
                <div style='background:#38ef7d;padding:20px;border-radius:10px;'>
                    <h3 style='color:white;'>Predicted Salary: {symbol} {converted_salary:,.2f}</h3>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.info("💡 Please check your input data and try again.")

