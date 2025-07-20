import streamlit as st
import pandas as pd
import joblib
import time
from datetime import datetime

# Set Streamlit page config
st.set_page_config(page_title="💰 Salary Regression Predictor", layout="wide")

# Load only the regression model
@st.cache_resource
def load_regression_model():
    return joblib.load("salary_regression_pipeline.pkl")

regressor = load_regression_model()

# Input form
def get_user_input():
    st.title("💼 Salary Prediction (Regression Only)")
    
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    education = st.selectbox("Education", ["10th", "12th", "Bachelors", "Masters", "Doctorate", "HS-grad", "Some-college"])
    occupation = st.selectbox("Occupation", ["Exec-managerial", "Craft-repair", "Sales", "Adm-clerical", "Prof-specialty", 
                                              "Tech-support", "Other-service", "Transport-moving", "Handlers-cleaners", 
                                              "Farming-fishing", "Machine-op-inspct", "Protective-serv", "Priv-house-serv", "Armed-Forces"])
    experience = st.slider("Years of Experience", 0, 50, 5)
    workclass = st.selectbox("Workclass", ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", 
                                           "State-gov", "Without-pay"])
    marital_status = st.selectbox("Marital Status", ["Never-married", "Married-civ-spouse", "Divorced", "Separated", "Widowed"])
    relationship = st.selectbox("Relationship", ["Not-in-family", "Husband", "Wife", "Own-child", "Unmarried", "Other-relative"])
    native_country = st.selectbox("Country", ["United-States", "India", "Mexico", "Philippines", "Germany", "Canada", "Cuba", "England"])
    currency = st.selectbox("Currency", ["₹ (INR)", "$ (USD)", "€ (EUR)", "£ (GBP)"])

    # Return all input data
    return {
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
        "native-country": native_country,
        "currency": currency
    }

# Predict using regression model
def predict_salary(input_data):
    df = pd.DataFrame([input_data])
    salary = regressor.predict(df)[0]
    return salary

# Display prediction result
def display_prediction(salary, currency):
    symbol = currency.split()[0]
    converted = salary * {
        "₹ (INR)": 1,
        "$ (USD)": 0.012,
        "€ (EUR)": 0.011,
        "£ (GBP)": 0.0095
    }[currency]

    st.subheader("🔍 Predicted Salary")
    st.markdown(f"""
        <div style='background:linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding:20px; border-radius:15px; text-align:center;'>
            <p style='font-size:1.2rem; color:white;'>Predicted Annual Salary</p>
            <div style='font-size:2.5rem; font-weight:700; color:white;'>{symbol} {converted:,.2f}</div>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Monthly Salary", f"{symbol} {converted / 12:,.2f}")
    col2.metric("Hourly Rate", f"{symbol} {converted / (40 * 52):.2f}")
    col3.metric("Daily Earning", f"{symbol} {converted / 365:.2f}")

# Main app flow
def main():
    user_input = get_user_input()

    if st.button("🔮 Predict Salary"):
        with st.spinner("Calculating..."):
            time.sleep(1)
            salary = predict_salary(user_input)
            display_prediction(salary, user_input["currency"])

if __name__ == "__main__":
    main()
