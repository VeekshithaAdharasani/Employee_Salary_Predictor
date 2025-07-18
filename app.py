import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Salary Predictor", page_icon="💰", layout="centered")

st.title("💼 Employee Salary Predictor")
st.markdown("Enter employee details below to predict whether the salary is >50K or ≤50K.")

# Collect user input
age = st.number_input("Age", min_value=18, max_value=70, value=30)
education = st.selectbox("Education", ["High School", "Bachelors", "Masters", "PhD"])
gender = st.selectbox("Gender", ["Male", "Female"])
job_title = st.selectbox("Job Title", ["Software Engineer", "Manager", "Sales Executive", "Technician", "Developer"])
location = st.selectbox("Location", ["India", "California", "Texas", "Florida", "New York"])
experience = st.slider("Years of Experience", 0, 40, 5)

# Prepare input dataframe (column names must match training)
input_df = pd.DataFrame({
    'age': [age],
    'education': [education],
    'gender': [gender],
    'job_title': [job_title],
    'location': [location],
    'experience': [experience]
})

# Display input
st.subheader("📄 Input Summary")
st.write(input_df)

# Prediction
if st.button("🔍 Predict Salary Class"):
    prediction = model.predict(input_df)[0]
    label = ">50K" if prediction == 1 else "≤50K"
    st.success(f"🎯 Predicted Salary Class: **{label}**")
