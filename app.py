import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Employee Salary Prediction", layout="centered")

st.title("💼 Employee Salary Predictor")

# Load model
model = joblib.load("best_model.pkl")

# Input fields
st.header("Enter Employee Details:")
experience = st.slider("Years of Experience", 0, 50, 1)
education = st.selectbox("Education Level", ["Bachelors", "Masters", "PhD"])
job_title = st.selectbox("Job Title", ["Software Engineer", "Data Scientist", "Manager", "Analyst"])
location = st.selectbox("Location", ["New York", "San Francisco", "Austin", "Remote"])

# Convert inputs to dummy format (basic example, adjust as needed)
education_dict = {"Bachelors": 0, "Masters": 1, "PhD": 2}
job_dict = {"Software Engineer": 0, "Data Scientist": 1, "Manager": 2, "Analyst": 3}
location_dict = {"New York": 0, "San Francisco": 1, "Austin": 2, "Remote": 3}

features = np.array([[experience, education_dict[education], job_dict[job_title], location_dict[location]]])

# Prediction
if st.button("Predict Salary"):
    prediction = model.predict(features)[0]
    st.success(f"💰 Predicted Salary: ${prediction:,.2f}")