import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("best_model.pkl")

# Title
st.set_page_config(page_title="Employee Salary Predictor", layout="centered")
st.title("💼 Employee Salary Predictor")

# Input Fields
st.subheader("Enter Employee Details")

age = st.slider("Age", 18, 65, 30)
experience = st.slider("Years of Experience", 0, 50, 5)
education = st.selectbox("Education Level", ["Bachelors", "Masters", "PhD", "HS-grad", "Some-college"])
job_title = st.selectbox("Job Title", ["Software Engineer", "Manager", "Data Scientist", "Analyst"])
location = st.selectbox("Location", ["New York", "San Francisco", "Austin", "Remote"])
gender = st.selectbox("Gender", ["Male", "Female"])

# Manual Encoding (must match training)
education_dict = {"HS-grad": 0, "Some-college": 1, "Bachelors": 2, "Masters": 3, "PhD": 4}
job_dict = {"Software Engineer": 0, "Manager": 1, "Data Scientist": 2, "Analyst": 3}
location_dict = {"New York": 0, "San Francisco": 1, "Austin": 2, "Remote": 3}
gender_dict = {"Male": 0, "Female": 1}

# Convert input to DataFrame
input_data = pd.DataFrame([[
    age,
    experience,
    education_dict[education],
    job_dict[job_title],
    location_dict[location],
    gender_dict[gender]
]], columns=["age", "experience", "education", "job_title", "location", "gender"])

# Prediction
if st.button("Predict Salary Class"):
    prediction = model.predict(input_data)[0]
    result = ">50K" if prediction == 1 else "≤50K"
    st.success(f"✅ Predicted Salary Class: {result}")
