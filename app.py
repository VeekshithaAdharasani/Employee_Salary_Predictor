import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Employee Salary Predictor", layout="centered")
st.title("💼 Employee Salary Classifier")
st.markdown("Predict whether an employee earns >50K or <=50K based on their profile.")

# Sidebar inputs
st.sidebar.header("Enter Employee Details")

education = st.sidebar.selectbox("Education Level", [
    "Bachelors", "Masters", "PhD", "HS-grad", "Assoc"
])
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)
location = st.sidebar.selectbox("Location", [
    "India", "California", "Texas", "Florida"
])
job_title = st.sidebar.selectbox("Job Title", [
    "Software Engineer", "Manager", "Analyst", "Developer", "Technician"
])
age = st.sidebar.slider("Age", 18, 65, 30)
gender = st.sidebar.radio("Gender", ["Male", "Female"])

# Prepare input
input_data = pd.DataFrame({
    'education': [education],
    'experience': [experience],
    'location': [location],
    'job_title': [job_title],
    'age': [age],
    'gender': [gender]
})

st.subheader("🔍 Input Summary")
st.write(input_data)

# Predict
if st.button("Predict Salary Class"):
    prediction = model.predict(input_data)[0]
    st.success(f"🧠 Predicted Salary Class: {prediction}")
