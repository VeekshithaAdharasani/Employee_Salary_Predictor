import streamlit as st
import pandas as pd
import joblib

# Load the trained model and encoder
model = joblib.load("best_model.pkl")
encoder = joblib.load("encoder.pkl")  # Encoder used during training (e.g., OneHotEncoder or ColumnTransformer)

# Streamlit App Config
st.set_page_config(page_title="💼 Salary Classifier", page_icon="💰", layout="centered")
st.title("💼 Employee Salary Predictor")
st.write("Fill in the employee details to predict whether the salary is <=50K or >50K.")

# Input Fields
age = st.slider("Age", 18, 65, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
education = st.selectbox("Education", [
    "Bachelors", "Masters", "PhD", "HS-grad", "Assoc"])
experience = st.slider("Years of Experience", 0, 40, 5)
job_title = st.selectbox("Job Title", [
    "Engineer", "Data Scientist", "Manager", "Sales Executive", "Developer", "HR"])
location = st.selectbox("Location", [
    "India", "California", "Texas", "Florida", "New York"])

# Build Input DataFrame
input_data = pd.DataFrame({
    'age': [age],
    'gender': [gender],
    'education': [education],
    'experience': [experience],
    'job_title': [job_title],
    'location': [location]
})

# Show user input
st.subheader("Input Summary")
st.table(input_data)

# Predict
if st.button("🔍 Predict Salary Class"):
    try:
        input_encoded = encoder.transform(input_data)
        prediction = model.predict(input_encoded)[0]
        label = ">50K" if prediction == 1 else "<=50K"
        st.success(f"✅ Predicted Salary Class: {label}")
    except Exception as e:
        st.error(f"⚠️ Prediction Failed: {e}")
