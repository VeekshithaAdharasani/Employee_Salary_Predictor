import streamlit as st
import pandas as pd
import joblib

# Load model and encoder
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Salary Classifier", layout="centered")
st.title("💼 Salary Predictor App")
st.write("Fill in the details below to predict if the salary is **>50K or <=50K**.")

# Input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 70, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        education = st.selectbox("Education", ["Bachelors", "Masters", "PhD", "HS-grad"])
    
    with col2:
        job = st.selectbox("Job Title", ["Software Engineer", "Manager", "Developer", "Technician"])
        location = st.selectbox("Location", ["India", "California", "Texas", "Florida", "New York"])
        experience = st.slider("Years of Experience", 0, 40, 5)

    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    input_df = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "education": [education],
        "job": [job],
        "location": [location],
        "experience": [experience]
    })

    # Preprocess input (encode categorical features)
    input_encoded = encoder.transform(input_df)
    
    prediction = model.predict(input_encoded)[0]
    prediction_label = ">50K" if prediction == 1 else "<=50K"

    st.success(f"✅ Predicted Salary Class: **{prediction_label}**")
