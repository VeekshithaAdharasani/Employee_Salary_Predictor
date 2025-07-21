import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from PIL import Image

# Load trained pipeline
model = joblib.load("salary_prediction_pipeline.pkl")

# Page config
st.set_page_config(page_title="Salary Predictor 💼", layout="centered")
st.markdown("<h1 style='text-align: center; color: #3366cc;'>AI-Powered Salary Predictor</h1>", unsafe_allow_html=True)
st.markdown("### 🚀 Predict your salary based on your profile")

# Sidebar animation
with st.sidebar:
    st.image("https://media.giphy.com/media/3o7TKzH2YEUOAY0cyk/giphy.gif", use_column_width=True)
    st.write("🧠 Powered by CatBoost ML Model")

# Input fields
education = st.selectbox("🎓 Qualification", ["High School", "Bachelor", "Master", "PhD", "Diploma"])
experience = st.slider("💼 Work Experience (Years)", 0, 40, 2)
age = st.slider("📅 Age", 18, 65, 25)
location = st.selectbox("📍 Location", ["Urban", "Rural", "Suburban"])
gender = st.radio("👤 Gender", ["Male", "Female"])
job_title = st.selectbox("💻 Job Title", ["Manager", "Director", "Analyst", "Engineer", "Accountant", "Other"])

# Predict button
if st.button("Predict Salary 💰"):
    input_df = pd.DataFrame({
        "Eduaction": [education],
        "Experience": [experience],
        "Age": [age],
        "Location": [location],
        "Gender": [gender],
        "Job_Title": [job_title]
    })

    try:
        prediction = model.predict(input_df)[0]
        st.success(f"✅ Estimated Salary: ₹ {prediction:,.2f}")

        # Visualization
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            title={'text': "Estimated Salary (INR)"},
            gauge={
                'axis': {'range': [None, 5000000]},
                'bar': {'color': "#4caf50"},
                'steps': [
                    {'range': [0, 500000], 'color': "#f9f9f9"},
                    {'range': [500000, 2000000], 'color': "#cde9d6"},
                    {'range': [2000000, 5000000], 'color': "#a5d6a7"}
                ],
            }
        ))
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
