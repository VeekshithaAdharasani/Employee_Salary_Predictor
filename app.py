import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="💼 Salary Prediction App",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Load trained pipeline
model = joblib.load("salary_prediction_pipeline.pkl")

# Page config
st.set_page_config(page_title="Salary Predictor 💼", layout="centered")
st.markdown("<h1 style='text-align: center; color: #3366cc;'>AI-Powered Salary Predictor</h1>", unsafe_allow_html=True)
st.markdown("### 🚀 Predict your salary based on your profile")

# Sidebar animation
with st.sidebar:
    st.image("https://media.giphy.com/media/3o7TKzH2YEUOAY0cyk/giphy.gif", use_container_width=True)
    st.write("🧠 Powered by CatBoost ML Model")

# Input fields
education = st.selectbox("🎓 Education", ["High School", "Bachelor", "Master", "PhD", "Diploma"])
age = st.slider("📅 Age", 18, 65, 25)
location = st.selectbox("📍 Location", ["Urban", "Rural", "Suburban"])
gender= st.selectbox("👤 Gender", ["Male", "Female"])
job_title = st.selectbox("💻 Job_Title", ["Manager", "Director", "Analyst", "Engineer", "Accountant", "Other"])
experience = st.slider("💼 Experience (Years)", 0, 40, 2)

# Predict button
if st.button("Predict Salary 💰"):
    input_df = pd.DataFrame({
        "Education": [education],
        "Age": [age],
        "Location": [location],
        "Gender": [gender],
        "Job_Title": [job_title],
        "Experience": [experience],
    })

    try:
        prediction = model.predict(input_df)[0]
        # Additional Metrices
        st.success(f"✅ Estimated Monthly Salary: {'₹'} {prediction:,.2f}")

        # Additional metrics
        annual_salary = prediction * 12
        hourly_rate = prediction / (40 * 4.33) 
        daily_earning = prediction / 30

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Annual Salary", f"{currency} {annual_salary:,.2f}")
            with col2:
                st.metric("Hourly Rate", f"{currency} {hourly_rate:,.2f}")
                with col3:
                    st.metric("Daily Earning", f"{currency} {daily_earning:,.2f}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
    
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
