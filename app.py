import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Salary Prediction", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Predictor")
st.markdown("This app predicts whether an employee earns **>50K** or **≤50K** based on their profile.")

# --- User Input ---
st.header("📋 Enter Employee Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 65, 30)
    education = st.selectbox("Education Level", [
        "Bachelors", "Masters", "PhD", "HS-grad", "Assoc", "Some-college"
    ])
    experience = st.slider("Years of Experience", 0, 40, 5)

with col2:
    occupation = st.selectbox("Job Role", [
        "Tech-support", "Craft-repair", "Other-service", "Sales",
        "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
        "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
        "Protective-serv", "Armed-Forces"
    ])
    hours = st.slider("Hours per week", 1, 80, 40)

# --- Format Input for Model ---
input_df = pd.DataFrame({
    'age': [age],
    'education': [education],
    'occupation': [occupation],
    'hours-per-week': [hours],
    'experience': [experience]
})

# --- Prediction ---
if st.button("🔍 Predict Salary Class"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Salary Class: **{prediction}**")
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")

# --- Batch Prediction ---
st.markdown("---")
st.subheader("📁 Batch Prediction (CSV Upload)")
batch_file = st.file_uploader("Upload CSV", type="csv")

if batch_file:
    try:
        data = pd.read_csv(batch_file)
        st.write("Data Preview:", data.head())
        results = model.predict(data)
        data['Prediction'] = results
        st.write("✅ Results:")
        st.write(data.head())

        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results", csv, file_name="predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"⚠️ Error processing file: {str(e)}")
