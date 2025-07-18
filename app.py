import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Salary Prediction App", layout="centered")
st.title("💼 Employee Salary Prediction")
st.markdown("Predict whether an employee earns **>50K** or **≤50K**")

# Input form
with st.form("prediction_form"):
    st.subheader("Enter Employee Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=65, value=30)
        education = st.selectbox("Education", ["Bachelors", "Masters", "PhD", "HS-grad", "Assoc", "Some-college"])
        experience = st.slider("Years of Experience", 0, 40, 5)

    with col2:
        occupation = st.selectbox("Occupation", [
            "Tech-support", "Craft-repair", "Other-service", "Sales",
            "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
            "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
            "Protective-serv", "Armed-Forces"
        ])
        hours = st.slider("Hours per Week", 1, 80, 40)

    submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame({
            "age": [age],
            "education": [education],
            "occupation": [occupation],
            "hours-per-week": [hours],
            "experience": [experience]
        })

        try:
            prediction = model.predict(input_df)[0]
            st.success(f"Prediction: **{prediction}**")
        except Exception as e:
            st.error(f"Error: {e}")

# Batch prediction
st.markdown("---")
st.subheader("📁 Batch Prediction")
csv_file = st.file_uploader("Upload CSV", type=["csv"])

if csv_file is not None:
    try:
        df = pd.read_csv(csv_file)
        st.write("📄 Uploaded Preview:", df.head())

        preds = model.predict(df)
        df['Prediction'] = preds
        st.write("✅ Predictions:")
        st.write(df.head())

        download = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results", download, file_name="salary_predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Error: {e}")
