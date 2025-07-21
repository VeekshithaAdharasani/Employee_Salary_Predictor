import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="💼 Employee Salary Prediction App",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <div style='text-align: center; padding: 20px; border: 2px solid #eee; border-radius: 10px; background-color: #f9f9f9;'>
        <h1 style='color: #4CAF50;'>🏢 Company Logo</h1>
        <h2>🎓 Internship Project</h2>
        <p style='font-size: 18px;'>
            <strong>Company:</strong> Edunet Foundation in collaboration with AICTE and IBM<br>
            <strong>Intern:</strong> Veekshitha Adharasani <br>
            <strong>Project:</strong> Machine Learning Salary Predictor<br>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)



# Load trained pipeline
model = joblib.load("salary_prediction_pipeline.pkl")
# Load the best model and its performance metrics
model, mse, r2 = joblib.load("best_model.pkl") 

# Sidebar: show model performance
st.sidebar.header("📈 Model Performance")
st.sidebar.metric("MSE (Mean Squared Error)", f"{mse:,.2f}")
st.sidebar.metric("R² Score", f"{r2:.4f}")

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

st.markdown("""
    <style>
    div.stButton > button {
        display: block;
        margin: auto;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        font-size: 18px;
        border-radius: 10px;
        padding: 0.6em 2em;
    }
    </style>
""", unsafe_allow_html=True)
# Validation and Prediction
# Check if all fields are filled
if not (education and location and gender and job_title and experience is not None):
    st.warning("⚠️ Please fill in all the input fields before proceeding.")
elif experience > age:
    st.error("❌ Experience cannot be greater than Age. Please correct the input.")
elif education == "PhD" and experience < 3:
    st.warning("⚠️ PhD holders typically have at least 3 years of experience.")
elif education == "Master" and experience < 1:
    st.warning("⚠️ Master graduates usually have at least 1 year of experience.")
elif education == "High School" and experience > (age - 16):
    st.warning("⚠️ High School grads cannot have that much experience at this age.")
else:
    try:
        input_df = pd.DataFrame({
            "Education": [education],
            "Age": [age],
            "Location": [location],
            "Gender": [gender],
            "Job_Title": [job_title],
            "Experience": [experience],
        })
            
        # Predict salary
        prediction = model.predict(input_df)[0]
        MIN_SALARY = 33510.51
        MAX_SALARY = 193016.60
        prediction = max(MIN_SALARY, min(MAX_SALARY, prediction))
        st.markdown(
            f"""
            <div style='text-align: center; margin-top: 30px; margin-bottom: 30px;'>
            <h1 style='font-size: 48px; color: green;'>✅ ₹ {prediction:,.2f}</h1>
            <h4 style='margin-top: -10px;'>Estimated Monthly Salary</h4>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Additional metrics
        annual_salary = prediction * 12
        hourly_rate = prediction / (40 * 4.33) 
        daily_earning = prediction / 30
        col1, col2, col3 = st.columns(3)
            
        with col1:
            st.metric("Annual Salary", f"{'₹'} {annual_salary:,.2f}")
        with col2:
            st.metric("Hourly Rate", f"{'₹'} {hourly_rate:,.2f}")
        with col3:
            st.metric("Daily Earning", f"{'₹'} {daily_earning:,.2f}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        #Visualization
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

    st.markdown(
            f"""
            <div style='text-align: center; margin-top: 30px; margin-bottom: 30px;'>
            <h1 style='font-size: 48px; color: green;'>✅ ₹ {prediction:,.2f}</h1>
            <h4 style='margin-top: -10px;'>Estimated Monthly Salary</h4>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Additional metrics
        annual_salary = prediction * 12
        hourly_rate = prediction / (40 * 4.33) 
        daily_earning = prediction / 30
        col1, col2, col3 = st.columns(3)
            
        with col1:
            st.metric("Annual Salary", f"{'₹'} {annual_salary:,.2f}")
        with col2:
            st.metric("Hourly Rate", f"{'₹'} {hourly_rate:,.2f}")
        with col3:
            st.metric("Daily Earning", f"{'₹'} {daily_earning:,.2f}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        #Visualization
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

        #  Model Evaluation Summary
        st.markdown("## 📊 Model Evaluation Summary")
        try:
            eval_df = pd.read_csv("model_evaluation.csv")
            def highlight_best(s):
                is_best = s["Model"] == "CatBoost"
                return ['background-color: lightgreen' if b else '' for b in is_best]
                st.dataframe(eval_df.style.apply(highlight_best, axis=1), use_container_width=True)
                
        st.markdown("## 🔍 Sample Predictions and Errors")
        try:
            sample_df = pd.read_csv("sample_predictions.csv")
            st.dataframe(sample_df.head(20), use_container_width=True)
            avg_error = sample_df["Absolute Error"].mean()
            st.info(f"📉 Average Absolute Error on test samples: ₹ {avg_error:,.2f}")
            
            # Load prediction results
            try:
                sample_df = pd.read_csv("sample_predictions.csv")
                # Scatter plot: Actual vs Predicted
                fig_scatter = px.scatter(
                    sample_df,
                    x="Actual Salary",
                    y="Predicted Salary (CatBoost)",
                    title="📈 Actual vs Predicted Salaries (CatBoost)",
                    labels={"Actual Salary": "Actual", "Predicted Salary (CatBoost)": "Predicted"},
                    trendline="ols"
                )
                fig_scatter.update_layout(height=500)
                st.plotly_chart(fig_scatter, use_container_width=True)
                # ✅ Salary Distribution Histogram (Predicted)
                fig_hist = px.histogram(
                    sample_df,
                    x="Predicted Salary (CatBoost)",
                    nbins=20,
                    title="💰 Salary Distribution (Predicted)",
                    labels={"Predicted Salary (CatBoost)": "Predicted Salary"}
                )
                fig_hist.update_layout(bargap=0.1, height=400)
                st.plotly_chart(fig_hist, use_container_width=True)
       
            except Exception as e:
                st.warning("⚠️ Could not display prediction graph.")
                st.text(f"Error: {e}")
           
        except Exception as e:
            st.warning("⚠️ Could not load sample prediction data.")
            st.text(f"Error: {e}")

        except Exception as eval_error:
            st.warning("⚠️ Could not load model evaluation data.")
            st.text(f"Error: {eval_error}")
        
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
