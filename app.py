import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="💼 Salary Prediction App",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model with error handling
@st.cache_resource
def load_model():
    try:
        model = joblib.load("salary_prediction_pipeline.pkl")
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found. Please ensure 'salary_prediction_pipeline.pkl' exists in the app directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

# Data validation functions
def validate_age(age):
    if age < 18 or age > 100:
        return False, "Age must be between 18 and 100"
    return True, ""

def validate_experience(experience, age):
    if experience < 0:
        return False, "Experience cannot be negative"
    if experience > (age - 16):
        return False, "Experience cannot exceed (Age - 16) years"
    return True, ""

def get_salary_insights(age, experience, education, job_title):
    insights = []
    if experience < 2:
        insights.append("📈 Entry-level position - Consider gaining more experience")
    elif experience < 5:
        insights.append("🔄 Mid-junior level - Good growth potential")
    elif experience < 10:
        insights.append("⭐ Experienced professional - Strong market position")
    else:
        insights.append("🏆 Senior expert - Premium salary expectations")

    if education == "PhD":
        insights.append("🎓 PhD qualification adds significant value")
    elif education == "Master":
        insights.append("📚 Master's degree provides competitive advantage")

    exp_ratio = experience / (age - 18) if age > 18 else 0
    if exp_ratio > 0.8:
        insights.append("🚀 Excellent experience-to-age ratio")
    elif exp_ratio < 0.3:
        insights.append("💡 Potential for rapid career growth")

    return insights

def main():
    model = load_model()
    st.title("💼 AI-Powered Salary Prediction")

    with st.sidebar:
        st.header("📊 Prediction Settings")
        show_insights = st.checkbox("Show Career Insights", value=True)
        show_charts = st.checkbox("Show Visualization", value=True)
        currency = st.selectbox("Currency", ["₹ (INR)", "$ (USD)", "€ (EUR)", "£ (GBP)"])

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("👤 Personal Information")
        age = st.number_input("Age", 18, 100, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        workclass = st.selectbox("Workclass", ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay"])
        marital_status = st.selectbox("Marital Status", ["Never-married", "Married-civ-spouse", "Divorced", "Separated", "Widowed", "Married-spouse-absent"])
        relationship = st.selectbox("Relationship", ["Not-in-family", "Husband", "Wife", "Own-child", "Unmarried", "Other-relative"])
        native_country = st.selectbox("Country", ["United-States", "Mexico", "Philippines", "Germany", "Canada", "India", "Cuba", "England", "Jamaica", "China", "South", "Italy", "Poland", "Columbia", "Vietnam", "Guatemala","Japan", "Iran", "Honduras", "Portugal", "Ireland", "France", "Greece", "Ecuador", "Taiwan", "Thailand",  "Nicaragua", "Scotland", "Hong", "Trinadad&Tobago", "Laos", "El-Salvador", "Cambodia", "Hungary"])
        
        st.subheader("💼 Professional Details")
        education_level = st.selectbox("Education Level", ["10th", "11th", "12th", "1st-4th", "5th-6th", "7th-8th", "9th", "Assoc-acdm", "Assoc-voc", "Bachelors", "Doctorate", "HS-grad", "Masters", "Preschool", "Prof-school", "Some-college"])
        job_title = st.selectbox("Occupation", ["Exec-managerial", "Craft-repair", "Sales", "Adm-clerical", "Prof-specialty", "Tech-support", "Other-service", "Transport-moving", "Handlers-cleaners", "Farming-fishing", "Machine-op-inspct","Protective-serv", "Priv-house-serv", "Armed-Forces"])
        experience = st.slider("Years of Experience", 0, 50, 5)

    with col2:
        age_valid, age_msg = validate_age(age)
        exp_valid, exp_msg = validate_experience(experience, age)
        if not age_valid:
            st.error(age_msg)
        if not exp_valid:
            st.error(exp_msg)

        if show_insights and age_valid and exp_valid:
            st.subheader("🔍 Career Insights")
            insights = get_salary_insights(age, experience, education_level, job_title)
            for i in insights:
                st.info(i)

    st.markdown("---")

    if st.button("🔮 Predict My Salary", use_container_width=True):
        if not (age_valid and exp_valid):
            st.error("❌ Please fix the validation errors above before predicting.")
        else:
            with st.spinner("🤖 AI is analyzing your profile..."):
                time.sleep(1)

                try:
                    education_map = {"Preschool": 1,
                                     "1st-4th": 2,
                                     "5th-6th": 3,
                                     "7th-8th": 4,
                                     "9th": 5,
                                     "10th": 6,
                                     "11th": 7,
                                     "12th": 8,
                                     "HS-grad": 9,
                                     "Some-college": 10,
                                     "Assoc-acdm": 11,
                                     "Assoc-voc": 12,
                                     "Bachelors": 13,
                                     "Masters": 14,
                                     "Prof-school": 15,
                                     "Doctorate": 16
                    }
                    input_df = pd.DataFrame({
                        "age": [age],
                        "workclass": [workclass],
                        "fnlwgt": [200000], 
                        "education": [education_level],
                        "educational-num": [education_map[education_level]],
                        "marital-status": [marital_status],
                        "occupation": [job_title],
                        "relationship": [relationship],
                        "race": ["White"],
                        "gender": [gender],
                        "capital-gain": [0],
                        "capital-loss": [0],
                        "hours-per-week": [40],
                        "native-country": [native_country]
                    })

                    prediction = model.predict(input_df)[0]
                    currency_multipliers = {
                        "₹ (INR)": 1,
                        "$ (USD)": 0.012,
                        "€ (EUR)": 0.011,
                        "£ (GBP)": 0.0095
                    }

                    symbol = currency.split()[0]
                    income_class = ">50K" if prediction == 1 else "<=50K"

                    st.markdown(f"""
                    <div style='background:linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding:20px; border-radius:15px; text-align:center;'>
                        <p style='font-size:1.2rem; color:white;'>💰 Predicted Annual Salary</p>
                        <div style='font-size:2.5rem; font-weight:700; color:white;'>{symbol} {converted_salary:,.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Monthly Salary", f"{symbol} {converted_salary / 12:,.2f}")
                    with col2:
                        st.metric("Hourly Rate", f"{symbol} {converted_salary / (40 * 52):.2f}")
                    with col3:
                        st.metric("Daily Earning", f"{symbol} {converted_salary / 365:.2f}")

                    if show_charts:
                        st.subheader("📊 Salary Breakdown")
                        job_avg_salaries = {
                            "Developer": 800000,
                            "Data Scientist": 1200000,
                            "Manager": 1500000,
                            "Analyst": 700000,
                            "Engineer": 900000
                        }

                        chart_data = pd.DataFrame({
                            "Role": list(job_avg_salaries.keys()),
                            "Average Salary": list(job_avg_salaries.values())
                        })

                        fig = px.bar(
                            chart_data,
                            x="Role",
                            y="Average Salary",
                            title="Average Salary by Role (INR)",
                            color="Average Salary",
                            color_continuous_scale="viridis"
                        )
                        fig.add_hline(
                            y=prediction,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Your Prediction: ₹{prediction:,.2f}"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    st.success("✅ Prediction completed successfully!")

                    result_data = {
                        "Prediction_Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Age": age,
                        "Gender": gender,
                        "Education": education_level,
                        "Job_Title": job_title,
                        "Experience": experience,
                        "Predicted_Salary": prediction,
                        "Currency": currency,
                        "Converted_Salary": converted_salary
                    }

                    result_df = pd.DataFrame([result_data])
                    csv = result_df.to_csv(index=False)

                    st.download_button(
                        label="📅 Download Prediction Report",
                        data=csv,
                        file_name=f"salary_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    st.info("💡 Please check your input data and try again.")

if __name__ == "__main__":
    main()

