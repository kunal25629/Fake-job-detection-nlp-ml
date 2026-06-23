
import streamlit as st
import pandas as pd
import joblib

# PAGE CONFIG
st.set_page_config(page_title="Fake Job Detector", page_icon="🔍", layout="wide")


# LOAD MODEL
@st.cache_resource
def load_model():
    return joblib.load("fake_job_detector.pkl")

model = load_model()


# SESSION STATE
if "history" not in st.session_state:
    st.session_state.history = []


# SIDEBAR
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go To", ["Prediction", "About Project"])
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Model Performance")
st.sidebar.metric("Accuracy", "98.77%")
st.sidebar.metric("Precision", "88.62%")
st.sidebar.metric("Recall", "85.55%")
st.sidebar.metric("F1 Score", "87.06%")
st.sidebar.metric("ROC-AUC", "99.17%")


# ABOUT PAGE
if page == "About Project":

    st.title("📊 Fake Job Posting Detection")

    st.markdown("---")

    st.write("""
### Project Overview

This Machine Learning application detects whether a job posting is Genuine or Fraudulent.

### Features

✅ NLP Based Text Processing

✅ TF-IDF Vectorization

✅ One-Hot Encoding

✅ XGBoost Classification

✅ Real-Time Prediction

### Model Performance

- Accuracy: 98.77%
- Precision: 88.62%
- Recall: 85.55%
- F1 Score: 87.06%
- ROC-AUC: 99.17%

### Tech Stack

- Python
- Pandas
- Scikit-Learn
- XGBoost
- Streamlit
""")


# PREDICTION PAGE
if page == "Prediction":

    st.title("🔍 Fake Job Detection System")

    st.markdown(
        "Detect whether a job posting is **Genuine** or **Fraudulent**.")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        title = st.text_input("Job Title")

        company_profile = st.text_area("Company Profile")

        description = st.text_area("Job Description")

        requirements = st.text_area("Requirements")

        benefits = st.text_area("Benefits")

    with col2:

        employment_type = st.selectbox(
            "Employment Type",
            [
                "Full-time",
                "Part-time",
                "Contract",
                "Temporary",
                "Unknown"
            ]
        )

        required_experience = st.selectbox(
            "Required Experience",
            [
                "Internship",
                "Entry level",
                "Associate",
                "Mid-Senior level",
                "Executive",
                "Unknown"
            ]
        )

        required_education = st.selectbox(
            "Required Education",
            [
                "High School",
                "Bachelor's Degree",
                "Master's Degree",
                "Doctorate",
                "Unknown"
            ]
        )

        industry = st.text_input("Industry")

        function = st.text_input("Function")

        salary_range = st.text_input("Salary Range")

        telecommuting = st.selectbox("Remote Job", [0, 1])

        has_company_logo = st.selectbox("Company Logo Available",[0, 1])

        has_questions = st.selectbox(
            "Screening Questions",[0, 1])

    st.markdown("---")

    if st.button(
        "🚀 Predict",
        use_container_width=True):

        # Input Validation

        if title.strip() == "" or description.strip() == "":
            st.warning("Please enter Job Title and Job Description.")
            st.stop()

        try:
            text = (
                str(title) + " " +
                str(company_profile) + " " +
                str(description) + " " +
                str(requirements) + " " +
                str(benefits)
            )

            input_df = pd.DataFrame({
                "text": [text],
                "employment_type": [employment_type],
                "required_experience": [required_experience],
                "required_education": [required_education],
                "industry": [industry],
                "function": [function],
                "salary_range": [salary_range],
                "telecommuting": [telecommuting],
                "has_company_logo": [has_company_logo],
                "has_questions": [has_questions]
            })

            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)

            fake_prob = probability[0][1]
            real_prob = probability[0][0]

            confidence = max(fake_prob,real_prob)
            st.markdown("## Prediction Result")

            if prediction == 1:

                st.error("⚠️ Fake Job Detected")

            else:

                st.success("✅ Genuine Job Posting")

            st.metric("Confidence Score",f"{confidence:.2%}")
            st.progress(float(confidence))
            st.markdown("---")
            st.subheader("📊 Prediction Probabilities")

            prob_df = pd.DataFrame({
                "Class": ["Genuine","Fake"],
                "Probability": [real_prob,fake_prob]
            })

            st.bar_chart(prob_df.set_index("Class"))
            st.write(f"✅ Genuine Job Probability: {real_prob:.2%}")
            st.write(f"⚠️ Fake Job Probability: {fake_prob:.2%}")

            # History
            st.session_state.history.append({

                "Job Title": title,

                "Prediction":
                "Fake"
                if prediction == 1
                else "Genuine",

                "Confidence (%)":
                round(
                    confidence * 100,
                    2
                )
            })

            if len(st.session_state.history) > 0:

                st.markdown("---")
                st.subheader("📜 Prediction History")
                history_df = pd.DataFrame(st.session_state.history)
                st.dataframe(history_df,use_container_width=True)

        except Exception as e:

            st.error(f"Error occurred: {e}")
