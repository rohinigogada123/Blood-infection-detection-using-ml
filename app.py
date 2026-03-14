import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# -----------------------------
# Project Paths
# -----------------------------
app_dir = Path(__file__).parent
project_root = app_dir.parent

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Blood Infection Risk Detection System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown("""
<style>

.high-risk{
background-color:#ffe5e5;
padding:25px;
border-radius:12px;
border-left:6px solid red;
font-size:28px;
font-weight:bold;
color:#8b0000;
text-align:center;
}

.low-risk{
background-color:#e6ffe6;
padding:25px;
border-radius:12px;
border-left:6px solid green;
font-size:28px;
font-weight:bold;
color:#006400;
text-align:center;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(project_root / "models" / "infection_model.pkl")
        scaler = joblib.load(project_root / "models" / "scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


# -----------------------------
# Feature Engineering
# -----------------------------
def engineer_features(df):

    df = df.copy()

    df["wbc_temp_interaction"] = df["wbc_count"] * df["temperature"]
    df["lactate_glucose_ratio"] = df["lactate"] / (df["glucose"] + 1e-5)

    return df


# -----------------------------
# Detection Pipeline
# -----------------------------
def detect_infection(model, scaler, df):

    df = engineer_features(df)

    feature_order = [
        "wbc_count",
        "temperature",
        "heart_rate",
        "respiratory_rate",
        "lactate",
        "glucose",
        "platelet_count",
        "bilirubin",
        "wbc_temp_interaction",
        "lactate_glucose_ratio"
    ]

    df = df[feature_order]

    X = scaler.transform(df)

    prediction = model.predict(X)

    probability = None

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(X)[:,1]

    return prediction, probability


# -----------------------------
# Detection Page
# -----------------------------
def detection_page(model, scaler):

    st.header("🧬 Blood Infection Risk Detection")

    tab1, tab2, tab3 = st.tabs(["📝 Manual Input","📤 Upload CSV","📋 Sample Data"])

    with tab1:

        st.subheader("🧾 Enter Patient Medical Data")

        col1, col2 = st.columns(2)

        with col1:

            wbc = st.number_input(
                "WBC Count (10³/μL)",
                min_value=0.0,
                max_value=50.0,
                value=0.0,
                help="Normal range: 4.5 – 11"
            )

            temp = st.number_input(
                "Temperature (°C)",
                min_value=30.0,
                max_value=42.0,
                value=30.0,
                help="Normal range: 36.5 – 37.5"
            )

            hr = st.number_input(
                "Heart Rate (bpm)",
                min_value=40,
                max_value=200,
                value=40,
                help="Normal range: 60 – 100"
            )

            rr = st.number_input(
                "Respiratory Rate (/min)",
                min_value=10,
                max_value=50,
                value=10,
                help="Normal range: 12 – 20"
            )

        with col2:

            lactate = st.number_input(
                "Lactate (mmol/L)",
                min_value=0.0,
                max_value=20.0,
                value=0.0,
                help="Normal range: 0.5 – 2"
            )

            glucose = st.number_input(
                "Glucose (mg/dL)",
                min_value=40,
                max_value=400,
                value=40,
                help="Normal range: 70 – 100"
            )

            platelets = st.number_input(
                "Platelet Count (10³/μL)",
                min_value=0,
                max_value=500,
                value=0,
                help="Normal range: 150 – 400"
            )

            bilirubin = st.number_input(
                "Bilirubin (mg/dL)",
                min_value=0.0,
                max_value=20.0,
                value=0.0,
                help="Normal range: 0.1 – 1.2"
            )

        if st.button("🔍 Detect Infection Risk"):

            if wbc == 0 or platelets == 0:
                st.warning("⚠️ Please enter valid patient values before running detection.")
                return

            df = pd.DataFrame({
                "wbc_count":[wbc],
                "temperature":[temp],
                "heart_rate":[hr],
                "respiratory_rate":[rr],
                "lactate":[lactate],
                "glucose":[glucose],
                "platelet_count":[platelets],
                "bilirubin":[bilirubin]
            })

            pred, prob = detect_infection(model, scaler, df)

            st.markdown("### 📊 Detection Results")

            colA, colB = st.columns([1,2])

            with colA:

                if pred[0] == 1:

                    st.markdown("""
                    <div class="high-risk">
                    <h1>⚠️ HIGH RISK</h1>
                    <p>Blood Infection Likely</p>
                    </div>
                    """, unsafe_allow_html=True)

                else:

                    st.markdown("""
                    <div class="low-risk">
                    <h1>✅ LOW RISK</h1>
                    <p>Blood Infection Unlikely</p>
                    </div>
                    """, unsafe_allow_html=True)

            with colB:

                if prob is not None:
                    st.metric("⚠️ Risk Score", f"{prob[0]*100:.2f}%")

                st.markdown("### 💊 Recommendations")

                if pred[0] == 1:

                    st.warning("""
- Immediate medical attention required
- Consider sepsis protocol initiation
- Blood cultures recommended
- ICU monitoring suggested
""")

                else:

                    st.info("""
- Continue routine monitoring
- Regular vital signs checks
- Follow-up as clinically indicated
""")

            st.markdown("### 📑 Input Data Summary")
            st.dataframe(df)


# -----------------------------
# About Page
# -----------------------------
def about_page():

    st.header("📊 About This System")

    st.write("""
This system detects **blood infection risk (sepsis)** using Machine Learning models.

The model analyzes patient vital signs and laboratory values to estimate infection probability.
""")


# -----------------------------
# Instructions Page
# -----------------------------
def instructions_page():

    st.header("ℹ️ Instructions")

    st.write("""
1. Enter patient data manually OR upload CSV file.
2. Click **Detect Infection Risk**.
3. The system will display infection probability and clinical recommendations.
""")


# -----------------------------
# Main Application
# -----------------------------
def main():

    model, scaler = load_artifacts()

    if model is None:
        st.stop()

    st.sidebar.title("🧭 Navigation")

    page = st.sidebar.radio(
        "Select Page",
        ["🔍 Detection","📊 About","ℹ️ Instructions"]
    )

    if page == "🔍 Detection":
        detection_page(model, scaler)

    elif page == "📊 About":
        about_page()

    else:
        instructions_page()


if __name__ == "__main__":
    main()
