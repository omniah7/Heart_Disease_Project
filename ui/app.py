import streamlit as st
import pandas as pd
import joblib
import io

# Config
st.set_page_config(page_title="Heart Disease Predictor", layout="wide", page_icon="❤️")

DATA_PATH = "data/heart_disease.csv"
MODEL_PATH = "models/final_model.pkl"

# Load resources with caching for performance
@st.cache_resource
def load_assets():
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    return df, model

df, model = load_assets()

# -----------------------------
# Sidebar: organized with Expanders
# -----------------------------
st.sidebar.header("📋 Patient Information")

with st.sidebar.expander("👤 Basic Demographics", expanded=True):
    age = st.number_input("Age", min_value=20, max_value=100, value=50)
    sex = st.radio("Sex", options=[("Female", 0), ("Male", 1)], format_func=lambda x: x[0])[1]

with st.sidebar.expander("🩺 Clinical Vital Signs"):
    trestbps = st.number_input("Resting BP (mm Hg)", 90, 200, 120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", 50, 700, 240)
    thalach = st.number_input("Max Heart Rate", 50, 250, 150)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]

with st.sidebar.expander("🧪 Test Results"):
    cp = st.selectbox("Chest Pain Type", [("Typical Angina", 1), ("Atypical Angina", 2), ("Non-anginal", 3), ("Asymptomatic", 4)], format_func=lambda x: x[0])[1]
    restecg = st.selectbox("Resting ECG", [("Normal", 0), ("ST-T Abnormality", 1), ("LV Hypertrophy", 2)], format_func=lambda x: x[0])[1]
    exang = st.radio("Exercise Induced Angina", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 10.0, 1.0, 0.1)
    slope = st.selectbox("ST Slope", [("Upsloping", 1), ("Flat", 2), ("Downsloping", 3)], format_func=lambda x: x[0])[1]
    ca = st.selectbox("Major Vessels (ca)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [("Normal", 3), ("Fixed Defect", 6), ("Reversible Defect", 7)], format_func=lambda x: x[0])[1]

# Prepare Data
data = {
    "age": age, "sex": sex, "trestbps": trestbps, "chol": chol, "fbs": fbs, "thalach": thalach,
    "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca,
    "cp_2": 1 if cp == 2 else 0, "cp_3": 1 if cp == 3 else 0, "cp_4": 1 if cp == 4 else 0,
    "restecg_1": 1 if restecg == 1 else 0, "restecg_2": 1 if restecg == 2 else 0,
    "thal_6": 1 if thal == 6 else 0, "thal_7": 1 if thal == 7 else 0
}
user_df = pd.DataFrame(data, index=[0])

# -----------------------------
# Main Page Layout
# -----------------------------
st.title("❤️ Heart Disease Diagnostic Dashboard")
st.markdown("---")

# Top row: Prediction Results
m1, m2, m3 = st.columns([1, 1, 2])

pred = model.predict(user_df)[0]
# probability if the model supports it
prob = model.predict_proba(user_df)[0][1] if hasattr(model, "predict_proba") else None

with m1:
    st.metric("Model Status", "Active" if model else "Offline")
with m2:
    risk_label = "HIGH" if pred == 1 else "LOW"
    st.metric("Risk Level", risk_label, delta="High Risk" if pred == 1 else "Normal", delta_color="inverse")

with m3:
    confidence = prob if pred == 1 else (1-prob)
    if pred == 1:
        st.error(f"### 🚨 High Risk Detected")
    else:
        st.success(f"### ✅ Low Risk Detected")

    st.write(f"Model confidence: {confidence*100:.1f}%")


# Lower section: Tabs for Data and Viz
tab1, tab2, tab3 = st.tabs(["📊 Patient Data", "📈 Trends", "📁 Dataset Info"])

with tab1:
    st.write("Current Input Vector:")
    st.dataframe(user_df, use_container_width=True)
    
    csv_buffer = io.StringIO()
    user_df.to_csv(csv_buffer, index=False)
    st.download_button("📩 Download Results", csv_buffer.getvalue(), "prediction.csv", "text/csv")

with tab2:
    st.subheader("How this patient compares to the dataset")
    # Example: Distribution of age
    import plotly.express as px
    fig = px.histogram(df, x="age", nbins=20, title="Age Distribution of Patients")
    fig.add_vline(x=age, line_color='red', annotation_text="Current Patient")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.write("Dataset Preview (Top 5 Rows)")
    st.dataframe(df.head(), use_container_width=True)

# Footer
st.markdown("---")
st.caption("Sprint X Microsoft Bootcamp Project | Developed by Omniah")