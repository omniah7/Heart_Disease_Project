import streamlit as st
import pandas as pd
import joblib
import io

st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

# -----------------------------
# Load resources
# -----------------------------
DATA_PATH = "data/heart_disease.csv"
MODEL_PATH = "models/final_model.pkl"

df = pd.read_csv(DATA_PATH)

model = joblib.load(MODEL_PATH)

# -----------------------------
# Sidebar: User Inputs
# -----------------------------
st.sidebar.header("Enter patient data")

def user_input_features():
    age = st.sidebar.number_input("Age (years)", min_value=20, max_value=100, value=50)
    sex = sex = st.sidebar.radio("Sex", options=[("Female", 0), ("Male", 1)], format_func=lambda x: x[0])[1]
    cp = st.sidebar.selectbox("Chest pain type (cp)", options=[
                    ("Typical Angina", 1),
                    ("Atypical Angina", 2), 
                    ("Non-anginal Pain", 3),
                    ("Asymptomatic", 4)
                ], format_func=lambda x: x[0])[1]
    trestbps = st.sidebar.number_input("Resting blood pressure (trestbps) (mm Hg)", min_value=90, max_value=200, value=120)
    chol = st.sidebar.number_input("Serum cholesterol (chol) (mg/dl)", min_value=50, max_value=700, value=240)
    fbs = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dl", 
                options=[("False", 0), ("True", 1)], 
                format_func=lambda x: x[0])[1]
    restecg = st.sidebar.selectbox("Resting ECG (restecg)", options=[
                    ("Normal", 0),
                    ("ST-T Wave Abnormality", 1),
                    ("Left Ventricular Hypertrophy", 2)
                ],
                format_func=lambda x: x[0])[1]
    thalach = st.sidebar.number_input("Max heart rate achieved (thalach)", min_value=50, max_value=250, value=150)
    exang = st.sidebar.selectbox("Exercise induced angina (exang)", options=[("No", 0), ("Yes", 1)],
                format_func=lambda x: x[0])[1]
    oldpeak = st.sidebar.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.sidebar.selectbox("Slope of peak exercise ST segment (slope)", options=[
                    ("Upsloping", 1),
                    ("Flat", 2),
                    ("Downsloping", 3)],
                format_func=lambda x: x[0])[1]
    ca = st.sidebar.number_input("number of major vessels (0-3) colored by flourosopy (ca)",
                           min_value=0, max_value=3, value=0)
    thal = st.sidebar.selectbox("Thalassemia (thal)", options=[
                    ("Normal", 3),
                    ("Fixed Defect", 6),
                    ("Reversible Defect", 7)],
                format_func=lambda x: x[0]
            )[1]
    # Convert categorical variables to one-hot encoded format
    cp_2 = 1 if cp == 2 else 0
    cp_3 = 1 if cp == 3 else 0
    cp_4 = 1 if cp == 4 else 0
    
    restecg_1 = 1 if restecg == 1 else 0
    restecg_2 = 1 if restecg == 2 else 0
    
    thal_6 = 1 if thal == 6 else 0
    thal_7 = 1 if thal == 7 else 0


    data = {
        "age": age,
        "sex": sex,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "cp_2": cp_2,
        "cp_3": cp_3,
        "cp_4": cp_4,
        "restecg_1": restecg_1,
        "restecg_2": restecg_2,
        "thal_6": thal_6,
        "thal_7": thal_7
    }
    return pd.DataFrame(data, index=[0])

user_df = user_input_features()

# -----------------------------
# Main Page
# -----------------------------
st.title("❤️ Heart Disease Prediction App")
st.write("Enter the patient's health information in the sidebar and get a real-time prediction.")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Patient Input")
    st.write(user_df)

with col2:
    st.subheader("Model & Data Status")
    st.write(f"Model loaded: {'Yes' if model is not None else 'No'}")
    st.write(f"Dataset loaded: {'Yes' if not df.empty else 'No'}")

# -----------------------------
# Prediction
# -----------------------------
pred = model.predict(user_df)

st.subheader("📊 Prediction Results")

if pred == 1:
    st.error(f"🚨 High Risk of Heart Disease")
else:
    st.success(f"✅ Low Risk of Heart Disease")



# Offer CSV download of the input + prediction
out_df = user_df.copy()
out_df["prediction"] = pred

csv_buffer = io.StringIO()
out_df.to_csv(csv_buffer, index=False)
st.download_button("Download prediction (CSV)", csv_buffer.getvalue(), file_name="prediction.csv")

# -----------------------------
# Data Visualization
# -----------------------------
st.markdown("---")
st.subheader("Exploratory Visualizations from Dataset")

# Basic info and head
st.markdown("**Dataset preview**")
st.dataframe(df.head())


# Footer
st.markdown("---")
st.caption("App generated with love By omniah (sprint X microsoft project)")
