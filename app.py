import streamlit as st
import numpy as np
import pickle

# ==========================================================
# 📘 LOAD MODEL
# ==========================================================
model = pickle.load(open("BC_ml_model.pkl", "rb"))

# ==========================================================
# 🧠 PAGE SETUP
# ==========================================================
st.set_page_config(page_title="Breast Cancer Classifier", page_icon="🩺", layout="wide")
st.title("🩺 Breast Cancer Classification App")
st.write("This app predicts whether the tumor is **Malignant (M)** or **Benign (B)** based on medical measurements.")

# ==========================================================
# 📊 FEATURE INPUTS WITH DEFAULT VALUES
# ==========================================================
default_values = {
    'radius_mean': 17.99,
    'texture_mean': 10.38,
    'perimeter_mean': 122.8,
    'area_mean': 1001.0,
    'smoothness_mean': 0.1184,
    'compactness_mean': 0.2776,
    'concavity_mean': 0.3001,
    'concave points_mean': 0.1471,
    'symmetry_mean': 0.2419,
    'fractal_dimension_mean': 0.0787,
    'radius_se': 1.095,
    'texture_se': 0.9053,
    'perimeter_se': 8.589,
    'area_se': 153.4,
    'smoothness_se': 0.0064,
    'compactness_se': 0.0490,
    'concavity_se': 0.0537,
    'concave points_se': 0.0159,
    'symmetry_se': 0.0300,
    'fractal_dimension_se': 0.0062,
    'radius_worst': 25.38,
    'texture_worst': 17.33,
    'perimeter_worst': 184.6,
    'area_worst': 2019.0,
    'smoothness_worst': 0.1622,
    'compactness_worst': 0.6656,
    'concavity_worst': 0.7119,
    'concave points_worst': 0.2654,
    'symmetry_worst': 0.4601,
    'fractal_dimension_worst': 0.1189
}

st.sidebar.header("🔧 Input Features")

inputs = []
for feature, default in default_values.items():
    val = st.sidebar.number_input(f"{feature}", value=float(default), step=0.01, format="%.5f")
    inputs.append(val)

# ==========================================================
# 🎯 PREDICTION
# ==========================================================
if st.button("Predict"):
    features = np.array(inputs).reshape(1, -1)
    prediction = model.predict(features)[0]
    result = "🧬 Malignant (Cancerous)" if prediction == "M" else "✅ Benign (Non-Cancerous)"

    st.subheader("Prediction Result:")
    st.success(result)
else:
    st.info("⬅️ Enter values on the left sidebar and click **Predict** to see the result.")

# ==========================================================
# 🩻 FOOTER
# ==========================================================
st.markdown("---")
st.caption("Developed by Pranav | AI-Powered Breast Cancer Detection")
