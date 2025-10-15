import streamlit as st
import numpy as np
import pandas as pd
import pickle
import warnings

# ==========================================================
# ⚙️ CONFIGURATION
# ==========================================================
warnings.filterwarnings("ignore", category=UserWarning)
st.set_page_config(page_title="🩺 Breast Cancer Classifier", page_icon="💗", layout="wide")

# ==========================================================
# 🧠 LOAD MODEL
# ==========================================================
model = pickle.load(open("breast_cancer_model.pkl", "rb"))

# ==========================================================
# 🌸 PAGE TITLE & DESCRIPTION
# ==========================================================
st.title("💗 Breast Cancer Classification App")
st.markdown(
    """
    ### 🔬 About this App
    This AI-powered app predicts whether a breast tumor is **Malignant (Cancerous)** or **Benign (Non-Cancerous)**  
    based on 30 key medical features from a fine needle aspiration test.
    """
)

st.markdown("---")

# ==========================================================
# 🎯 DEFAULT VALUES (mean sample from dataset)
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

# ==========================================================
# 🧩 INPUT SECTION
# ==========================================================
st.sidebar.header("🧮 Input Features")

st.sidebar.markdown("Enter or adjust values below:")

inputs = {}

with st.sidebar.expander("📊 Mean Features"):
    for feature in list(default_values.keys())[:10]:
        inputs[feature] = st.number_input(feature, value=float(default_values[feature]), step=0.01, format="%.5f")

with st.sidebar.expander("📈 SE (Error) Features"):
    for feature in list(default_values.keys())[10:20]:
        inputs[feature] = st.number_input(feature, value=float(default_values[feature]), step=0.01, format="%.5f")

with st.sidebar.expander("🔍 Worst Features"):
    for feature in list(default_values.keys())[20:]:
        inputs[feature] = st.number_input(feature, value=float(default_values[feature]), step=0.01, format="%.5f")

# ==========================================================
# 🧾 PREDICTION
# ==========================================================
if st.button("🔍 Predict Tumor Type"):
    # Convert to DataFrame (fixes sklearn warning)
    input_df = pd.DataFrame([inputs])
    prediction = model.predict(input_df)[0]

    # 🎨 Stylish Result
    if prediction == "M":
        st.markdown(
            """
            <div style="background-color:#ffcccc;padding:20px;border-radius:10px;text-align:center;">
            <h2>🧬 Prediction: <span style="color:#b30000;">Malignant (Cancerous)</span></h2>
            <p>⚠️ Immediate medical consultation is recommended.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background-color:#d4edda;padding:20px;border-radius:10px;text-align:center;">
            <h2>✅ Prediction: <span style="color:#155724;">Benign (Non-Cancerous)</span></h2>
            <p>💚 No signs of malignancy detected.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

else:
    st.info("🧭 Adjust the input features in the sidebar and click **Predict Tumor Type** to see the result.")

# ==========================================================
# 🌟 FOOTER
# ==========================================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center;">
        <p>Developed by <b>Pranav</b> 🧠 | Powered by <b>Machine Learning</b> 🤖</p>
    </div>
    """,
    unsafe_allow_html=True,
)
