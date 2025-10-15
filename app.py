import streamlit as st
import numpy as np
import pandas as pd
import pickle
import warnings
import os

# ==========================================================
# ⚙️ PAGE CONFIGURATION
# ==========================================================
warnings.filterwarnings("ignore", category=UserWarning)
st.set_page_config(page_title="💗 Breast Cancer Classifier", page_icon="💗", layout="wide")

# ==========================================================
# 🎨 CUSTOM CSS STYLING
# ==========================================================
st.markdown("""
    <style>
        /* Global background and font */
        html, body, [class*="css"]  {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
            color: #333333;
        }

        /* Title Animation */
        .title {
            font-size: 2.5em;
            font-weight: 700;
            text-align: center;
            background: linear-gradient(to right, #e91e63, #9c27b0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: glow 3s ease-in-out infinite alternate;
        }
        @keyframes glow {
            from { text-shadow: 0 0 10px #e91e63; }
            to { text-shadow: 0 0 20px #9c27b0; }
        }

        /* Subheader */
        .subheader {
            text-align: center;
            font-size: 1.1em;
            color: #555;
            margin-bottom: 20px;
        }

        /* Prediction Card */
        .result-card {
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0px 5px 25px rgba(0,0,0,0.15);
            text-align: center;
            background-color: white;
            margin-top: 30px;
            transition: transform 0.3s;
        }
        .result-card:hover {
            transform: scale(1.02);
        }

        /* Button styling */
        div.stButton > button:first-child {
            background: linear-gradient(to right, #ff416c, #ff4b2b);
            border: none;
            color: white;
            padding: 0.7em 2em;
            font-size: 1em;
            border-radius: 10px;
            transition: 0.3s;
        }
        div.stButton > button:hover {
            transform: scale(1.05);
            background: linear-gradient(to right, #ff4b2b, #ff416c);
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f5a9bc 0%, #ffd1dc 100%);
            border-right: 2px solid #f8c8dc;
            color: #333333;
        }
        .sidebar .stNumberInput input {
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(255, 105, 135, 0.2);
            transition: 0.3s;
        }
        .sidebar .stNumberInput input:focus {
            box-shadow: 0 0 12px rgba(255, 105, 135, 0.5);
        }

        /* Footer */
        .footer {
            text-align: center;
            color: #666;
            font-size: 0.9em;
            margin-top: 40px;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================================
# 🧠 LOAD MODEL (with safety)
# ==========================================================
MODEL_PATH = "BC_ml_model.pkl"
if os.path.exists(MODEL_PATH):
    model = pickle.load(open(MODEL_PATH, "rb"))
else:
    st.error("❌ Model file not found! Please ensure 'BC_ml_model.pkl' is in the same folder as this app.")
    st.stop()

# ==========================================================
# 🌸 HEADER
# ==========================================================
st.markdown('<div class="title">💗 Breast Cancer Classification App</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">🔬 AI-powered diagnostic tool to predict tumor malignancy using 30 clinical features.</div>', unsafe_allow_html=True)
st.markdown("---")

# ==========================================================
# 🧩 DEFAULT FEATURE VALUES
# ==========================================================
default_values = {
    'radius_mean': 17.99, 'texture_mean': 10.38, 'perimeter_mean': 122.8, 'area_mean': 1001.0,
    'smoothness_mean': 0.1184, 'compactness_mean': 0.2776, 'concavity_mean': 0.3001,
    'concave points_mean': 0.1471, 'symmetry_mean': 0.2419, 'fractal_dimension_mean': 0.0787,
    'radius_se': 1.095, 'texture_se': 0.9053, 'perimeter_se': 8.589, 'area_se': 153.4,
    'smoothness_se': 0.0064, 'compactness_se': 0.0490, 'concavity_se': 0.0537,
    'concave points_se': 0.0159, 'symmetry_se': 0.0300, 'fractal_dimension_se': 0.0062,
    'radius_worst': 25.38, 'texture_worst': 17.33, 'perimeter_worst': 184.6, 'area_worst': 2019.0,
    'smoothness_worst': 0.1622, 'compactness_worst': 0.6656, 'concavity_worst': 0.7119,
    'concave points_worst': 0.2654, 'symmetry_worst': 0.4601, 'fractal_dimension_worst': 0.1189
}

# ==========================================================
# 🧮 SIDEBAR INPUTS
# ==========================================================
st.sidebar.header("🧬 Input Features")
st.sidebar.markdown("Adjust feature values below:")

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
# 🩺 PREDICTION LOGIC
# ==========================================================
if st.button("🔍 Predict Tumor Type"):
    input_df = pd.DataFrame([inputs])
    prediction = model.predict(input_df)[0]
    pred_prob = getattr(model, "predict_proba", lambda x: [[0.5, 0.5]])(input_df)
    confidence = np.max(pred_prob) * 100

    if prediction == "M":
        st.markdown(f"""
            <div class="result-card" style="background:#ffe5e5;">
                <h2>🧬 Prediction: <span style="color:#b30000;">Malignant (Cancerous)</span></h2>
                <p>⚠️ Immediate medical consultation is recommended.</p>
                <p><b>Model Confidence:</b> {confidence:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="result-card" style="background:#e8f5e9;">
                <h2>✅ Prediction: <span style="color:#155724;">Benign (Non-Cancerous)</span></h2>
                <p>💚 No signs of malignancy detected.</p>
                <p><b>Model Confidence:</b> {confidence:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)
else:
    st.info("🧭 Adjust sidebar inputs and click **Predict Tumor Type** to view the result.")

# ==========================================================
# 🌟 FOOTER
# ==========================================================
st.markdown("""
    <div class="footer">
        <hr>
        <p>Developed with ❤️ by <b>Pranav</b> | Powered by <b>Machine Learning & Streamlit</b></p>
    </div>
""", unsafe_allow_html=True)
