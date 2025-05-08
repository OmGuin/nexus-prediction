import streamlit as st
from inference import calculate_irscore
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="NexFlow", layout="centered", page_icon="💡")

# Styling
st.markdown(
    """
    <style>
    body {
        background-color: #f9f8fc;
    }
    .main {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 2rem;
    }
    .stButton > button {
        background-color: #a29bfe;
        color: white;
        border-radius: 10px;
        font-size: 16px;
        padding: 0.6em 2em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Simulated login screen
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align: center; color: #2d3436;'>NexFlow Login</h1>", unsafe_allow_html=True)
    st.text_input("Username")
    st.text_input("Password", type="password")
    if st.button("Login"):
        st.session_state.logged_in = True
        st.rerun()
    st.stop()

# Main IRScore page
st.markdown("<h1 style='text-align: center; color: #2d3436;'>NexFlow</h1>", unsafe_allow_html=True)
st.markdown("### Estimate your IRScore (Insulin Resistance Risk)")

# Input fields
st.write("Please enter the following details:")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    weight = st.number_input("Body Weight (lbs)", min_value=150.0, max_value=600.0, step=1.0)

with col2:
    gender_label = st.selectbox("Gender", ["Male", "Female"])
    height = st.number_input("Height (inches)", min_value=70.0, max_value=96.0, step=0.5)

# Gender mapping
gender = 1 if gender_label == "Male" else 0

# BMI calculation
if height > 0:
    bmi = (weight / (height * height)) * 703
else:
    bmi = 0
shap_values = None
if st.button("Calculate IRScore"):
    features = [age, gender, bmi, weight, height]
    score, shap_values = calculate_irscore([features])

    # Determine most important feature
    feature_names = ['Age', 'Gender', 'BMI', 'Body weight', 'Height']
    top_feature_idx = int(np.argmax(np.abs(shap_values)))
    top_feature = feature_names[top_feature_idx]

    st.markdown(f"""
        <div style='text-align: center; padding: 2em;'>
            <h2 style='color: #6c5ce7;'>Your IRScore</h2>
            <div style='font-size: 72px; font-weight: bold; color: #2d3436;'>{int(score)}</div>
            <p style='color: #636e72;'>Your IRScore is <strong>{int(score)}</strong> out of 100.</p>
            <p>This score reflects your overall metabolic health — a higher score indicates better health and a lower risk of insulin resistance.</p>
            <hr style='margin-top:2em;'>
            <p style='color: #6c5ce7;'>🔍 The most influential feature for your score was: <strong>{top_feature}</strong></p>
        </div>
    """, unsafe_allow_html=True)
