import streamlit as st
from inference import calculate_irscore  # Make sure this function returns a numeric score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import io
import base64

st.set_page_config(page_title="NexFlow", layout="centered", page_icon="💡")

# ------------------------ Gauge Drawing Function ------------------------ #
def draw_gauge(score):
    fig, ax = plt.subplots(figsize=(4, 2.4))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.axis('off')

    # Background arc
    bg_arc = Arc((0, 0), 2, 2, theta1=0, theta2=180, linewidth=25, color="#dfe6e9")
    ax.add_patch(bg_arc)

    # Score arc
    angle = (score / 100) * 180
    score_arc = Arc((0, 0), 2, 2, theta1=0, theta2=angle, linewidth=25, color="#6c5ce7")
    ax.add_patch(score_arc)

    # Text
    ax.text(0, 0.1, f'{int(score)}', ha='center', va='center', fontsize=40, fontweight='bold', color='#2d3436')
    ax.text(0, -0.15, 'IRScore', ha='center', va='center', fontsize=14, color='#2d3436')

    # Convert to base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ------------------------ Styling ------------------------ #
st.markdown("""
    <style>
    body { background-color: #fdfcfa; }
    .stButton > button {
        background-color: #a29bfe;
        color: white;
        border-radius: 10px;
        font-size: 16px;
        padding: 0.6em 2em;
    }
    .result-block {
        animation: fadeIn 1.5s ease-in-out forwards;
        opacity: 0;
        transform: scale(0.95);
    }
    @keyframes fadeIn {
        to { opacity: 1; transform: scale(1); }
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------ Login State ------------------------ #
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

# ------------------------ Main Page ------------------------ #
st.markdown("<h1 style='text-align: center; color: #2d3436;'>NexFlow</h1>", unsafe_allow_html=True)
st.markdown("### Estimate your IRScore (Insulin Resistance Risk)")

st.write("Please enter the following details:")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=21.0, max_value=120.0, step=1.0)
    weight = st.number_input("Body Weight (lbs)", min_value=150.0, max_value=600.0, step=1.0)

with col2:
    gender_label = st.selectbox("Gender", ["Male", "Female"])
    height = st.number_input("Height (inches)", min_value=70.0, max_value=96.0, step=0.5)

gender = 1 if gender_label == "Male" else 0

# BMI calculation
bmi = (weight / (height * height)) * 703 if height > 0 else 0

# ------------------------ Calculate IRScore ------------------------ #
if st.button("Calculate IRScore"):
    features = np.array([[age, gender, bmi, weight, height]])
    score, contributions = calculate_irscore(features)
    top_feature = max(contributions.items(), key=lambda x: abs(x[1]))[0]


    # Placeholder for top feature (replace with SHAP if needed)
    feature_names = ['Age', 'Gender', 'BMI', 'Body weight', 'Height']

    gauge_base64 = draw_gauge(score)

    st.markdown(f"""
    <div class='result-block' style='text-align: center; padding: 2em 1em;'>
    <h2 style='color: #6c5ce7;'>Your IRScore</h2>
    
    <div style='display: flex; justify-content: center; margin: 2em 0;'>
        <img src="data:image/png;base64,{gauge_base64}" style="max-width: 300px;">
    </div>
        <p style='color: #636e72;'>Your IRScore is <strong>{int(score)}</strong> out of 100.</p>
        <p>This score reflects your overall metabolic health — a higher score indicates better health and a lower risk of insulin resistance.</p>
        <div style='margin-bottom: 1em; font-size: 16px; color: #6c5ce7;'>
        🔍 The most influential feature for your score was: <strong style='color: #6c5ce7;'>{top_feature}</strong>
    </div>
    <div style='height: 30px;'></div>
    </div>
    
    """, unsafe_allow_html=True)