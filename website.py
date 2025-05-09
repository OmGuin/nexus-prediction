import streamlit as st
from inference import calculate_irscore
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle
import io

st.set_page_config(page_title="NexFlow", layout="centered", page_icon="💡")


def draw_gauge(score):
    fig, ax = plt.subplots(figsize=(4, 2.4))

    # Set up the base plot
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.axis('off')

    # Draw background arc
    background_arc = Arc((0, 0), 2, 2, theta1=0, theta2=180, linewidth=25, color="#dfe6e9")
    ax.add_patch(background_arc)

    # Draw score arc
    angle = (score / 100) * 180
    score_arc = Arc((0, 0), 2, 2, theta1=0, theta2=angle, linewidth=25, color="#2d3436")
    ax.add_patch(score_arc)

    # Add center text
    ax.text(0, 0.1, f'{int(score)}', ha='center', va='center', fontsize=40, fontweight='bold', color='#2d3436')
    ax.text(0, -0.15, 'IRScore', ha='center', va='center', fontsize=14, color='#2d3436')
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', transparent=True)
    buf.seek(0)
    plt.close(fig)
    st.pyplot(fig)
    return buf





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
    age = st.number_input("Age", min_value=21.0, max_value=120.0, step=1.0)
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
    features = np.array([[age, gender, bmi, weight, height]])
    score = calculate_irscore(features)

    # Determine most important feature
    feature_names = ['Age', 'Gender', 'BMI', 'Body weight', 'Height']
    #top_feature_idx = int(np.argmax(np.abs(shap_values)))
    #top_feature = feature_names[top_feature_idx]
    gauge_img = draw_gauge(score)

    # Fade-in style for the image
    st.markdown("""
        <style>
        .fade-in-img {
            opacity: 0;
            transform: scale(0.95);
            animation: fadeInImage 1s ease-in-out forwards;
        }

        @keyframes fadeInImage {
            to {
                opacity: 1;
                transform: scale(1);
            }
        }
        </style>
    """, unsafe_allow_html=True)

    # Display image in fading div using markdown and st.image
    st.markdown("<div class='fade-in-img'>", unsafe_allow_html=True)
    st.image(gauge_img)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <style>
    .result-block {
        opacity: 0;
        transform: scale(0.95);
        animation: fadeIn 5s forwards;
    }

    @keyframes fadeIn {
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='result-block' style='text-align: center; padding: 1.5em;'>
        <h2 style='color: #6c5ce7;'>Your IRScore</h2>
        <div style='font-size: 48px; font-weight: bold; color: #2d3436;'>{int(score)}</div>
        <p style='color: #636e72;'>Your IRScore is <strong>{int(score)}</strong> out of 100.</p>
        <p>This score reflects your overall metabolic health — a higher score indicates better health and a lower risk of insulin resistance.</p>
        <hr style='margin-top:2em;'>
        <p style='color: #6c5ce7;'>🔍 The most influential feature for your score was: <strong>{"top_feature"}</strong></p>
    </div>
""", unsafe_allow_html=True)
