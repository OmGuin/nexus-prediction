import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import io
import base64
from inference import calculate_irscore
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="NexFlow", layout="centered", page_icon="💡")

# ------------------------ Gauge Drawing Function ------------------------ #
def draw_gauge(score):
    fig, ax = plt.subplots(figsize=(4, 2.4))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.axis('off')

    bg_arc = Arc((0, 0), 2, 2, theta1=0, theta2=180, linewidth=25, color="#dfe6e9")
    ax.add_patch(bg_arc)

    angle = (score / 100) * 180
    score_arc = Arc((0, 0), 2, 2, theta1=0, theta2=angle, linewidth=25, color="#6c5ce7")
    ax.add_patch(score_arc)

    ax.text(0, 0.1, f'{int(score)}', ha='center', va='center', fontsize=40, fontweight='bold', color='#2d3436')
    ax.text(0, -0.15, 'IRScore', ha='center', va='center', fontsize=14, color='#2d3436')

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ------------------------ Styling ------------------------ #
css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:2rem;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)
st.markdown("""
<style>
body {
    background-color: #fdfcfa;
}

/* Tabs: Increase size + space them out */
[data-baseweb="tab-list"] {
    justify-content: space-evenly !important;
}

[data-baseweb="tab"] {
    font-size: 2rem !important;
    padding: 1.2rem 2rem !important;
    flex: 1 1 auto !important;
    text-align: center;
}

/* Button Styling */
.stButton > button {
    background-color: #a29bfe;
    color: white;
    border-radius: 10px;
    font-size: 16px;
    padding: 0.6em 2em;
}

/* Fade in gauge block */
.result-block {
    animation: fadeIn 1.5s ease-in-out forwards;
    opacity: 0;
    transform: scale(0.95);
}

.fade-in-gauge {
    animation: fadeIn 1.5s ease-in-out forwards;
}

@keyframes fadeIn {
    from {opacity: 0; transform: scale(0.75);}
    to { opacity: 1; transform: scale(1.1); }
}
</style>
""", unsafe_allow_html=True)



# ------------------------ Login ------------------------ #
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align: center; color: #2d3436;'>NexFlow Login</h1>", unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        st.session_state.logged_in = True
        st.rerun()
    st.stop()

# ------------------------ Tabs ------------------------ #
tab1, tab2, tab3 = st.tabs(["Home", "Trajectory", "Recommendations"])

# ------------------------ Home Tab ------------------------ #
with tab1:
    st.markdown("<h1 style='text-align: center; color: #2d3436;'>NexFlow</h1>", unsafe_allow_html=True)
    st.markdown("### Estimate your IRScore (Insulin Resistance Risk)")
    st.write("Please enter the following details:")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=21.0, max_value=120.0, step=1.0)
        weight = st.number_input("Body Weight (lbs)", min_value=150.0, max_value=600.0, step=1.0)
    with col2:
        gender_label = st.selectbox("Gender", ["Male", "Female"])
        height = st.number_input("Height (inches)", min_value=65.0, max_value=96.0, step=1.0)

    gender = 1 if gender_label == "Male" else 0
    bmi = (weight / (height * height)) * 703 if height > 0 else 0

    if st.button("Calculate IRScore"):
        features = np.array([[age, gender, bmi, weight, height]])
        score, contributions = calculate_irscore(features)
        top_feature = max(contributions.items(), key=lambda x: abs(x[1]))[0]

        gauge_base64 = draw_gauge(score)

        st.markdown(f"""
        <div class='result-block' style='text-align: center; padding: 2em 1em;'>
            <h2 style='color: #6c5ce7;'>Your IRScore</h2>
            <div class='fade-in-gauge' style='display: flex; justify-content: center; margin: 2em 0;'>
                <img src="data:image/png;base64,{gauge_base64}" style="max-width: 300px;">
            </div>
            <p style='color: #636e72;'>Your IRScore is <strong>{int(score)}</strong> out of 100.</p>
            <p>This score reflects your overall metabolic health — a higher score indicates better health and a lower risk of insulin resistance.</p>
            <div style='margin-bottom: 1em; font-size: 16px; color: #6c5ce7;'>
                🔍 The most influential feature for your score was: <strong style='color: #6c5ce7;'>{top_feature}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
<div class='fade-in-gauge' style='display: flex; justify-content: space-around; margin-top: 2em;'>
    <div style='text-align: center;'>
        <p style='font-weight: bold;'>IRScore</p>
        <p style='font-size: 24px;'>{int(score)}</p>
        <p style='color: green;'>+2 from last week</p>
    </div>
    <div style='text-align: center;'>
        <p style='font-weight: bold;'>BMI</p>
        <p style='font-size: 24px;'>{bmi:.1f}</p>
        <p style='color: green;'>+0.2 from last week</p>
    </div>
    <div style='text-align: center;'>
        <p style='font-weight: bold;'>Weight</p>
        <p style='font-size: 24px;'>{weight}</p>
        <p style='color: green;'>+0.9 from last week</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ------------------------ Trajectory Tab ------------------------ #
with tab2:
    st.markdown("<h1 style='text-align: center; color: #2d3436;'>Trajectory Panel</h1>", unsafe_allow_html=True)

    if "exercise_on" not in st.session_state:
        st.session_state.exercise_on = False

    with st.container():
        c1, c2, c3 = st.columns(3)
        c1.toggle("Carbs", key="carbs_toggle")
        c2.toggle("Sleep", key="sleep_toggle")
        c3.toggle("Exercise", key="exercise_toggle")
    features = np.array([[age, gender, bmi, weight, height]])
    score, _ = calculate_irscore(features)

    score = int(score)
    if st.session_state.get("exercise_toggle", False):
        days = ["Today", "1w", "2w", "3w", "4w"]
        scores = [score, score+1, score+3, score+4, score+5]
    else:
        days = ["Today", "1w", "2w", "3w", "4w"]
        scores = [score, score-1, score-3, score-6, score-7]

    df = pd.DataFrame({'Week': days, 'Score': scores})
    fig = px.line(df, x='Week', y='Score', title='Trajectory', markers=True)
    st.plotly_chart(fig)

    #fig, ax = plt.subplots()
    #ax.plot(days, scores, color="#2980b9", linewidth=3)
    #ax.set_ylim(60, 100)
    #st.pyplot(fig)

# ------------------------ Recommendations Tab ------------------------ #
with tab3:
    st.markdown("<h1 style='text-align: center; color: #2d3436;'>Recommendations</h1>", unsafe_allow_html=True)
    st.write("Personalized pathways to lower your IRScore.")

    recs = [
        ("Sleep", "Increase duration by 1 hour", -4, "🌙"),
        ("Diet", "Eat more whole grains", -3, "🍽️"),
        ("Exercise", "Walk for 30 minutes/day", -5, "🏋️")
    ]

    for title, desc, delta, icon in recs:
        st.markdown(f"""
        <div style='background-color:#f4f3f8; padding: 1em; border-radius: 10px; margin-bottom: 1em;'>
            <h4 style='color: #2d3436;'>{icon} {title}</h4>
            <p style='margin: 0;'>{desc}</p>
        </div>
        """, unsafe_allow_html=True)
