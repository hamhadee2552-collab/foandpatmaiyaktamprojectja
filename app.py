import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# =========================
# LOAD TRAINED AI MODEL
# =========================
model = joblib.load("potato_blight_model.pkl")

st.set_page_config(layout="centered")

# =========================
# 🌸 UI STYLE
# =========================
st.markdown("""
<style>

/* remove top spacing */
header {visibility:hidden;}
[data-testid="stAppViewContainer"] {padding-top:0;}
section.main > div {padding-top:0;}

/* background */
.stApp{
background:linear-gradient(#ffe6f2,#ffd6e7);
}

/* container */
.block-container{
background:rgba(255,255,255,0.65);
border-radius:25px;
padding:30px;
margin-top:-40px;
}

/* button */
.stButton button{
background:#ff80ab;
color:white;
border-radius:15px;
font-size:18px;
}

/* title card */
.title-card{
background:linear-gradient(135deg,#ffcce6,#ffe6f2);
padding:20px;
border-radius:20px;
text-align:center;
margin-bottom:20px;
animation:fadeGlow 1.2s ease;
box-shadow:0 0 20px rgba(255,105,180,0.3);
}

@keyframes fadeGlow{
from{opacity:0; transform:translateY(-10px);}
to{opacity:1; transform:translateY(0);}
}

.title-text{
font-size:42px;
color:#ff4da6;
text-shadow:0 0 8px rgba(255,105,180,0.5);
}

</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.markdown("""
<div class="title-card">
<div class="title-text">
🌸 Potato Blight AI Helper
</div>
</div>
""", unsafe_allow_html=True)

# =========================
# INPUT SLIDERS
# =========================
temp = st.slider("Temperature °C", 10, 40, 22)
humidity = st.slider("Humidity %", 20, 100, 80)
rain = st.slider("Rain mm", 0, 30, 5)
leafwet = st.slider("Leaf wet hours", 0, 12, 4)
age = st.slider("Plant age", 1, 120, 40)

predict = st.button("💗 Analyze Risk")

# =========================
# SMART RECOMMENDATION LOGIC
# =========================
def smart_suggestions():

    tips=[]

    # temperature
    if temp > 28:
        tips.append("Lower temperature → target ~22–26°C")
    elif temp < 18:
        tips.append("Increase temperature → target ~22–26°C")

    # humidity
    if humidity > 85:
        tips.append("Reduce humidity below ~80%")
    elif humidity < 50:
        tips.append("Increase humidity to ~60–80%")

    # rainfall
    if rain > 15:
        tips.append("Reduce watering to avoid excess moisture")
    elif rain < 3:
        tips.append("Increase watering slightly")

    # leaf wetness
    if leafwet > 6:
        tips.append("Improve airflow to dry leaves faster")
    elif leafwet < 1:
        tips.append("Maintain moderate moisture on foliage")

    if not tips:
        tips.append("Environment looks optimal!")

    return tips

# =========================
# PREDICT + DISPLAY
# =========================
if predict:

    # AI prediction input
    df = pd.DataFrame([{
        "Temp_C": temp,
        "Humidity_%": humidity,
        "Rain_mm": rain,
        "LeafWet_hours": leafwet,
        "PlantAge_days": age
    }])

    probs = model.predict_proba(df)[0]

    risk_index = np.argmax(probs)

    risk_names = ["Low Risk", "Medium Risk", "High Risk"]
    icons = ["🟢", "🟡", "🔴"]

    st.success(f"{icons[risk_index]} {risk_names[risk_index]}")

    # =========================
    # GRAPH — SIMPLE AI PROBABILITY
    # =========================
    labels = ["Low", "Medium", "High"]

    fig, ax = plt.subplots()

    bars = ax.bar(
        labels,
        probs,
        color=["#ffc0cb", "#ff99cc", "#ff4da6"]
    )

    ax.set_ylim(0, 1)
    ax.set_facecolor("#fff0f6")

    for bar in bars:
        ax.text(
            bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.02,
            "💗",
            ha="center",
            fontsize=16
        )

    ax.spines[:].set_visible(False)
    ax.tick_params(left=False, bottom=False)

    st.pyplot(fig)

    # =========================
    # RECOMMENDATIONS
    # =========================
    st.subheader("🌿 Smart Suggestions")

    for tip in smart_suggestions():
        st.write("•", tip)

    # =========================
    # DISEASE FORECAST
    # =========================
    base_days = 7
    timeline = [base_days+7, base_days, base_days-3]

    st.subheader("⏳ Disease Projection")

    st.write(
        f"If conditions stay the same, disease may develop in **~{timeline[risk_index]} days**."
    )
