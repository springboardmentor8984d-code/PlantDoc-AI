import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import joblib
import pandas as pd
import time

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Plant Doctor AI",
    page_icon="🌿",
    layout="wide",
)

# ------------------------------------------------------------
# LOTTIE ANIMATION LOADER
# ------------------------------------------------------------
from streamlit_lottie import st_lottie
import requests

def load_lottie(url):
    return requests.get(url).json()

plant_lottie = load_lottie("https://lottie.host/7cadbf17-ff67-4a49-a3f8-cba5f2b5756a/KCEXL8uZ5Q.json")
scan_lottie = load_lottie("https://lottie.host/e3cf3b1a-f3c4-45e2-84ed-996a96f2efdf/tBi8bxFJmG.json")

# ------------------------------------------------------------
# CSS — INSANE UI LEVEL
# ------------------------------------------------------------
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #e9fdf1 0%, #d9ffe8 100%);
}

/* Floating glowing emoji */
.floating-emoji {
    animation: float 4s ease-in-out infinite;
    font-size: 45px;
    opacity: 0.8;
}
@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-14px); }
    100% { transform: translateY(0px); }
}

/* Glass Header */
.header-card {
    background: rgba(255,255,255,0.32);
    border-radius: 25px;
    padding: 30px;
    backdrop-filter: blur(14px);
    margin-bottom: 30px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.12);
    text-align: center;
}

/* Big title gradient */
.big-title {
    font-size: 40px !important;
    font-weight: 900 !important;
    background: linear-gradient(90deg,#23bd68,#0b8f53,#16d471);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Glow buttons */
.stButton>button {
    background: linear-gradient(90deg,#16c66d,#0a9f57);
    color: white;
    border-radius: 14px;
    padding: 10px 25px;
    font-size: 18px;
    border: none;
    transition: 0.25s;
    box-shadow: 0px 5px 15px rgba(0,128,64,0.25);
}
.stButton>button:hover {
    transform: scale(1.06);
    box-shadow: 0px 8px 22px rgba(0,128,64,0.33);
}

/* Output card - glass effect */
.output-box {
    background: rgba(255,255,255,0.60);
    backdrop-filter: blur(10px);
    padding: 20px;
    border-radius: 20px;
    margin-top: 15px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.10);
    animation: fadeIn 0.6s ease-out;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Subheader */
.sub-header {
    font-size: 22px;
    font-weight: 700;
    color: #0a8f54;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# LOAD MODELS
# ------------------------------------------------------------
IMAGE_MODEL = tf.keras.models.load_model("saved_models/mobilenetv2.h5")
TEXT_MODEL = joblib.load("saved_models/text_model.pkl")
VECTORIZER = joblib.load("saved_models/text_vectorizer.pkl")
df = pd.read_csv("datasets/text1.csv")


# ------------------------------------------------------------
# PREDICT FUNCTIONS
# ------------------------------------------------------------
def predict_image(img):
    img_resized = cv2.resize(img, (224, 224)) / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)
    preds = IMAGE_MODEL.predict(img_resized)
    class_index = np.argmax(preds)

    class_names = sorted(df["disease"].unique())
    disease = class_names[class_index]
    row = df[df["disease"] == disease].iloc[0]

    return row["plant_name"], disease, row["cure"]


def predict_text(plant, symptoms):
    text = plant.lower() + " " + symptoms.lower()
    vec = VECTORIZER.transform([text])
    predicted = TEXT_MODEL.predict(vec)[0]
    row = df[df["disease"] == predicted].iloc[0]

    return plant.title(), predicted, row["cure"]


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.markdown("""
<div class="header-card">
    <div class="floating-emoji">🌿</div>
    <div class="big-title">Plant Doctor AI — Ultra Premium</div>
    <div style="font-size:18px;">Plant health? Sorted. Upload or describe symptoms.</div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# TABS
# ------------------------------------------------------------
tab1, tab2 = st.tabs(["📸 Image Diagnosis", "📝 Text Diagnosis"])


# ------------------------------------------------------------
# IMAGE TAB
# ------------------------------------------------------------
with tab1:
    st.markdown("<div class='sub-header'>Upload a Plant Leaf Image</div>", unsafe_allow_html=True)
    st_lottie(plant_lottie, height=140)

    img_file = st.file_uploader("Upload leaf photo", type=["jpg","jpeg","png"])

    if img_file:
        img_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        st.image(img, width=350)

        if st.button("🔍 Analyze Image"):
            with st.spinner("Scanning leaf…"):
                st_lottie(scan_lottie, height=160)
                time.sleep(1.5)
                plant, disease, cure = predict_image(img)

            st.markdown(f"""
            <div class="output-box">
                <h4>🌿 Plant: <b>{plant}</b></h4>
                <h4>😷 Disease: <b>{disease}</b></h4>
                <p>💊 <b>Treatment:</b> {cure}</p>
            </div>
            """, unsafe_allow_html=True)


# ------------------------------------------------------------
# TEXT TAB
# ------------------------------------------------------------
with tab2:
    st.markdown("<div class='sub-header'>Describe Symptoms</div>", unsafe_allow_html=True)
    st_lottie(scan_lottie, height=150)

    plant = st.text_input("Plant Name")
    symptoms = st.text_area("Describe the symptoms")

    if st.button("🧠 Analyze Text"):
        if not plant.strip() or not symptoms.strip():
            st.error("⚠️ Enter both fields!")
        else:
            with st.spinner("Analyzing…"):
                time.sleep(1.3)
                plant_name, disease, cure = predict_text(plant, symptoms)

            st.markdown(f"""
            <div class="output-box">
                <h4>🌿 Plant: <b>{plant_name}</b></h4>
                <h4>😷 Predicted Disease: <b>{disease}</b></h4>
                <p>💊 <b>Treatment:</b> {cure}</p>
            </div>
            """, unsafe_allow_html=True)
