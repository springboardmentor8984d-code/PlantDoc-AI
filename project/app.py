import streamlit as st
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from PIL import Image
import numpy as np
import os

st.write("TensorFlow version:", tf.__version__)
try:
    import keras
    st.write("Keras version:", keras.__version__)
except ImportError:
    st.write("Standalone Keras not installed")


# ------------------ Load CNN Model ------------------

MODEL_PATH = "plant_disease_model_mobilenetv2.keras"

@st.cache_resource
def load_cnn_model():
    try:
        if not os.path.exists(MODEL_PATH):
            st.error("❌ Model file not found! Place the .h5 file in the project folder.")
            st.stop()
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load CNN model: {e}")
        st.stop()

cnn_model = load_cnn_model()


def preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img).astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


# ------------------ Load DistilBERT Model ------------------
TEXT_MODEL_PATH = "distilBert_model"

try:
    tokenizer = DistilBertTokenizer.from_pretrained(TEXT_MODEL_PATH)
    text_model = TFDistilBertForSequenceClassification.from_pretrained(TEXT_MODEL_PATH)
except Exception as e:
    st.error(f"❌ Error loading Text Model: {e}")
    st.stop()


def predict_text_class(text):
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=128)
    outputs = text_model(**inputs)
    pred = int(tf.argmax(outputs.logits, axis=1).numpy()[0])
    return pred


# Responses
CLASS_RESPONSES = {
    0: "Apple Scab detected.",
    1: "Powdery Mildew detected.",
    2: "Leaf Blight detected.",
    3: "Plant is Healthy!",
    4: "Rust Infection found.",
    5: "Early Blight detected."
}


# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Plant Health Assistant", layout="wide")

st.markdown("<h1 style='text-align:center;color:#2b8a3e;'>🌿 Plant Health Assistant</h1>", unsafe_allow_html=True)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Image Diagnosis", "Text Diagnosis", "About"])


# ------------------ Image Page ------------------
if page == "Image Diagnosis":
    st.header("📸 Image-based Disease Detection")

    uploaded_image = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        img = Image.open(uploaded_image)

        col1, col2 = st.columns([1.5, 1])   # Left output + Right image

        with col2:
            st.image(img, caption="Uploaded Image", width=260)   # Smaller view

        with col1:
            if st.button("Analyze Image"):
                processed = preprocess_image(img)
                pred_prob = cnn_model.predict(processed)[0]
                pred = int(np.argmax(pred_prob))
                confidence = round(float(np.max(pred_prob) * 100), 2)

                # Handle non-plant/irrelevant images
                if confidence < 40:      # threshold adjustable
                    st.warning("⚠ The uploaded image does not seem to be a plant leaf.\n👉 Please try again with a correct plant image.")
                else:
                    st.success(f"**Result:** {CLASS_RESPONSES.get(pred,'Unknown class')}")
                    st.write(f"🔍 **Confidence:** `{confidence}%`")


# ------------------ Text Page ------------------
elif page == "Text Diagnosis":
    st.header("✍️ Text-based Disease Detection")
    user_text = st.text_area("Describe symptoms...")

    if st.button("Analyze Text") and user_text.strip():
        pred = predict_text_class(user_text)
        st.success(f"**Prediction:** {CLASS_RESPONSES.get(pred,'Unknown')}")


# ------------------ About ------------------
else:
    st.header("ℹ About")
    st.write("""
    This AI tool identifies **plant diseases using image + text inputs** 🌿  
    Models used:  
    - **MobileNetV2 CNN** for leaf image prediction  
    - **DistilBERT** for text-based predictions  
    """)
