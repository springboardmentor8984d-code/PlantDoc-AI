import streamlit as st
import tensorflow as tf
import torch
import pickle
import json
from PIL import Image
import numpy as np
import os

# Import our custom modules
from plantbot.chatbot_manager import ChatbotManager
from utils.treatments import get_treatment

st.set_page_config(
    page_title="PlantDoc Chatbot",
from chatbot.chatbot_manager import ChatbotManager
from knowledge_base.treatments import get_treatment, format_treatment_message

st.set_page_config(
    page_title="🌿 Plant Disease Chatbot",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for chat interface
st.markdown("""
<style>
    .user-message {
        background-color: #007bff;
        color: white;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 5px 0;
        max-width: 70%;
        margin-left: auto;
        text-align: right;
    }
    .bot-message {
        background-color: #f1f3f4;
        color: #202124;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 5px 0;
        max-width: 70%;
        margin-right: auto;
    }
    .chat-container {
        height: 400px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Helper: clean CNN raw label -> human label
# ----------------------------
def clean_cnn_label(raw):
    name = raw.replace("PlantDoc_", "").replace("PlantVillage_", "")
    # Fix: Removed erroneous replace line that was spacing out characters
    name = name.replace("_leaf","").replace("_Leaf","").replace("_leaves","")
    name = name.replace("_"," ").strip()
    name = " ".join(w.capitalize() for w in name.split())
    return name

# ----------------------------
# Load resources (cached)
# ----------------------------
@st.cache_resource
def load_cnn():
    """Load CNN model for image classification."""
    # Use TensorFlow SavedModel loader (Keras 3 compatible)
    cnn_model = tf.saved_model.load("models/final_saved_model")
    
    # Load mapping dict {class_name: idx}
    with open("resources/class_names.pkl", "rb") as f:
        class_names = pickle.load(f)
    
    idx_to_raw = {v: k for k, v in class_names.items()}
    idx_to_label = {idx: clean_cnn_label(raw) for idx, raw in idx_to_raw.items()}
    
    return cnn_model, idx_to_label


@st.cache_resource
def load_chatbot(device_str="cpu"):
    """Load chatbot manager with BERT model."""
    chatbot = ChatbotManager(
        model_path="models/bert_plant_chatbot_model.pt",
        label_mapping_path="resources/label_mapping.json",
        device=device_str
    )
    return chatbot


# Load models
cnn_model, cnn_idx_to_label = load_cnn()
device = "cuda" if torch.cuda.is_available() else "cpu"
chatbot = load_chatbot(device)

# ----------------------------
# Prediction helpers
# ----------------------------
def preprocess_image_file(uploaded_file):
    """Preprocess uploaded image for CNN."""
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0), img


def predict_cnn(img_array):
    """Predict disease using CNN model."""
    # Use SavedModel serve signature
    infer = cnn_model.signatures["serving_default"]
    
    # Get input tensor name
    input_name = list(infer.structured_input_signature[1].keys())[0]
    
    # Make prediction
    predictions = infer(tf.constant(img_array, dtype=tf.float32))
    
    # Get output
    output_key = list(predictions.keys())[0]
    probs = predictions[output_key].numpy()[0]
    
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    label = cnn_idx_to_label.get(idx, "Unknown")
    return label, conf


# ----------------------------
# Initialize session state
# ----------------------------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    # Add initial greeting
    greeting = "Hello! 🌿 I'm your plant disease assistant. Describe the symptoms you're seeing on your plant, and I'll help diagnose the issue."
    st.session_state.chat_history.append({"role": "bot", "content": greeting})

if 'chatbot_initialized' not in st.session_state:
    st.session_state.chatbot_initialized = False

# ----------------------------
# UI
# ----------------------------
st.title("🌿 Plant Disease Chatbot")
st.write("Upload a leaf image, describe symptoms, or chat with our AI assistant for diagnosis and treatment recommendations.")

# Create tabs
# Create tabs
tab1, tab2 = st.tabs(["📸 Image Diagnosis", "🤖 Chat Diagnosis"])

# ----------------------------
# Tab 1: Image Diagnosis
# ----------------------------
with tab1:
    st.subheader("Upload a Leaf Image")
    uploaded = st.file_uploader("Choose an image (jpg/png)", type=["jpg","jpeg","png"], key="image_upload")
    
    if uploaded:
        img_array, img = preprocess_image_file(uploaded)
        st.image(img, caption="Uploaded image", use_container_width=True)
        
        with st.spinner("🔍 Analyzing image..."):
            disease, conf = predict_cnn(img_array)
        
        st.success(f"**Predicted disease:** {disease}")
        st.info(f"**Confidence:** {conf*100:.2f}%")
        
        # Get treatment
        treatment_info = get_treatment(disease)
        
        st.subheader("📋 Treatment Recommendations")
        st.write(treatment_info['treatment'])
        
        with st.expander("🛡️ Prevention Tips"):
            st.write(treatment_info['prevention'])



# ----------------------------
# Tab 3: Chat Diagnosis (Conversational)
# ----------------------------
# ----------------------------
# Tab 2: Chat Diagnosis (Conversational)
# ----------------------------
with tab2:
    st.subheader("💬 Chat with AI Assistant")
    st.write("Have a conversation with our AI to diagnose plant diseases. The bot will ask clarifying questions to provide accurate diagnosis.")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-message">🤖 {message["content"]}</div>', unsafe_allow_html=True)
    
    # User input
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "Your message:",
            placeholder="Describe your plant's symptoms...",
            key="chat_input",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("Send", key="send_chat", use_container_width=True)
    
    # Clear conversation button
    if st.button("🔄 Clear Conversation", key="clear_chat"):
        chatbot.reset_conversation()
        st.session_state.chat_history = []
        greeting = "Hello! 🌿 I'm your plant disease assistant. Describe the symptoms you're seeing on your plant, and I'll help diagnose the issue."
        st.session_state.chat_history.append({"role": "bot", "content": greeting})
        st.rerun()
    
    # Handle send
    if send_button and user_input.strip():
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Get bot response
        with st.spinner("🤔 Thinking..."):
            bot_response = chatbot.generate_response(user_input)
        
        # Add bot response to history
        st.session_state.chat_history.append({"role": "bot", "content": bot_response})
        
        # Rerun to update chat display
        st.rerun()

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using CNN (image) + BERT (text/chat) models</p>
    <p><small>⚠️ This tool provides general guidance. For serious plant health issues, consult a professional.</small></p>
</div>
""", unsafe_allow_html=True)
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
import streamlit as st
import tensorflow as tf
import torch
import pickle
import json
from PIL import Image
import numpy as np
import os

# Import our custom modules
from chatbot.chatbot_manager import ChatbotManager
from knowledge_base.treatments import get_treatment, format_treatment_message

st.set_page_config(
    page_title="🌿 Plant Disease Chatbot",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for chat interface
st.markdown("""
<style>
    .user-message {
        background-color: #007bff;
        color: white;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 5px 0;
        max-width: 70%;
        margin-left: auto;
        text-align: right;
    }
    .bot-message {
        background-color: #f1f3f4;
        color: #202124;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 5px 0;
        max-width: 70%;
        margin-right: auto;
    }
    .chat-container {
        height: 400px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Helper: clean CNN raw label -> human label
# ----------------------------
def clean_cnn_label(raw):
    name = raw.replace("PlantDoc_", "").replace("PlantVillage_", "")
    # Fix: Removed erroneous replace line that was spacing out characters
    name = name.replace("_leaf","").replace("_Leaf","").replace("_leaves","")
    name = name.replace("_"," ").strip()
    name = " ".join(w.capitalize() for w in name.split())
    return name

# ----------------------------
# Load resources (cached)
# ----------------------------
@st.cache_resource
def load_cnn():
    """Load CNN model for image classification."""
    # Use TensorFlow SavedModel loader (Keras 3 compatible)
    cnn_model = tf.saved_model.load("models/final_saved_model")
    
    # Load mapping dict {class_name: idx}
    with open("resources/class_names.pkl", "rb") as f:
        class_names = pickle.load(f)
    
    idx_to_raw = {v: k for k, v in class_names.items()}
    idx_to_label = {idx: clean_cnn_label(raw) for idx, raw in idx_to_raw.items()}
    
    return cnn_model, idx_to_label


@st.cache_resource
def load_chatbot(device_str="cpu"):
    """Load chatbot manager with BERT model."""
    chatbot = ChatbotManager(
        model_path="models/bert_plant_chatbot_model.pt",
        label_mapping_path="resources/label_mapping.json",
        device=device_str
    )
    return chatbot


# Load models
cnn_model, cnn_idx_to_label = load_cnn()
device = "cuda" if torch.cuda.is_available() else "cpu"
chatbot = load_chatbot(device)

# ----------------------------
# Prediction helpers
# ----------------------------
def preprocess_image_file(uploaded_file):
    """Preprocess uploaded image for CNN."""
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0), img


def predict_cnn(img_array):
    """Predict disease using CNN model."""
    # Use SavedModel serve signature
    infer = cnn_model.signatures["serving_default"]
    
    # Get input tensor name
    input_name = list(infer.structured_input_signature[1].keys())[0]
    
    # Make prediction
    predictions = infer(tf.constant(img_array, dtype=tf.float32))
    
    # Get output
    output_key = list(predictions.keys())[0]
    probs = predictions[output_key].numpy()[0]
    
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    label = cnn_idx_to_label.get(idx, "Unknown")
    return label, conf


# ----------------------------
# Initialize session state
# ----------------------------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    # Add initial greeting
    greeting = "Hello! 🌿 I'm your plant disease assistant. Describe the symptoms you're seeing on your plant, and I'll help diagnose the issue."
    st.session_state.chat_history.append({"role": "bot", "content": greeting})

if 'chatbot_initialized' not in st.session_state:
    st.session_state.chatbot_initialized = False

# ----------------------------
# UI
# ----------------------------
st.title("🌿 Plant Disease Chatbot")
st.write("Upload a leaf image, describe symptoms, or chat with our AI assistant for diagnosis and treatment recommendations.")

# Create tabs
# Create tabs
tab1, tab2 = st.tabs(["📸 Image Diagnosis", "🤖 Chat Diagnosis"])

# ----------------------------
# Tab 1: Image Diagnosis
# ----------------------------
with tab1:
    st.subheader("Upload a Leaf Image")
    uploaded = st.file_uploader("Choose an image (jpg/png)", type=["jpg","jpeg","png"], key="image_upload")
    
    if uploaded:
        img_array, img = preprocess_image_file(uploaded)
        st.image(img, caption="Uploaded image", use_container_width=True)
        
        with st.spinner("🔍 Analyzing image..."):
            disease, conf = predict_cnn(img_array)
        
        st.success(f"**Predicted disease:** {disease}")
        st.info(f"**Confidence:** {conf*100:.2f}%")
        
        # Get treatment
        treatment_info = get_treatment(disease)
        
        st.subheader("📋 Treatment Recommendations")
        st.write(treatment_info['treatment'])
        
        with st.expander("🛡️ Prevention Tips"):
            st.write(treatment_info['prevention'])



# ----------------------------
# Tab 3: Chat Diagnosis (Conversational)
# ----------------------------
# ----------------------------
# Tab 2: Chat Diagnosis (Conversational)
# ----------------------------
with tab2:
    st.subheader("💬 Chat with AI Assistant")
    st.write("Have a conversation with our AI to diagnose plant diseases. The bot will ask clarifying questions to provide accurate diagnosis.")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-message">🤖 {message["content"]}</div>', unsafe_allow_html=True)
    
    # User input
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "Your message:",
            placeholder="Describe your plant's symptoms...",
            key="chat_input",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("Send", key="send_chat", use_container_width=True)
    
    # Clear conversation button
    if st.button("🔄 Clear Conversation", key="clear_chat"):
        chatbot.reset_conversation()
        st.session_state.chat_history = []
        greeting = "Hello! 🌿 I'm your plant disease assistant. Describe the symptoms you're seeing on your plant, and I'll help diagnose the issue."
        st.session_state.chat_history.append({"role": "bot", "content": greeting})
        st.rerun()
    
    # Handle send
    if send_button and user_input.strip():
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Get bot response
        with st.spinner("🤔 Thinking..."):
            bot_response = chatbot.generate_response(user_input)
        
        # Add bot response to history
        st.session_state.chat_history.append({"role": "bot", "content": bot_response})
        
        # Rerun to update chat display
        st.rerun()

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using CNN (image) + BERT (text/chat) models</p>
    <p><small>⚠️ This tool provides general guidance. For serious plant health issues, consult a professional.</small></p>
</div>
""", unsafe_allow_html=True)
