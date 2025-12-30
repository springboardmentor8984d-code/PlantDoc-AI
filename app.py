


# app.py (Replicating the Screenshot UI)
import os
import io
import json
import glob
import re
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from difflib import get_close_matches

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="PlantDocBot - Plant Health Assistant", page_icon="🌿", layout="wide") 

# Inject custom CSS for the desired button color/style
# Streamlit has limited native color control, so this uses CSS.
st.markdown("""
<style>


            

 
    
        .treatment-card {
            padding: 12px 18px;
            border-radius: 14px;
            background: rgba(255,255,255,0.15);
            backdrop-filter: blur(10px);
            margin-top: 10px;
            border-left: 4px solid #4CAF50;
            font-size: 15px;
            color: #ffffff;
        }
   









/* Style for the main analyze button */
.stButton>button {
    background-color: #388e3c; /* Darker green */
    color: white;
    font-weight: bold;
    border-radius: 8px;
    padding: 10px 40px;
    border: none;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: background-color 0.3s ease;
}
.stButton>button:hover {
    background-color: #4CAF50; /* Lighter green on hover */
}

/* Style for the card containers (like in the screenshots) */
div[data-testid="stVerticalBlock"] > div[style*="border-radius: 10px"] {
    border: 1px solid #e0e0e0;
    padding: 20px;
    border-radius: 10px;
}

/* Center main header */
div[data-testid="stHorizontalBlock"] h1 {
    text-align: center;
}

/* Custom H2 color for "Expert Care" */
.expert-care {
    color: #388e3c;
    font-size: 3.5rem;
    font-weight: 700;
}
.ai-powered {
    color: #6a6a6a;
    font-size: 1rem;
    padding: 8px 15px;
    background-color: #e8f5e9;
    border-radius: 20px;
    display: inline-block;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)


# -------------------------
# Config (NO CHANGE TO LOGIC/PATHS)
# -------------------------
CNN_MODEL_PATH = "models/mobilenetv2_final.h5"
BERT_MODEL_DIR = "models/final_distilbert_model"
CNN_CLASSES_FILE = "models/cnn_classes.json"
CNN_TREATMENT_FILE = "disease_treatment_map.json"
BERT_TREATMENT_FILE = "disease_treatment_bert.json"
BERT_CLASSES_FILE = "models/final_distilbert_model/bert_classes.json"
IMAGE_TARGET_SIZE = (224, 224)
NUM_CNN_CLASSES = 15
RESAMPLE = Image.LANCZOS

# -------------------------
# Utilities (NO CHANGE TO LOGIC)
# -------------------------
def load_json_safe(path):
    # ... (function body remains the same)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def find_classes_in_dir(search_dir, expected_count=NUM_CNN_CLASSES):
    # ... (function body remains the same)
    if not os.path.isdir(search_dir):
        return None, []
    tried = []
    candidates = [
        "class_names.json", "classes.json", "labels.json",
        "class_names.txt", "classes.txt", "my_classes.json", "classes_nmes.json"
    ]
    for name in candidates:
        p = os.path.join(search_dir, name)
        if os.path.exists(p):
            tried.append(p)
            data = load_json_safe(p)
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                return data, tried
            try:
                with open(p, "r", encoding="utf-8") as f:
                    lines = [ln.strip() for ln in f if ln.strip()]
                    if lines:
                        return lines, tried
            except Exception:
                pass
    for p in glob.glob(os.path.join(search_dir, "*.json")):
        tried.append(p)
        data = load_json_safe(p)
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            if len(data) == expected_count:
                return data, tried
            return data, tried
    for p in glob.glob(os.path.join(search_dir, "*.txt")):
        tried.append(p)
        try:
            with open(p, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
                if lines:
                    if len(lines) == expected_count:
                        return lines, tried
                    return lines, tried
        except Exception:
            continue
    return None, tried

def normalize_key(s):
    # ... (function body remains the same)
    if not isinstance(s, str):
        return s
    s2 = s.strip().lower()
    s2 = s2.replace("___", "_")
    s2 = s2.replace("-", " ")
    s2 = "_".join(s2.split())
    s2 = re.sub(r"[^a-z0-9_]", "", s2)
    return s2

# -------------------------
# Load treatment maps + class lists (NO CHANGE TO LOGIC)
# -------------------------
treatment_map_cnn = load_json_safe(CNN_TREATMENT_FILE) or {}
treatment_map_bert = load_json_safe(BERT_TREATMENT_FILE) or {}
bert_classes = load_json_safe(BERT_CLASSES_FILE)
if not isinstance(bert_classes, list):
    bert_classes = None
classes_list = load_json_safe(CNN_CLASSES_FILE)

classes_search_tried = []
if isinstance(classes_list, list) and len(classes_list) == NUM_CNN_CLASSES:
    classes_search_tried.append(CNN_CLASSES_FILE)
else:
    classes_list, classes_search_tried = find_classes_in_dir(
        os.path.dirname(os.path.abspath(CNN_TREATMENT_FILE)) if os.path.exists(CNN_TREATMENT_FILE) else "models",
        NUM_CNN_CLASSES
    )
    if classes_list is None:
        classes_list = [f"class_{i}" for i in range(NUM_CNN_CLASSES)]
        st.warning("Could not auto-detect CNN class names; using placeholders class_0..class_14.")

# -------------------------
# Load models (cached) (NO CHANGE TO LOGIC)
# -------------------------
@st.cache_resource
def load_cnn():
    if not os.path.exists(CNN_MODEL_PATH):
        raise FileNotFoundError(f"CNN model not found at {CNN_MODEL_PATH}")
    return tf.keras.models.load_model(CNN_MODEL_PATH)

@st.cache_resource
def load_bert():
    if not os.path.isdir(BERT_MODEL_DIR):
        return None, None, torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
    model.to(device)
    model.eval()
    return model, tokenizer, device

load_errors = []
cnn_model = None
bert_model = None
bert_tokenizer = None
bert_device = torch.device("cpu")

try:
    cnn_model = load_cnn()
except Exception as e:
    load_errors.append(f"CNN load error: {e}")

try:
    bert_model, bert_tokenizer, bert_device = load_bert()
except Exception as e:
    load_errors.append(f"BERT load error: {e}")

# -------------------------
# Build normalized treatment maps for robust lookup (NO CHANGE TO LOGIC)
# -------------------------
normalized_treatment_cnn = { normalize_key(k): v for k, v in (treatment_map_cnn or {}).items() }
normalized_treatment_bert = { normalize_key(k): v for k, v in (treatment_map_bert or {}).items() }

def get_treatment_for_label(predicted_label, source="cnn"):
    # ... (function body remains the same)
    nl = normalize_key(predicted_label)
    if source == "cnn":
        if nl in normalized_treatment_cnn:
            return normalized_treatment_cnn[nl]
    else:
        if nl in normalized_treatment_bert:
            return normalized_treatment_bert[nl]
    if nl in normalized_treatment_bert:
        return normalized_treatment_bert[nl]
    if nl in normalized_treatment_cnn:
        return normalized_treatment_cnn[nl]
    candidates = list(normalized_treatment_bert.keys()) + list(normalized_treatment_cnn.keys())
    matches = get_close_matches(nl, candidates, n=1, cutoff=0.85)
    if matches:
        return (normalized_treatment_bert.get(matches[0]) or normalized_treatment_cnn.get(matches[0]))
    for k, v in {**normalized_treatment_bert, **normalized_treatment_cnn}.items():
        if k.endswith(nl) or nl.endswith(k):
            return v
    return None

# -------------------------
# Preprocess & Predict helpers (NO CHANGE TO LOGIC)
# -------------------------
# def preprocess_image_bytes_for_cnn(file_bytes, target_size=IMAGE_TARGET_SIZE):
#     # ... (function body remains the same)
#     try:
#         pil_img = Image.open(io.BytesIO(file_bytes))
#     except Exception as e:
#         raise ValueError(f"Failed to open image: {e}")
#     if pil_img.mode != "RGB":
#         pil_img = pil_img.convert("RGB")
#     pil_img = ImageOps.fit(pil_img, target_size, method=RESAMPLE)
#     arr = np.asarray(pil_img).astype(np.float32)
#     arr = mobilenet_preprocess(arr)
#     arr = np.expand_dims(arr, 0)
#     return arr

def preprocess_image_bytes_for_cnn(file_bytes, target_size=IMAGE_TARGET_SIZE):
    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
    except Exception as e:
        raise ValueError(f"Failed to open image: {e}")

    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    pil_img = ImageOps.fit(pil_img, target_size, method=RESAMPLE)

    arr = np.asarray(pil_img).astype(np.float32)

    # IMPORTANT: use same preprocessing as training
    arr = arr / 255.0

    arr = np.expand_dims(arr, 0)

    return arr


def predict_cnn_from_bytes(file_bytes):
    # ... (function body remains the same)
    if cnn_model is None:
        return None, None, "CNN model not loaded"
    try:
        inp = preprocess_image_bytes_for_cnn(file_bytes)
        preds = cnn_model.predict(inp)
        if isinstance(preds, (list, tuple)):
            logits = preds[0]
        else:
            logits = preds
        probs = logits[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        return idx, conf, None
    except Exception as e:
        return None, None, f"CNN prediction error: {e}"

def predict_text_with_bert_local(text, max_length=128):
    # ... (function body remains the same)
    if bert_model is None or bert_tokenizer is None:
        return None, None, "BERT model/tokenizer not loaded"
    try:
        enc = bert_tokenizer(text, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(bert_device) for k, v in enc.items()}
        with torch.no_grad():
            out = bert_model(**enc)
            logits = out.logits
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            return idx, conf, None
    except Exception as e:
        return None, None, f"BERT inference error: {e}"

def cnn_idx_to_label(idx):
    # ... (function body remains the same)
    if 0 <= idx < len(classes_list):
        return classes_list[idx]
    return f"class_{idx}"

def bert_idx_to_label(idx):
    # ... (function body remains the same)
    if isinstance(bert_classes, list) and 0 <= idx < len(bert_classes):
        return bert_classes[idx]
    return f"class_{idx}"

# -------------------------
# Streamlit UI (REPLICATING SCREENSHOTS)
# -------------------------

# Sidebar Navigation Setup
st.sidebar.title("🌿 PlantDocBot")
st.sidebar.markdown("---")
# Use exact names from your screenshots for navigation
page = st.sidebar.selectbox("Navigate", ["Home", "Diagnose", "About"]) 

if load_errors:
    st.sidebar.error("⚠️ Model Loading Errors:")
    for e in load_errors:
        st.sidebar.write(f"- {e}")
    st.sidebar.markdown("---")


if page == "Home":
    
    # --- Home Page Content (Replicating Screenshot 1 & 2) ---
    
    # Centered Header
    col_center, col_main, col_center2 = st.columns([1, 4, 1])
    with col_main:
        # Custom CSS classes for the main title look
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.markdown("<span class='ai-powered'>🌿 AI-Powered Plant Health Assistant</span>", unsafe_allow_html=True)
        st.markdown("<h1>Your Plants Deserve <span class='expert-care'>Expert Care</span></h1>", unsafe_allow_html=True)
        st.markdown("""
        <p style='text-align: center; font-size: 1.1rem; color: #6a6a6a;'>
        Instant diagnosis and expert treatment advice for your plants. Upload a photo or<br>
        describe symptoms to get started.
        </p>
        """, unsafe_allow_html=True)

        # Start Diagnosis button (using a custom ID for the CSS)
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        if st.button("Start Diagnosis", key="start_diagnosis_home"):
            st.session_state.page = "Diagnose" # Change page programmatically
            st.rerun() # Rerun to switch page

        st.markdown("<br><br><br></div>", unsafe_allow_html=True)
        
        # How It Works Section
        st.markdown("<h2 style='text-align: center;'>How It Works</h2>", unsafe_allow_html=True)
        st.markdown("""
        <p style='text-align: center; color: #6a6a6a;'>
        Two powerful AI models working together to identify and treat plant diseases
        </p>
        """, unsafe_allow_html=True)
        
        # Two Columns for the cards
        col_img, col_sym = st.columns(2)
        
        # Image Analysis Card (using a container to simulate the card look)
        with col_img, st.container():
            st.markdown("### 📸 Image Analysis")
            st.markdown("""
            <p style='color: #6a6a6a;'>
            Upload a photo of your plant and our CNN model will analyze visual patterns to 
            identify diseases with high accuracy.
            </p>
            """, unsafe_allow_html=True)
        
        # Symptom Analysis Card
        with col_sym, st.container():
            st.markdown("### 📝 Symptom Analysis")
            st.markdown("""
            <p style='color: #6a6a6a;'>
            Describe what you're seeing and our NLP model will understand and classify 
            the disease based on your description.
            </p>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


elif page == "About":
    
    # --- About Page Content (Replicating Screenshot 3 & 4) ---
    
    st.markdown("<h1 style='text-align: center;'>About PlantDocBot</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6a6a6a;'>Advanced AI technology for accurate plant disease detection</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Three Feature Cards
    col_feat1, col_feat2, col_feat3 = st.columns(3)

    with col_feat1, st.container():
        st.markdown("### 🖼️ Image Recognition")
        st.markdown("**MobileNetV2 CNN**")
        st.caption("Deep learning model trained on thousands of plant images to identify visual disease patterns.")

    with col_feat2, st.container():
        st.markdown("### 💬 Text Analysis")
        st.markdown("**DistilBERT NLP**")
        st.caption("Natural language processing to understand and classify diseases from symptom descriptions.")
    
    with col_feat3, st.container():
        st.markdown("### ⚡ Instant Results")
        st.markdown("**Real-time Analysis**")
        st.caption("Fast, accurate diagnoses with detailed treatment recommendations in seconds.")

    st.markdown("---")
    st.markdown("<h2 style='text-align: center;'>How It Works</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Detailed Steps
    steps = [
        "**Input Your Data:** Upload a clear photo of the affected plant or describe the symptoms you're observing.",
        "**AI Analysis:** The system automatically routes the input to the appropriate model (CNN for images, BERT for text).",
        "**Prediction & Confidence:** The model provides the most probable disease and a confidence score.",
        "**Treatment Lookup:** The predicted disease name is matched to a comprehensive database to retrieve specific treatment and prevention advice."
    ]
    
    for i, step in enumerate(steps):
        st.markdown(f"### {i+1}. {step.split(':')[0]}", unsafe_allow_html=True)
        st.markdown(step.split(':')[1] if len(step.split(':')) > 1 else "", unsafe_allow_html=True)
        st.markdown("---")
        

elif page == "Diagnose":
    
    # --- Disease Recognition Page Content (Replicating Screenshot 5) ---

    st.markdown("<h1 style='text-align: center;'>Plant Disease Diagnosis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6a6a6a;'>Upload an image or describe symptoms for instant analysis</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Input Columns (Cards)
    col1, col2 = st.columns(2)

    with col1:
        with st.container(): # Simulating the dashed box container
            st.markdown("## ⬆️ Upload Image", unsafe_allow_html=True)
            st.markdown("Take a clear photo of the affected plant", unsafe_allow_html=True)
            
            # The actual file uploader is placed inside a Streamlit container
            uploaded = st.file_uploader("Click to upload image", type=["jpg", "jpeg", "png"], accept_multiple_files=False, key="diagnosis_uploader", label_visibility="collapsed")
            st.markdown("<p style='text-align: center; color: #6a6a6a; font-size:0.9rem;'>Click to upload image</p>", unsafe_allow_html=True)


    with col2:
        with st.container(): # Simulating the text box container
            st.markdown("## 📝 Describe Symptoms", unsafe_allow_html=True)
            st.markdown("Or type what you're observing", unsafe_allow_html=True)
            symptoms = st.text_area("Symptoms", placeholder="E.g., Yellow spots on leaves, wilting stems...", height=180, label_visibility="collapsed")
            
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Center the custom-styled Analyze Button
    col_btn_center, col_btn_main, col_btn_center2 = st.columns([1, 1, 1])
    with col_btn_main:
        if st.button("Analyze Plant", use_container_width=True, key="analyze_plant_main"):
            
            # --- PREDICTION LOGIC (NO CHANGE) ---
            predicted_label = None
            confidence = None
            source = None
            last_error = None
            raw_idx = None

            if uploaded is not None:
                try:
                    file_bytes = uploaded.read()
                    st.image(Image.open(io.BytesIO(file_bytes)), caption="Uploaded Image", use_container_width=True)
                except Exception:
                    pass
                idx, conf, err = predict_cnn_from_bytes(file_bytes)
                raw_idx = idx
                
                if err:
                    last_error = err
                else:
                    predicted_label = cnn_idx_to_label(idx)
                    confidence = conf
                    source = "image (CNN)"

            if predicted_label is None and symptoms and symptoms.strip():
                idx, conf, err = predict_text_with_bert_local(symptoms)
                if err:
                    last_error = err
                else:
                    predicted_label = bert_idx_to_label(idx)
                    confidence = conf
                    source = "text (BERT)"
            
            # --- DISPLAY RESULTS ---
            if predicted_label is None:
                if last_error:
                    st.error(f"❌ Prediction failed: {last_error}")
                else:
                    st.info("💡 Please provide an image or enter symptoms then press **Analyze Plant**.")
            else:
                pct = round(confidence * 100, 2) if confidence is not None else "N/A"

                # Success message for prediction
                st.success(f"✅ **Predicted Disease:** {predicted_label}")

                # Metrics in a styled container
                with st.container():
                    st.markdown("### 📊 Analysis Details")
                    meta_col1, meta_col2 = st.columns(2)
                    with meta_col1:
                        st.metric(label="🔍 Source Model", value=source)
                    with meta_col2:
                        st.metric(label="🎯 Confidence Score", value=f"{pct}%")

                if source == "image (CNN)":
                    info = get_treatment_for_label(predicted_label, source="cnn")
                else:
                    info = get_treatment_for_label(predicted_label, source="bert")

                st.markdown("---")

                if info is None:
                    st.warning("⚠️ No treatment information found for this disease.")
                else:
                    st.markdown("### 🩹 Treatment & Management Recommendations")

                    if isinstance(info, list):
                        with st.expander("📋 General Advice", expanded=True):
                            for s in info:
                                st.write(f"• {s}")
                    elif isinstance(info, dict):
                        # Description
                        if info.get("description"):
                            with st.expander("📖 Disease Description", expanded=True):
                                st.write(info['description'])

                        # Symptoms
                        if info.get("symptoms"):
                            with st.expander("🔴 Common Symptoms"):
                                for s in info["symptoms"]:
                                    st.write(f"• {s}")

                        # Treatment
                        if info.get("treatment"):
                            with st.expander("🛠️ Recommended Treatment", expanded=True):
                                for t in info["treatment"]:
                                    st.write(f"• {t}")

                        # Prevention
                        if info.get("prevention"):
                            with st.expander("🛡️ Prevention Tips"):
                                for p in info["prevention"]:
                                    st.write(f"• {p}")

                        # Severity
                        if info.get("severity"):
                            st.info(f"❗ **Severity Level:** {info['severity']}")

                        # When to seek help
                        if info.get("when_to_seek_help"):
                            st.warning(f"🚨 **When to Seek Professional Help:** {info['when_to_seek_help']}")
                    else:
                        st.write(info)

st.markdown("---")

st.caption("© PlantDocBot. Uses MobileNetV2 for images and DistilBERT for text.")














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
