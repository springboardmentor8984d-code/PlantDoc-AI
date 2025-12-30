# app.py - PlantDocBot (final integrated UI - STYLISH & CORRECTED)
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import re
import torch

# ---- Transformers (text classifier) ----
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---- TensorFlow Keras for CNN ----
import tensorflow as tf

# ---------------- CONFIG - update these if needed ----------------
CNN_MODEL_PATH = r"C:\Users\paul\Plant_Disease_CNN_Model.h5"   # <- your local CNN .h5
BERT_MODEL_PATH = r"C:\Users\paul\Plant_Symptom_DistilBERT"      # <- your local HF text-classifier folder
CSV_PATH = r"C:\Users\paul\disease_treatment_mapping_full.csv"    # <- uploaded CSV

# ---------------- Custom CSS for Styling ----------------
def set_styles():
    """Applies custom CSS for a stylish, nature-inspired look."""
    st.markdown("""
        <style>
        /* Main background color for a subtle, earthy feel */
        .stApp {
            background-color: #f7fbf8; 
        }
        /* Style for the main title */
        h1 {
            color: #1a5e30; /* Dark green */
            font-weight: 700;
            border-bottom: 2px solid #a9dfa8; /* Light green underline */
            padding-bottom: 10px;
        }
        /* Style for subheaders */
        h2, h3 {
            color: #2e8b57; /* Sea green */
        }
        /* Make the result boxes stand out */
        .stAlert {
            border-left: 6px solid #2e8b57 !important;
        }
        /* Center image display */
        [data-testid="stImage"] {
            text-align: center;
        }
        /* Customize the sidebar for a better feel */
        [data-testid="stSidebar"] {
            background-color: #e0f2e0; /* Very light green background */
        }
        </style>
    """, unsafe_allow_html=True)


# ---------------- Streamlit page (FIXED ORDER) ----------------

# 🛑 FIX: THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="PlantDocBot", page_icon="🌿", layout="wide")

# Now apply the styling
set_styles()

st.title("🌿 PlantDocBot: AI Plant Disease Diagnosis")
st.caption("Diagnosis via Image Recognition (CNN) and Symptom Analysis (BERT)")
st.write("") 

# ----------------- Load CSV mapping -----------------
@st.cache_data
def load_mapping(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8", encoding_errors="ignore")
    for c in ["plant","disease","key","treatment"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    df["plant_norm"] = df["plant"].str.lower().str.strip()
    df["disease_norm"] = df["disease"].str.lower().str.strip()
    df["key_norm"] = df["key"].str.lower().str.strip()
    return df

treatment_df = load_mapping(CSV_PATH)

# 🛑 CRITICAL: PASTE YOUR FINAL 32-CLASS LIST HERE 
# Replace the placeholder below with the exact 'final_class_order' list from your BERT notebook
disease_list = [
    # Example structure (MUST BE 32 ITEMS, ALPHABETICALLY SORTED)
    'alternaria', 
    'anthracnose', 
    'bacterial blight', 
    'bacterial spot', 
    'black rot', 
    'blight', 
    'brown rust', 
    'canker', 
    'cercospora leaf spot', 
    'common rust', 
    'curl virus', 
    'downy mildew', 
    'early blight', 
    'healthy', 
    'late blight', 
    'leaf mold', 
    'leaf spot', 
    'mildew', 
    'mosaic virus', 
    'northern leaf blight', 
    'other', 
    'phytophthora', 
    'powdery mildew', 
    'rot', 
    'rust', 
    'scab', 
    'septoria leaf spot', 
    'southern leaf blight', 
    'stem rust', 
    'target spot', 
    'wilt', 
    'yellow rust'
]


# ----------------- Load models (robust) -----------------
@st.cache_resource
def load_cnn_model(path):
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load CNN model from {path}: {e}")

@st.cache_resource
def load_text_model(path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.eval()
        return tokenizer, model
    except Exception as e:
        raise RuntimeError(f"Failed to load text model from {path}: {e}")

# Load safely and show status in sidebar
cnn_model = None
try:
    cnn_model = load_cnn_model(CNN_MODEL_PATH)
    st.sidebar.success(f"✅ CNN loaded ({cnn_model.output_shape[1]} classes)")
except Exception as e:
    st.sidebar.error("⚠️ CNN load failed. Check path/model format.")
    st.sidebar.caption(str(e)[:50] + "...")

tokenizer = model = None
try:
    tokenizer, model = load_text_model(BERT_MODEL_PATH)
    st.sidebar.success(f"✅ BERT loaded ({model.config.num_labels} classes)")
except Exception as e:
    st.sidebar.error("⚠️ BERT load failed. Check path/files.")
    st.sidebar.caption(str(e)[:50] + "...")

# Check for crucial class mismatch
if cnn_model is not None and model is not None and cnn_model.output_shape[1] != model.config.num_labels:
    st.sidebar.warning(f"🚨 Class Mismatch: CNN has {cnn_model.output_shape[1]} classes, BERT has {model.config.num_labels}. Fix your model training!")
elif cnn_model is not None and model is not None and cnn_model.output_shape[1] != len(disease_list):
    st.sidebar.warning(f"🚨 List Mismatch: Models have {cnn_model.output_shape[1]} classes, but disease_list has {len(disease_list)}. Update `disease_list`!")


# ----------------- Helper utilities (No change needed) -----------------
def normalize_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()

def get_treatment_by_plant_and_disease(plant: str, disease: str):
    plant_n = normalize_text(plant)
    disease_n = normalize_text(disease)
    df = treatment_df[
        (treatment_df["plant_norm"].str.contains(plant_n, na=False)) &
        (treatment_df["disease_norm"] == disease_n)
    ]
    if not df.empty:
        return df.iloc[0]["treatment"], df.iloc[0]["plant"]
    df2 = treatment_df[treatment_df["disease_norm"] == disease_n]
    if not df2.empty:
        return df2.iloc[0]["treatment"], df2.iloc[0]["plant"]
    return None, None

def predict_image_disease(img_file):
    if cnn_model is None: return None, 0.0
    img = Image.open(img_file).convert("RGB").resize((224,224))
    arr = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)
    preds = cnn_model.predict(arr, verbose=0)
    probs = preds[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    try:
        disease_norm = disease_list[idx]
        return disease_norm.title(), conf
    except Exception:
        return None, conf

def predict_text_disease(plant_name, symptoms):
    if tokenizer is None or model is None: return None, 0.0
    full_text = f"Plant: {plant_name}. Symptoms: {symptoms}" if plant_name else symptoms
    
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    try:
        disease_norm = disease_list[idx]
        return disease_norm.title(), conf
    except Exception:
        return None, conf

# ----------------- Streamlit UI inputs (Styled with Columns) -----------------

st.markdown("### 📥 Input Data")
col_img, col_text = st.columns([1, 1.5], gap="large")

with col_img:
    uploaded_image = st.file_uploader(
        "Upload a leaf image (JPG/PNG)", 
        type=["jpg","jpeg","png"], 
        help="The CNN model will diagnose the image."
    )
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
with col_text:
    plant_name = st.text_input(
        "1. Plant Name", 
        placeholder="e.g., Tomato, Apple, Wheat", 
        help="Required for accurate treatment lookup."
    )
    symptoms = st.text_area(
        "2. Describe Symptoms", 
        placeholder="e.g., Yellow spots on leaves, white powdery growth on stem, wilting.",
        help="The BERT model will analyze this text."
    )
    st.markdown("") 
    diagnose_button = st.button("🚀 Run Diagnosis", use_container_width=True, type="primary")

st.markdown("---")


# ----------------- Main Diagnosis Logic -----------------

if diagnose_button:
    # 1. Basic validation
    if (not uploaded_image) and (not (plant_name.strip() and symptoms.strip())):
        st.error("Please upload an image **OR** provide both plant name and symptoms.")
        st.stop()
    
    # 2. Setup the Diagnosis Status Block
    diagnosis_status = st.status(
        label="🔬 Analyzing input...", 
        expanded=True
    )
    
    # 3. Run Models and Update Status
    with diagnosis_status:
        # --- Image Model ---
        image_disease, image_conf = None, 0.0
        if uploaded_image and cnn_model is not None:
            st.write("Running CNN for image classification...")
            image_disease, image_conf = predict_image_disease(uploaded_image)
        
        # --- Text Model ---
        text_disease, text_conf = None, 0.0
        if plant_name.strip() and symptoms.strip() and tokenizer is not None and model is not None:
            st.write("Running BERT for symptom analysis...")
            text_disease, text_conf = predict_text_disease(plant_name, symptoms)
        
        # --- Post-Processing / Heuristics (Your custom rule) ---
        if text_disease and "mildew" in text_disease.lower():
            if "water-soaked" in symptoms.lower() or "rapid" in symptoms.lower():
                if "late blight" in [d.lower() for d in disease_list]:
                    text_disease = "Late Blight"
                    text_conf = min(1.0, text_conf + 0.20)
                    st.caption("✨ Heuristic applied: Boosted to Late Blight due to water-soaked/rapid symptom keywords.")

        # --- Combination ---
        final_disease, final_conf, final_source = None, 0.0, None
        
        if image_disease and text_disease:
            if image_conf >= text_conf:
                final_disease, final_conf, final_source = image_disease, image_conf, "Image (Higher Confidence)"
            else:
                final_disease, final_conf, final_source = text_disease, text_conf, "Text (Higher Confidence)"
        elif image_disease:
            final_disease, final_conf, final_source = image_disease, image_conf, "Image Only"
        elif text_disease:
            final_disease, final_conf, final_source = text_disease, text_conf, "Text Only"

    # 4. Final Output Display
    
    if final_disease:
        diagnosis_status.update(label="✅ Analysis Complete!", state="complete", expanded=True)
        
        st.markdown("<br><h2>🌱 Diagnosis Result</h2>", unsafe_allow_html=True)
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.metric(
                label="Predicted Disease",
                value=final_disease,
                delta=f"Source: {final_source}",
                delta_color="off"
            )
        
        with col_res2:
            st.metric(
                label="Confidence Score",
                value=f"{final_conf:.2f}",
                delta=f"{final_conf * 100:.0f}% likelihood",
                delta_color="inverse" if final_conf < 0.7 else "normal"
            )
            
        # 5. Treatment Lookup
        st.markdown("### 💊 Recommended Treatment")
        
        if plant_name.strip():
            treatment, matched_plant = get_treatment_by_plant_and_disease(plant_name, final_disease)
        else:
            treatment, matched_plant = get_treatment_by_plant_and_disease("", final_disease)

        if treatment:
            st.success(f"Treatment found for **{matched_plant.title()}** facing **{final_disease}**:")
            # Use st.markdown with a bulleted list for clean display
            st.markdown(treatment.replace("•", "\n* ")) 
            st.caption("Disclaimer: Always consult local agricultural experts before applying treatments.")
        else:
            st.warning("⚠️ No specific treatment found in the CSV mapping for this diagnosis. Check if the disease name is spelled correctly in the CSV or try giving plant name explicitly.")

    else:
        diagnosis_status.update(label="❌ Analysis Failed.", state="error", expanded=True)
        st.error("Could not determine a reliable disease diagnosis with the available inputs. Please try providing more detailed symptoms or a clearer image.")

st.markdown("---")
st.caption("PlantDocBot — Final Integrated UI — Developed by you")