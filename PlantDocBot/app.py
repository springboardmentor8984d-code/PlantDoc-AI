# rv10.py — Plant Health Assistant (Cloud Ready + Professional UI + Detailed Text Logic)

import os
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import gdown

# ---------------------------------------------------------
# PAGE CONFIG & GLOBAL STYLE
# ---------------------------------------------------------
st.set_page_config(page_title="Plant Health Assistant", page_icon="🌿", layout="wide")

st.markdown(
    """
    <style>
    .app-title {
        font-size: 28px;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .app-subtitle {
        color: #9ca3af;
        margin-bottom: 1.2rem;
    }
    .card {
        background: #0f1720;
        color: #e6eef8;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.55);
    }
    .card h4 {
        margin-top: 0;
        margin-bottom: 0.75rem;
        font-size: 18px;
        font-weight: 600;
        color: #e6eef8;
    }
    .tag {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-weight: 600;
    }
    .tag-healthy {
        background: rgba(16, 185, 129, 0.18);
        color: #6ee7b7;
        border: 1px solid rgba(16, 185, 129, 0.45);
    }
    .tag-disease {
        background: rgba(248, 113, 113, 0.16);
        color: #fecaca;
        border: 1px solid rgba(248, 113, 113, 0.45);
    }
    .muted {
        color: #9ca3af;
        font-size: 13px;
    }
    .section-label {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #6b7280;
        margin-bottom: 0.3rem;
    }
    .result-line {
        font-size: 16px;
        margin-bottom: 0.25rem;
    }
    .result-label {
        color: #93c5fd;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='app-title'>🌿 Plant Health Assistant</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='app-subtitle'>AI-powered plant health insights from leaf images and symptom descriptions.</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# PATHS (LOCAL) + GOOGLE DRIVE IDS
# ---------------------------------------------------------
APP_DIR = Path(__file__).parent.resolve()

LABEL_ENCODER_FILE = APP_DIR / "label_encoder_merged.joblib"
IMAGE_LABEL_ENCODER_FILE = APP_DIR / "image_label_encoder.joblib"
DISEASE_TO_MERGED_CSV = APP_DIR / "disease_to_merged.csv"

MODELS_BASE = APP_DIR / "models"
MODELS_BASE.mkdir(exist_ok=True)

TEXT_MODEL_DIR = MODELS_BASE / "Final_Text_Disease_Model"
IMAGE_MODEL_FILE = MODELS_BASE / "plant_disease_model_cloud.h5"
REMEDY_CSV_PATH = MODELS_BASE / "plant_disease_dataset_clean_lemmatized03.csv"

# ✅ FINAL CLOUD MODEL IDS
GDRIVE_IMAGE_ID = "12aRYV9_laCwvonv20mJneshu5fBedZZL"
GDRIVE_TEXT_FOLDER_ID = "1L_yTMpvW5xFSKUFHQN-_mz5t2K9fjxnG"
GDRIVE_CSV_ID = "1hVEoCo-EecTqFdVWTP7c-nqSGl1iV3e6"

# ---------------------------------------------------------
# HELPERS TO DOWNLOAD FROM GOOGLE DRIVE (RUN ONCE)
# ---------------------------------------------------------
@st.cache_resource
def ensure_image_model_path() -> str:
    """Download image analysis component from Drive if missing."""
    if not IMAGE_MODEL_FILE.exists() or IMAGE_MODEL_FILE.stat().st_size == 0:
        gdown.download(
            id=GDRIVE_IMAGE_ID,
            output=str(IMAGE_MODEL_FILE),
            quiet=False,
        )
    return str(IMAGE_MODEL_FILE)


@st.cache_resource
def ensure_text_model_dir() -> str:
    """Download text analysis component from Drive if missing."""
    config_path = TEXT_MODEL_DIR / "config.json"
    if not config_path.exists():
        TEXT_MODEL_DIR.mkdir(exist_ok=True)
        gdown.download_folder(
            id=GDRIVE_TEXT_FOLDER_ID,
            output=str(TEXT_MODEL_DIR),
            quiet=False,
            use_cookies=False,
        )
    return str(TEXT_MODEL_DIR)


@st.cache_resource
def ensure_remedy_csv_path() -> str:
    """Download treatment knowledge base from Drive if missing."""
    if not REMEDY_CSV_PATH.exists() or REMEDY_CSV_PATH.stat().st_size == 0:
        gdown.download(
            id=GDRIVE_CSV_ID,
            output=str(REMEDY_CSV_PATH),
            quiet=False,
        )
    return str(REMEDY_CSV_PATH)

# ---------------------------------------------------------
# COARSE NORMALIZATION (group-level, for remedies + tags)
# ---------------------------------------------------------
def normalize_disease(d: str) -> str:
    d = "" if d is None else str(d).lower()
    if "healthy" in d:
        return "healthy"
    if "bacterial" in d or "spot" in d:
        return "bacterial"
    if "early blight" in d or "late blight" in d or "blight" in d:
        return "blight"
    if "powdery" in d or "mildew" in d or "mold" in d:
        return "mildew"
    if "virus" in d or "mosaic" in d:
        return "viral"
    if "rot" in d or "canker" in d:
        return "rot_mold"
    if "scab" in d:
        return "scab"
    if "curl" in d:
        return "curl"
    return "other"

# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------
@st.cache_resource
def load_nlp_model():
    text_model_path = ensure_text_model_dir()
    tokenizer = AutoTokenizer.from_pretrained(text_model_path)
    model = TFAutoModelForSequenceClassification.from_pretrained(text_model_path)
    label_encoder = joblib.load(LABEL_ENCODER_FILE)
    return tokenizer, model, label_encoder


@st.cache_resource
def load_image_model():
    image_model_path = ensure_image_model_path()
    return tf.keras.models.load_model(
        image_model_path,
        compile=False,
    )


@st.cache_resource
def load_image_label_encoder():
    enc = joblib.load(IMAGE_LABEL_ENCODER_FILE)
    if isinstance(enc, dict):
        # dict: class_name -> idx  => we want idx -> class_name
        return {v: k for k, v in enc.items()}
    # sklearn LabelEncoder
    return {i: cls for i, cls in enumerate(enc.classes_)}

# ---------------------------------------------------------
# TREATMENT MAPS (detailed, like local version)
#   - remedy_by_merged:  exact merged_label  -> remedy
#   - remedy_by_norm:    normalized group    -> remedy
#   - display_name_by_merged: merged_label   -> friendly disease name
# ---------------------------------------------------------
@st.cache_resource
def load_treatment_map():
    csv_path = ensure_remedy_csv_path()
    df = pd.read_csv(csv_path)

    # If merged_label not present, rebuild from DISEASE_TO_MERGED_CSV or fallback
    if "merged_label" not in df.columns:
        if DISEASE_TO_MERGED_CSV.exists():
            mapdf = pd.read_csv(DISEASE_TO_MERGED_CSV)
            disease_to_merged = dict(
                zip(
                    mapdf["Disease"].astype(str).str.lower(),
                    mapdf["merged_label"].astype(str).str.lower(),
                )
            )
            df["merged_label"] = (
                df["Disease"].astype(str).str.lower().map(disease_to_merged)
            )
            df["merged_label"] = df["merged_label"].fillna("other")
        else:
            df["merged_label"] = df["Disease"].astype(str).str.lower()

    df["merged_label"] = df["merged_label"].astype(str).str.lower()
    df["merged_norm"] = df["merged_label"].apply(normalize_disease)

    # Helper – most common non-empty remedy
    def most_common_remedy(series):
        s = series.dropna().astype(str).str.strip()
        if len(s) == 0:
            return "No remedy available."
        return s.value_counts().index[0]

    # 1) Remedy keyed by exact merged_label
    remedy_by_merged = (
        df.groupby("merged_label")["Remedy"]
        .agg(most_common_remedy)
        .to_dict()
    )

    # 2) Remedy keyed by normalized group
    remedy_by_norm = (
        df.groupby("merged_norm")["Remedy"]
        .agg(most_common_remedy)
        .to_dict()
    )

    remedy_by_merged.setdefault("other", "No remedy available.")
    remedy_by_norm.setdefault("other", "No remedy available.")
    remedy_by_norm.setdefault("healthy", "No treatment needed.")

    # 3) Friendly display name per merged_label (most common Disease name)
    display_name_by_merged = (
        df.groupby("merged_label")["Disease"]
        .agg(lambda x: x.dropna().astype(str).value_counts().index[0])
        .to_dict()
    )

    for k in list(display_name_by_merged.keys()):
        if pd.isna(display_name_by_merged[k]) or display_name_by_merged[k] == "":
            display_name_by_merged[k] = k.title()

    display_name_by_merged.setdefault("other", "Other")
    display_name_by_merged.setdefault("healthy", "Healthy")

    return remedy_by_merged, remedy_by_norm, display_name_by_merged

# ---------------------------------------------------------
# LOAD CORE COMPONENTS ONCE
# ---------------------------------------------------------
with st.spinner("Preparing assistant…"):
    tokenizer, nlp_model, label_encoder = load_nlp_model()
    cnn_model = load_image_model()
    img_idx_to_class = load_image_label_encoder()
    remedy_by_merged, remedy_by_norm, display_name_by_merged = load_treatment_map()

# ---------------------------------------------------------
# BACKEND PREDICT HELPERS
# ---------------------------------------------------------
def split_plant_and_disease(class_name):
    if not isinstance(class_name, str):
        class_name = str(class_name)
    c = class_name.replace("___", "_").replace("__", "_")
    parts = c.split("_")
    plant = parts[0].title() if parts else "Unknown"
    disease = " ".join(parts[1:]).replace("_", " ").title() if len(parts) > 1 else "Unknown"
    return plant, disease


def predict_text(symptoms: str):
    """
    Uses BERT -> merged_label index, then:
      - picks the exact merged_label string
      - maps it to a friendly Disease name using display_name_by_merged
      - picks remedy by exact merged_label first, then fallback to normalized group
    This is the detailed version (e.g. 'Powdery Mildew', not only 'Mildew').
    """
    enc = tokenizer(
        symptoms,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="tf",
    )
    outputs = nlp_model(enc)
    probs = tf.nn.softmax(outputs.logits, axis=1).numpy().squeeze()
    pred_idx = int(np.argmax(probs))
    confidence = float(np.max(probs))

    merged_label = label_encoder.inverse_transform([pred_idx])[0]
    merged_label_str = str(merged_label).lower()

    # Friendly name: prefer Disease name from CSV
    display_name = display_name_by_merged.get(
        merged_label_str,
        merged_label_str.replace("_", " ").title(),
    )

    # Remedy: exact merged_label first, then group
    remedy = remedy_by_merged.get(merged_label_str, None)
    if not remedy:
        norm_key = normalize_disease(merged_label_str)
        remedy = remedy_by_norm.get(norm_key, "No remedy available.")
    group_key = normalize_disease(merged_label_str)

    return {
        "display_name": display_name,
        "merged_label": merged_label_str,
        "group_key": group_key,
        "confidence": confidence,
        "remedy": remedy,
    }


def predict_image(image_file):
    """
    Image CNN → class index → class_name
    Then:
      - plant + detailed disease name from class_name
      - remedy by exact disease string (if present in merged map)
      - fallback via normalized group
    """
    img = Image.open(image_file).convert("RGB").resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, 0).astype(np.float32)

    preds = cnn_model.predict(arr)
    idx = int(np.argmax(preds))
    confidence = float(np.max(preds))

    class_name = img_idx_to_class.get(idx, str(idx))
    plant, disease_full = split_plant_and_disease(class_name)

    disease_key_lower = str(disease_full).lower()
    group_key = normalize_disease(disease_key_lower)

    # Try exact disease name first
    remedy = remedy_by_merged.get(disease_key_lower, None)
    if not remedy:
        remedy = remedy_by_norm.get(group_key, "No remedy available.")

    return {
        "plant": plant,
        "disease_display": disease_full,
        "group_key": group_key,
        "confidence": confidence,
        "remedy": remedy,
    }

# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
if "image_result" not in st.session_state:
    st.session_state["image_result"] = None
if "text_result" not in st.session_state:
    st.session_state["text_result"] = None
if "uploaded_image" not in st.session_state:
    st.session_state["uploaded_image"] = None
if "symptom_text" not in st.session_state:
    st.session_state["symptom_text"] = ""

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("<div class='section-label'>Workspace</div>", unsafe_allow_html=True)
    st.write("Switch between **Leaf Image** and **Symptom Text** to analyze plant health.")
    st.divider()
    if st.button("Reset session"):
        st.session_state["image_result"] = None
        st.session_state["text_result"] = None
        st.session_state["uploaded_image"] = None
        st.session_state["symptom_text"] = ""
        st.success("Session cleared.")
    st.markdown("---")
    st.caption("Tip: use clear, close-up leaf photos and descriptive symptoms for best results.")

# ---------------------------------------------------------
# MAIN LAYOUT — TABS
# ---------------------------------------------------------
tab_img, tab_text = st.tabs(["📸 Leaf Image", "📝 Symptom Text"])

# ======================= IMAGE TAB =======================
with tab_img:
    st.markdown("<div class='section-label'>Image Input</div>", unsafe_allow_html=True)
    st.subheader("Analyze a leaf photo")

    c1, c2 = st.columns([1.1, 0.9])

    with c1:
        uploaded = st.file_uploader(
            "Upload a leaf image (JPG / PNG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded is not None:
            st.session_state["uploaded_image"] = uploaded

        if st.session_state["uploaded_image"] is not None:
            st.image(
                st.session_state["uploaded_image"],
                caption="Preview",
                use_container_width=True,
            )

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Run analysis", type="primary"):
                if st.session_state["uploaded_image"] is None:
                    st.warning("Please upload a leaf image first.")
                else:
                    with st.spinner("Analyzing leaf image…"):
                        prog = st.progress(0)
                        for i in range(0, 101, 25):
                            prog.progress(i)
                        res = predict_image(st.session_state["uploaded_image"])
                    st.session_state["image_result"] = res
                    st.success("Analysis complete.")

        with col_btn2:
            if st.button("Clear image"):
                st.session_state["uploaded_image"] = None
                st.session_state["image_result"] = None

    with c2:
        st.markdown("<div class='card'><h4>Result</h4>", unsafe_allow_html=True)

        res = st.session_state["image_result"]
        if res is None:
            st.markdown(
                "<span class='muted'>Upload a leaf image and run the analysis to see results here.</span>",
                unsafe_allow_html=True,
            )
        else:
            is_healthy = res["group_key"] == "healthy"
            tag_class = "tag-healthy" if is_healthy else "tag-disease"
            tag_text = "Likely healthy" if is_healthy else "Likely diseased"

            st.markdown(
                f"<span class='tag {tag_class}'>{tag_text}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='result-line'><span class='result-label'>Plant:</span> {res['plant']}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='result-line'><span class='result-label'>Condition detected:</span> {res['disease_display']}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='result-line'><span class='result-label'>Confidence:</span> {res['confidence']*100:.2f}%</div>",
                unsafe_allow_html=True,
            )

            st.markdown("---")
            st.markdown("**Recommended treatment / next steps:**")
            st.write(res["remedy"])

        st.markdown("</div>", unsafe_allow_html=True)

# ======================= TEXT TAB =======================
with tab_text:
    st.markdown("<div class='section-label'>Text Input</div>", unsafe_allow_html=True)
    st.subheader("Describe what you see on the plant")

    st.markdown(
        "<span class='muted'>Mention leaf colour changes, spots, patterns, spread speed, and where on the plant it appears.</span>",
        unsafe_allow_html=True,
    )

    c3, c4 = st.columns([1.1, 0.9])

    with c3:
        st.session_state["symptom_text"] = st.text_area(
            "Describe symptoms here…",
            height=160,
            label_visibility="collapsed",
            value=st.session_state["symptom_text"],
        )

        btn_analyze, btn_clear = st.columns(2)
        with btn_analyze:
            if st.button("Analyze description", type="primary"):
                if not st.session_state["symptom_text"].strip():
                    st.warning("Please describe the symptoms first.")
                else:
                    with st.spinner("Analyzing description…"):
                        prog2 = st.progress(0)
                        for i in range(0, 101, 25):
                            prog2.progress(i)
                        res_txt = predict_text(st.session_state["symptom_text"])
                    st.session_state["text_result"] = res_txt
                    st.success("Analysis complete.")

        with btn_clear:
            if st.button("Clear text"):
                st.session_state["symptom_text"] = ""
                st.session_state["text_result"] = None

    with c4:
        st.markdown("<div class='card'><h4>Result</h4>", unsafe_allow_html=True)

        res = st.session_state["text_result"]
        if res is None:
            st.markdown(
                "<span class='muted'>Enter symptoms and run the analysis to see a detailed interpretation here.</span>",
                unsafe_allow_html=True,
            )
        else:
            is_healthy = res["group_key"] == "healthy"
            tag_class = "tag-healthy" if is_healthy else "tag-disease"
            tag_text = "Likely healthy" if is_healthy else "Likely diseased"

            st.markdown(
                f"<span class='tag {tag_class}'>{tag_text}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='result-line'><span class='result-label'>Condition detected:</span> {res['display_name']}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='result-line'><span class='result-label'>Confidence:</span> {res['confidence']*100:.2f}%</div>",
                unsafe_allow_html=True,
            )

            st.markdown("---")
            st.markdown("**Recommended treatment / next steps:**")
            st.write(res["remedy"])

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")
f1, f2 = st.columns([3, 1])
with f1:
    st.caption(
        "For best results, avoid very blurry photos and include clear descriptions of all visible symptoms."
    )
with f2:
    st.caption("PlantDocBot · Demo interface")