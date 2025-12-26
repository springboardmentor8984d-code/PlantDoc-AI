import streamlit as st
import numpy as np
import pickle
from PIL import Image
import io
import requests
import json
from pathlib import Path
import base64

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="🌿 PlantDoc Bot",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

def load_custom_css():
    st.markdown("""
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&family=Playfair+Display:wght@700&display=swap');
        
        /* Main Background with Gradient */
        .stApp {
            background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
            font-family: 'Poppins', sans-serif;
        }
        
        /* Header Styling */
        .main-header {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            padding: 2rem;
            border-radius: 20px;
            text-align: center;
            box-shadow: 0 10px 40px rgba(17, 153, 142, 0.3);
            margin-bottom: 2rem;
            animation: fadeInDown 0.8s ease-in-out;
        }
        
        .main-header h1 {
            font-family: 'Playfair Display', serif;
            font-size: 3.5rem;
            color: white;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
            margin: 0;
            letter-spacing: 2px;
        }
        
        .main-header p {
            font-size: 1.2rem;
            color: #e8f5e9;
            margin-top: 0.5rem;
            font-weight: 300;
        }
        
        /* Card Styling */
        .custom-card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
            margin-bottom: 2rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            animation: fadeInUp 0.8s ease-in-out;
        }
        
        /* Feature Box */
        .feature-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            padding: 1.5rem;
            text-align: center;
            color: white;
            margin: 1rem 0;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        }
        
        .feature-box:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.5);
        }
        
        .feature-box h3 {
            font-size: 1.3rem;
            margin-bottom: 0.5rem;
            font-weight: 600;
        }
        
        .feature-box p {
            font-size: 0.9rem;
            opacity: 0.9;
            margin: 0;
        }
        
        /* Result Box */
        .result-box {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            border-radius: 20px;
            padding: 2rem;
            color: white;
            margin: 2rem 0;
            box-shadow: 0 10px 30px rgba(17, 153, 142, 0.4);
            animation: slideInUp 0.5s ease-in-out;
        }
        
        .result-box h2 {
            font-size: 2rem;
            margin-bottom: 1rem;
            font-weight: 700;
        }
        
        .confidence-bar {
            background: rgba(255, 255, 255, 0.3);
            border-radius: 10px;
            height: 30px;
            margin: 1rem 0;
            overflow: hidden;
        }
        
        .confidence-fill {
            background: linear-gradient(90deg, #ff6b6b, #feca57);
            height: 100%;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            transition: width 1s ease-in-out;
        }
        
        /* Treatment Card */
        .treatment-card {
            background: white;
            border-left: 5px solid #11998e;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        
        .treatment-card h4 {
            color: #11998e;
            font-size: 1.3rem;
            margin-bottom: 1rem;
        }
        
        .treatment-card ul {
            list-style-type: none;
            padding-left: 0;
        }
        
        .treatment-card li {
            padding: 0.5rem 0;
            padding-left: 1.5rem;
            position: relative;
        }
        
        .treatment-card li:before {
            content: "✓";
            position: absolute;
            left: 0;
            color: #11998e;
            font-weight: bold;
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            border-radius: 50px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.6);
        }
        
        /* Sidebar Styling */
        .css-1d391kg {
            background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
        }
        
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
        }
        
        /* Animations */
        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes slideInUp {
            from {
                opacity: 0;
                transform: translateY(50px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Tab Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 0.5rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 10px;
            color: white;
            font-weight: 600;
            font-size: 1.1rem;
            padding: 1rem 2rem;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        /* File Uploader */
        .uploadedFile {
            border: 2px dashed #11998e;
            border-radius: 15px;
            padding: 2rem;
            text-align: center;
            background: rgba(17, 153, 142, 0.05);
        }
        
        /* Text Input */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {
            background: rgba(255, 255, 255, 0.95);
            border: 2px solid #11998e;
            border-radius: 10px;
            padding: 1rem;
            font-size: 1rem;
            color: #2c3e50;
        }
        
        /* Success/Error Messages */
        .stSuccess {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            border-radius: 10px;
            padding: 1rem;
        }
        
        .stError {
            background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
            border-radius: 10px;
            padding: 1rem;
        }
        
        /* Loading Spinner */
        .stSpinner > div {
            border-top-color: #11998e !important;
        }
        
        /* Info Box */
        .info-box {
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            color: #2c3e50;
            box-shadow: 0 5px 15px rgba(252, 182, 159, 0.3);
        }
        
        .info-box h4 {
            color: #d35400;
            margin-bottom: 0.5rem;
        }
        
        /* Metric Cards */
        .metric-card {
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            margin: 1rem;
        }
        
        .metric-card h3 {
            color: #11998e;
            font-size: 2.5rem;
            margin: 0;
        }
        
        .metric-card p {
            color: #7f8c8d;
            margin-top: 0.5rem;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 2rem;
            margin-top: 3rem;
            color: rgba(255, 255, 255, 0.7);
            font-size: 0.9rem;
        }
        
        /* Glow Effect */
        .glow {
            animation: glow 2s ease-in-out infinite alternate;
        }
        
        @keyframes glow {
            from {
                box-shadow: 0 0 10px rgba(17, 153, 142, 0.5);
            }
            to {
                box-shadow: 0 0 20px rgba(17, 153, 142, 0.8);
            }
        }
        </style>
    """, unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Model paths
    MODEL_PATH = 'models/finetuned'
    FINETUNED_MODEL = 'finetuned_plantdoc_model.h5'
    LABEL_ENCODER = 'combined_label_encoder.pkl'
    VECTORIZER = 'combined_vectorizer.pkl'
    
    # Image parameters
    IMG_SIZE = (224, 224)
    
    # NVIDIA API Configuration
    NVIDIA_API_KEY = st.secrets.get("NVIDIA_API_KEY", "api_key")
    NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
    NVIDIA_MODEL = "meta/llama3-8b-instruct"  # You can change this

config = Config()

# ============================================================================
# LOAD MODELS AND COMPONENTS
# ============================================================================

@st.cache_resource
def load_model_components():
    """Load the trained model and components"""
    try:
        model = keras.models.load_model(f'{config.MODEL_PATH}/{config.FINETUNED_MODEL}')
        
        with open(f'{config.MODEL_PATH}/{config.LABEL_ENCODER}', 'rb') as f:
            label_encoder = pickle.load(f)
        
        with open(f'{config.MODEL_PATH}/{config.VECTORIZER}', 'rb') as f:
            vectorizer = pickle.load(f)
        
        return model, label_encoder, vectorizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# ============================================================================
# TREATMENT DATABASE
# ============================================================================

def get_treatment_database():
    """Comprehensive treatment database"""
    return {
        'tomato_early_blight': {
            'description': 'Early blight is caused by the fungus Alternaria solani. It appears as dark spots with concentric rings on older leaves.',
            'symptoms': ['Dark spots with target-like rings', 'Yellow halo around spots', 'Leaf yellowing and dropping'],
            'treatments': [
                'Remove and destroy infected leaves immediately',
                'Apply copper-based fungicide every 7-10 days',
                'Use chlorothalonil or mancozeb fungicides',
                'Maintain proper plant spacing (24-36 inches)',
                'Avoid overhead watering - water at base',
                'Practice 3-4 year crop rotation'
            ],
            'prevention': [
                'Use disease-resistant varieties',
                'Mulch around plants to prevent soil splash',
                'Water early in the day',
                'Remove plant debris at end of season'
            ],
            'severity': 'Moderate'
        },
        'tomato_late_blight': {
            'description': 'Late blight is caused by Phytophthora infestans, a devastating disease that can destroy entire crops.',
            'symptoms': ['Water-soaked spots on leaves', 'White fuzzy growth on undersides', 'Rapid plant death', 'Brown rotting fruit'],
            'treatments': [
                'Remove and destroy infected plants IMMEDIATELY',
                'Apply copper fungicide preventively',
                'Use mancozeb or chlorothalonil',
                'Apply fixed copper sprays every 5-7 days in wet weather',
                'Harvest healthy fruit immediately'
            ],
            'prevention': [
                'Plant certified disease-free seeds',
                'Avoid overhead irrigation',
                'Ensure excellent air circulation',
                'Monitor weather (favors cool 60-70°F, wet conditions)'
            ],
            'severity': 'Severe'
        },
        'potato_early_blight': {
            'description': 'Early blight affects potatoes causing yield reduction and quality issues.',
            'symptoms': ['Dark brown spots with concentric rings', 'Lower leaves affected first', 'Reduced tuber size'],
            'treatments': [
                'Apply mancozeb or chlorothalonil fungicide',
                'Remove infected foliage',
                'Improve soil drainage',
                'Ensure adequate potassium fertilization'
            ],
            'prevention': [
                'Use certified disease-free seed potatoes',
                'Hill soil properly around plants',
                'Practice 2-3 year crop rotation'
            ],
            'severity': 'Moderate'
        },
        'healthy': {
            'description': 'Your plant appears to be healthy with no visible disease symptoms!',
            'symptoms': ['Green, vigorous leaves', 'Normal growth pattern', 'No discoloration or spots'],
            'treatments': [
                'Continue regular care and monitoring',
                'Maintain consistent watering schedule',
                'Ensure adequate nutrition with balanced fertilizer'
            ],
            'prevention': [
                'Practice good garden hygiene',
                'Monitor plants weekly for early signs',
                'Maintain proper spacing between plants',
                'Water at base of plants, not leaves'
            ],
            'severity': 'None - Healthy!'
        }
    }

# ============================================================================
# NVIDIA API INTEGRATION
# ============================================================================

def query_nvidia_api(user_query: str, disease_context: str = None) -> str:
    """Query NVIDIA API for plant disease information"""
    
    if config.NVIDIA_API_KEY == "your-nvidia-api-key-here":
        return "⚠️ Please configure NVIDIA API key in Streamlit secrets"
    
    system_prompt = """You are PlantDoc Bot, an expert AI assistant specialized in plant disease diagnosis and agricultural health.

Your expertise includes:
- Plant disease identification and diagnosis
- Treatment recommendations and preventive measures
- Agricultural best practices
- Organic and chemical treatment options

IMPORTANT RESTRICTIONS:
- ONLY answer questions about plants, crops, agriculture, and plant diseases
- If asked about non-plant topics, politely decline and redirect to plant-related queries
- Provide practical, actionable advice
- Be concise but comprehensive
- Include both organic and chemical treatment options when relevant

If the query is not related to plants or agriculture, respond with:
"I'm PlantDoc Bot, specialized in plant disease diagnosis and treatment. I can only assist with plant health, agriculture, and crop-related questions. Please ask me about plant diseases, symptoms, treatments, or farming concerns."
"""
    
    user_message = user_query
    if disease_context:
        user_message = f"Context: {disease_context}\n\nUser Query: {user_query}"
    
    headers = {
        'Authorization': f'Bearer {config.NVIDIA_API_KEY}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'model': config.NVIDIA_MODEL,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_message}
        ],
        'temperature': 0.7,
        'max_tokens': 1024,
        'top_p': 1.0,
        'stream': False
    }
    
    try:
        response = requests.post(config.NVIDIA_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        return f"⚠️ API Error: {str(e)}"

# ============================================================================
# IMAGE PREDICTION
# ============================================================================

def predict_from_image(image, model, vectorizer, label_encoder):
    """Predict disease from image"""
    # Preprocess image
    img = image.resize(config.IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Create text features (generic for image-only prediction)
    text_features = vectorizer.transform(["plant leaf disease"]).toarray()
    
    # Predict
    predictions = model.predict([img_array, text_features], verbose=0)
    predicted_idx = np.argmax(predictions[0])
    predicted_disease = label_encoder.inverse_transform([predicted_idx])[0]
    confidence = predictions[0][predicted_idx] * 100
    
    # Get top 3 predictions
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    top_3_diseases = label_encoder.inverse_transform(top_3_idx)
    top_3_conf = predictions[0][top_3_idx] * 100
    
    return predicted_disease, confidence, list(zip(top_3_diseases, top_3_conf))

# ============================================================================
# TEXT QUERY PROCESSING
# ============================================================================

def is_plant_related(query: str) -> bool:
    """Check if query is plant-related"""
    plant_keywords = [
        'plant', 'leaf', 'crop', 'disease', 'pest', 'fungus', 'bacteria',
        'virus', 'blight', 'rust', 'mold', 'spot', 'rot', 'wilt', 'scab',
        'tomato', 'potato', 'pepper', 'corn', 'apple', 'grape', 'agriculture',
        'farming', 'garden'
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in plant_keywords)

# ============================================================================
# MAIN UI
# ============================================================================

def main():
    # Load custom CSS
    load_custom_css()
    
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🌿 PlantDoc Bot</h1>
            <p>Advanced AI-Powered Plant Disease Detection & Treatment System</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load model
    with st.spinner("🔄 Loading AI models..."):
        model, label_encoder, vectorizer = load_model_components()
    
    if model is None:
        st.error("❌ Failed to load model. Please train the model first!")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎯 Navigation")
        
        st.markdown("""
            <div class="info-box">
                <h4>🌱 About PlantDoc Bot</h4>
                <p>PlantDoc Bot uses advanced deep learning and NLP to diagnose plant diseases from both images and text descriptions.</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 📊 System Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🎯 Diseases", len(label_encoder.classes_))
        with col2:
            st.metric("📈 Accuracy", "84%")
        
        st.markdown("---")
        
        st.markdown("### ⚙️ Settings")
        show_confidence = st.checkbox("Show Confidence Scores", value=True)
        show_treatments = st.checkbox("Show Treatments", value=True)
        
        st.markdown("---")
        
        st.markdown("""
            <div class="info-box">
                <h4>💡 Tips</h4>
                <p>• Use clear, well-lit images<br>
                • Focus on affected areas<br>
                • Describe symptoms clearly</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📸 Image Analysis", "💬 Text Query", "📚 Disease Database"])
    
    # ======== TAB 1: IMAGE ANALYSIS ========
    with tab1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown("""
                <div class="feature-box">
                    <h3>📤 Upload</h3>
                    <p>Upload a clear image of the affected plant</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="feature-box">
                    <h3>🔍 Analyze</h3>
                    <p>AI processes the image using CNN</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class="feature-box">
                    <h3>💊 Treat</h3>
                    <p>Get treatment recommendations</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image of the plant",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image showing the disease symptoms"
        )
        
        if uploaded_file is not None:
            # Display image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            with col2:
                if st.button("🔬 Analyze Image", type="primary", use_container_width=True):
                    with st.spinner("🔄 Analyzing image with AI..."):
                        # Predict
                        disease, confidence, top_3 = predict_from_image(
                            image, model, vectorizer, label_encoder
                        )
                        
                        # Display results
                        st.markdown(f"""
                            <div class="result-box">
                                <h2>🎯 Detection Results</h2>
                                <h3>Identified Disease: {disease}</h3>
                                <div class="confidence-bar">
                                    <div class="confidence-fill" style="width: {confidence}%">
                                        {confidence:.2f}%
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Top 3 predictions
                        if show_confidence:
                            st.markdown("### 📊 Top 3 Predictions")
                            for i, (d, conf) in enumerate(top_3, 1):
                                st.markdown(f"""
                                    <div class="treatment-card">
                                        <h4>{i}. {d}</h4>
                                        <div class="confidence-bar">
                                            <div class="confidence-fill" style="width: {conf}%">
                                                {conf:.2f}%
                                            </div>
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)
                        
                        # Get treatment info
                        if show_treatments:
                            treatments_db = get_treatment_database()
                            disease_key = disease.lower().replace(' ', '_').replace('___', '_')
                            
                            # Find matching treatment
                            treatment_info = None
                            for key, info in treatments_db.items():
                                if key in disease_key or disease_key in key:
                                    treatment_info = info
                                    break
                            
                            if treatment_info:
                                st.markdown("### 💊 Treatment Recommendations")
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown(f"""
                                        <div class="treatment-card">
                                            <h4>📋 Description</h4>
                                            <p>{treatment_info.get('description', 'N/A')}</p>
                                            <h4>🔴 Severity</h4>
                                            <p>{treatment_info.get('severity', 'Unknown')}</p>
                                        </div>
                                    """, unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown(f"""
                                        <div class="treatment-card">
                                            <h4>🔍 Symptoms</h4>
                                            <ul>
                                                {''.join([f'<li>{s}</li>' for s in treatment_info.get('symptoms', [])])}
                                            </ul>
                                        </div>
                                    """, unsafe_allow_html=True)
                                
                                st.markdown(f"""
                                    <div class="treatment-card">
                                        <h4>💊 Treatments</h4>
                                        <ul>
                                            {''.join([f'<li>{t}</li>' for t in treatment_info.get('treatments', [])])}
                                        </ul>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown(f"""
                                    <div class="treatment-card">
                                        <h4>🛡️ Prevention</h4>
                                        <ul>
                                            {''.join([f'<li>{p}</li>' for p in treatment_info.get('prevention', [])])}
                                        </ul>
                                    </div>
                                """, unsafe_allow_html=True)
    
    # ======== TAB 2: TEXT QUERY ========
    with tab2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        
        st.markdown("""
            <div class="info-box">
                <h4>💬 Ask PlantDoc Bot</h4>
                <p>Describe your plant's symptoms or ask questions about plant diseases. Our AI will provide expert guidance!</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Text input
        user_query = st.text_area(
            "Enter your query:",
            placeholder="Example: My tomato plants have dark spots with yellow halos. What should I do?",
            height=100
        )
        
        if st.button("🤖 Ask AI", type="primary", use_container_width=True):
            if user_query.strip():
                # Check if plant-related
                if not is_plant_related(user_query):
                    st.error("""
                        ⚠️ I'm PlantDoc Bot, specialized in plant disease diagnosis and treatment. 
                        I can only assist with plant health, agriculture, and crop-related questions. 
                        Please ask me about plant diseases, symptoms, treatments, or farming concerns.
                    """)
                else:
                    with st.spinner("🤖 Consulting AI expert..."):
                        # Query NVIDIA API
                        response = query_nvidia_api(user_query)
                        
                        st.markdown(f"""
                            <div class="result-box">
                                <h2>🤖 AI Expert Response</h2>
                                <p>{response}</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Match with known diseases
                        query_lower = user_query.lower()
                        matched_diseases = []
                        for disease in label_encoder.classes_:
                            disease_parts = disease.lower().replace('___', ' ').replace('__', ' ').replace('_', ' ').split()
                            for part in disease_parts:
                                if len(part) > 3 and part in query_lower:
                                    matched_diseases.append(disease)
                                    break
                        
                        if matched_diseases and show_treatments:
                            st.markdown("### 🔍 Possible Disease Matches")
                            treatments_db = get_treatment_database()
                            
                            for disease in matched_diseases[:3]:
                                disease_key = disease.lower().replace(' ', '_').replace('___', '_')
                                
                                for key, info in treatments_db.items():
                                    if key in disease_key or disease_key in key:
                                        st.markdown(f"""
                                            <div class="treatment-card">
                                                <h4>🦠 {disease}</h4>
                                                <p><strong>Severity:</strong> {info.get('severity', 'Unknown')}</p>
                                                <p>{info.get('description', 'N/A')}</p>
                                                <details>
                                                    <summary style="cursor: pointer; color: #11998e; font-weight: 600;">View Treatment Details</summary>
                                                    <h5>Treatments:</h5>
                                                    <ul>
                                                        {''.join([f'<li>{t}</li>' for t in info.get('treatments', [])])}
                                                    </ul>
                                                </details>
                                            </div>
                                        """, unsafe_allow_html=True)
                                        break
            else:
                st.warning("⚠️ Please enter a query")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Example queries
        st.markdown("### 💡 Example Queries")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🍅 Tomato leaf spots", use_container_width=True):
                st.session_state.example_query = "My tomato plants have dark spots with yellow halos"
            if st.button("🥔 Potato blight", use_container_width=True):
                st.session_state.example_query = "I see water-soaked lesions on my potato leaves"
        
        with col2:
            if st.button("🍎 Apple scab", use_container_width=True):
                st.session_state.example_query = "My apple tree has olive-green spots on leaves"
            if st.button("🌽 Corn disease", use_container_width=True):
                st.session_state.example_query = "Corn leaves have rectangular gray lesions"
    
    # ======== TAB 3: DISEASE DATABASE ========
    with tab3:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        
        st.markdown("## 📚 Comprehensive Disease Database")
        
        treatments_db = get_treatment_database()
        
        # Search functionality
        search_term = st.text_input("🔍 Search diseases", placeholder="Type disease name...")
        
        # Display diseases
        diseases_to_show = label_encoder.classes_
        if search_term:
            diseases_to_show = [d for d in diseases_to_show if search_term.lower() in d.lower()]
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
                <div class="metric-card">
                    <h3>42</h3>
                    <p>Total Diseases</p>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
                <div class="metric-card">
                    <h3>2</h3>
                    <p>AI Models</p>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
                <div class="metric-card">
                    <h3>84%</h3>
                    <p>Accuracy</p>
                </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown("""
                <div class="metric-card">
                    <h3>2552</h3>
                    <p>Images Trained</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Categorize diseases by plant type
        plant_categories = {
            'Tomato': [],
            'Potato': [],
            'Pepper': [],
            'Corn': [],
            'Apple': [],
            'Grape': [],
            'Other': []
        }
        
        for disease in diseases_to_show:
            disease_lower = disease.lower()
            categorized = False
            for plant in plant_categories.keys():
                if plant.lower() in disease_lower:
                    plant_categories[plant].append(disease)
                    categorized = True
                    break
            if not categorized:
                plant_categories['Other'].append(disease)
        
        # Display by category
        for category, diseases in plant_categories.items():
            if diseases:
                with st.expander(f"🌱 {category} Diseases ({len(diseases)})", expanded=False):
                    for disease in diseases:
                        disease_key = disease.lower().replace(' ', '_').replace('___', '_')
                        
                        # Find treatment info
                        treatment_info = None
                        for key, info in treatments_db.items():
                            if key in disease_key or disease_key in key:
                                treatment_info = info
                                break
                        
                        if treatment_info:
                            severity_color = {
                                'Severe': '#e74c3c',
                                'Moderate': '#f39c12',
                                'None - Healthy!': '#27ae60'
                            }.get(treatment_info.get('severity', 'Unknown'), '#95a5a6')
                            
                            st.markdown(f"""
                                <div class="treatment-card">
                                    <h4 style="color: {severity_color};">🦠 {disease}</h4>
                                    <p><strong>Severity:</strong> <span style="color: {severity_color};">{treatment_info.get('severity', 'Unknown')}</span></p>
                                    <p>{treatment_info.get('description', 'N/A')}</p>
                                    
                                    <details>
                                        <summary style="cursor: pointer; color: #11998e; font-weight: 600; margin: 1rem 0;">📋 View Full Details</summary>
                                        
                                        <h5 style="color: #11998e; margin-top: 1rem;">🔍 Symptoms:</h5>
                                        <ul>
                                            {''.join([f'<li>{s}</li>' for s in treatment_info.get('symptoms', [])])}
                                        </ul>
                                        
                                        <h5 style="color: #11998e; margin-top: 1rem;">💊 Treatment Options:</h5>
                                        <ul>
                                            {''.join([f'<li>{t}</li>' for t in treatment_info.get('treatments', [])])}
                                        </ul>
                                        
                                        <h5 style="color: #11998e; margin-top: 1rem;">🛡️ Prevention Measures:</h5>
                                        <ul>
                                            {''.join([f'<li>{p}</li>' for p in treatment_info.get('prevention', [])])}
                                        </ul>
                                    </details>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                                <div class="treatment-card">
                                    <h4>🦠 {disease}</h4>
                                    <p>No detailed information available yet.</p>
                                </div>
                            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div class="footer">
            <p>🌿 <strong>PlantDoc Bot</strong> - Powered by Deep Learning & NVIDIA AI</p>
            <p>Protecting crops, one diagnosis at a time 🌱</p>
            <p style="font-size: 0.8rem; margin-top: 1rem;">
                Built with ❤️ using TensorFlow, Streamlit & NVIDIA API<br>
                © 2025 PlantDoc Bot. All rights reserved.
            </p>
        </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()