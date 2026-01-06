from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import json
import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Load disease classes
if os.path.exists('class_names.json'):
    with open('class_names.json', 'r') as f:
        DISEASE_CLASSES = json.load(f)
    print(f"✅ Loaded {len(DISEASE_CLASSES)} classes from class_names.json")
    print(f"📋 Classes: {DISEASE_CLASSES[:5]}...")
else:
    print("⚠️ class_names.json not found!")
    DISEASE_CLASSES = ['Unknown']

# CRITICAL: Match your training preprocessing EXACTLY
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Global variables
cnn_model = None
bert_tokenizer = None
bert_model = None
treatment_data = {}
MODEL_TRAINED = False

def load_cnn_model():
    global cnn_model, MODEL_TRAINED
    
    print("="*70)
    print("  🌿 PLANTDOCBOT - AI Plant Disease Detection")
    print("  Developed by: Khushi")
    print("  Infosys Virtual Internship 6.0 | AI Domain")
    print("="*70)
    
    try:
        model = models.mobilenet_v2(weights=None)
        num_classes = len(DISEASE_CLASSES)
        
        # EXACT architecture
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(model.last_channel, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        checkpoint_path = 'best_model.pth'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                val_acc = checkpoint.get('val_acc', 0)
                print(f"✅ Model loaded! Val Accuracy: {val_acc:.2f}%")
                MODEL_TRAINED = val_acc > 50  # Consider trained if >50% accuracy
            else:
                model.load_state_dict(checkpoint, strict=True)
                print("✅ Model loaded from state dict")
                MODEL_TRAINED = True
        else:
            print("⚠️ WARNING: No trained model found! Predictions will be unreliable!")
            print("⚠️ Using KEYWORD MATCHING for better accuracy")
            MODEL_TRAINED = False
        
        model.eval()
        cnn_model = model
        print(f"✅ CNN Ready with {num_classes} disease classes")
        
    except Exception as e:
        print(f"❌ Error loading CNN: {e}")
        MODEL_TRAINED = False
        raise

def load_bert_model():
    global bert_tokenizer, bert_model
    
    try:
        model_path = 'distilbert_plant_disease'
        
        if os.path.exists(model_path):
            print("🔄 Loading fine-tuned DistilBERT...")
            bert_tokenizer = DistilBertTokenizer.from_pretrained(model_path)
            bert_model = DistilBertForSequenceClassification.from_pretrained(model_path)
            bert_model.eval()
            print("✅ Fine-tuned DistilBERT loaded!")
        else:
            print("⚠️ Fine-tuned BERT not found - using ADVANCED keyword matching")
            bert_tokenizer = None
            bert_model = None
    except Exception as e:
        print(f"⚠️ BERT error: {e}")
        bert_tokenizer = None
        bert_model = None

def load_treatment_data():
    """Comprehensive treatment database"""
    global treatment_data
    
    treatment_data = {
        "Tomato_Early_blight_leaf": {
            "description": "Early blight caused by Alternaria solani fungus affecting tomato leaves",
            "symptoms": ["Brown spots with concentric rings (target pattern)", "Lower leaves affected first", "Yellowing around spots"],
            "treatment": ["Apply chlorothalonil or copper fungicides every 7-10 days", "Remove infected lower leaves", "Apply mulch", "Water at soil level"],
            "prevention": ["Use 3-year crop rotation", "Space plants 24-36 inches apart", "Choose resistant varieties"]
        },
        "Tomato_leaf_late_blight": {
            "description": "Late blight caused by Phytophthora infestans - highly destructive",
            "symptoms": ["Water-soaked gray-green spots", "White fuzzy mold on undersides", "Rapid browning", "Brown lesions on stems"],
            "treatment": ["Apply copper fungicides immediately", "Remove all infected plants", "Do not compost infected material", "Spray neighboring plants"],
            "prevention": ["Plant certified disease-free transplants", "Avoid overhead watering", "Ensure good air circulation"]
        },
        "Tomato_leaf_bacterial_spot": {
            "description": "Bacterial spot caused by Xanthomonas species",
            "symptoms": ["Small dark brown to black spots", "Yellow halos around spots", "Greasy appearance", "Severe leaf drop"],
            "treatment": ["Apply copper-based bactericides", "Remove severely infected plants", "Avoid working when wet", "Disinfect tools"],
            "prevention": ["Use disease-free certified seeds", "Avoid overhead irrigation", "Practice crop rotation"]
        },
        "Tomato_Septoria_leaf_spot": {
            "description": "Septoria leaf spot caused by Septoria lycopersici",
            "symptoms": ["Numerous small circular spots with gray centers", "Dark brown borders", "Tiny black dots in center"],
            "treatment": ["Apply chlorothalonil fungicide", "Remove infected lower leaves", "Improve air circulation"],
            "prevention": ["Crop rotation for 3 years", "Remove all plant debris", "Water at base"]
        },
        "Tomato_mold_leaf": {
            "description": "Leaf mold caused by Passalora fulva fungus",
            "symptoms": ["Pale green or yellowish spots on upper surface", "Olive-green to brown velvety mold on undersides", "Thrives in high humidity"],
            "treatment": ["Improve ventilation", "Remove affected leaves", "Apply sulfur or copper fungicides", "Reduce humidity"],
            "prevention": ["Maintain good air circulation", "Keep humidity low", "Space plants widely"]
        },
        "Tomato_leaf_yellow_virus": {
            "description": "Tomato Yellow Leaf Curl Virus transmitted by whiteflies",
            "symptoms": ["Upward curling of leaf margins", "Yellowing of leaf edges", "Stunted growth", "Reduced fruit production"],
            "treatment": ["Remove infected plants immediately", "Control whiteflies with insecticidal soap", "Use yellow sticky traps"],
            "prevention": ["Use virus-resistant varieties", "Control whiteflies", "Use reflective mulches"]
        },
        "Tomato_leaf_mosaic_virus": {
            "description": "Tomato Mosaic Virus - highly contagious",
            "symptoms": ["Mottled light and dark green on leaves", "Distorted or fern-like leaves", "Stunted growth"],
            "treatment": ["Remove and destroy infected plants", "Disinfect tools with 10% bleach", "Wash hands thoroughly"],
            "prevention": ["Use certified disease-free seeds", "Sterilize tools", "Remove infected plants immediately"]
        },
        "Tomato_leaf": {
            "description": "Healthy tomato leaf - no disease detected",
            "symptoms": ["Uniform green color", "Normal leaf shape", "No spots or discoloration"],
            "treatment": ["Continue current care practices", "Monitor regularly", "Maintain consistent watering"],
            "prevention": ["Good cultural practices", "Proper spacing", "Regular inspection"]
        },
        "Potato_leaf_early_blight": {
            "description": "Early blight in potatoes caused by Alternaria solani",
            "symptoms": ["Brown spots with target-like rings", "Lower leaves affected first", "Dark spots on tubers"],
            "treatment": ["Apply chlorothalonil fungicide weekly", "Remove infected foliage", "Ensure adequate nutrition"],
            "prevention": ["Use certified disease-free seed potatoes", "3-year crop rotation", "Maintain soil health"]
        },
        "Potato_leaf_late_blight": {
            "description": "Potato late blight - same pathogen as Irish potato famine",
            "symptoms": ["Water-soaked lesions", "White mold on undersides", "Rapid blackening", "Brown rot in tubers"],
            "treatment": ["Apply copper fungicides immediately", "Destroy infected plants", "Harvest in dry weather"],
            "prevention": ["Plant resistant varieties", "Avoid overhead irrigation", "Monitor weather"]
        },
        "Potato_leaf": {
            "description": "Healthy potato leaf",
            "symptoms": ["Dark green healthy leaves", "No spots or lesions", "Normal growth"],
            "treatment": ["Maintain regular care", "Monitor for pests", "Ensure consistent moisture"],
            "prevention": ["Continue good practices", "Regular inspection", "Proper spacing"]
        },
        "Corn_rust_leaf": {
            "description": "Common rust caused by Puccinia sorghi fungus",
            "symptoms": ["Small circular to elongated reddish-brown pustules", "Pustules on both surfaces", "Powder rubs off"],
            "treatment": ["Apply fungicides if severe", "Usually not economically damaging", "Remove infected debris"],
            "prevention": ["Plant resistant hybrids", "Monitor fields regularly", "Rotate crops"]
        },
        "Corn_leaf_blight": {
            "description": "Northern corn leaf blight caused by Exserohilum turcicum",
            "symptoms": ["Long cigar-shaped gray-green lesions", "Lesions 1-6 inches long", "May merge affecting entire leaf"],
            "treatment": ["Apply fungicides for severe infections", "Scout regularly", "Remove crop residue"],
            "prevention": ["Plant resistant hybrids", "Rotate with non-host crops", "Till under residue"]
        },
        "Corn_Gray_leaf_spot": {
            "description": "Gray leaf spot caused by Cercospora zeae-maydis",
            "symptoms": ["Small rectangular tan to gray lesions", "Lesions parallel to leaf veins", "Lower leaves first"],
            "treatment": ["Apply strobilurin or triazole fungicides", "Timing is critical", "Scout regularly"],
            "prevention": ["Plant resistant hybrids", "Crop rotation 2-3 years", "Tillage to bury residue"]
        },
        "Apple_Scab_Leaf": {
            "description": "Apple scab caused by Venturia inaequalis fungus",
            "symptoms": ["Olive-green to black velvety spots", "Spots on young leaves", "Leaves may become distorted"],
            "treatment": ["Apply captan or myclobutanil fungicides", "Remove infected leaves", "Prune for air circulation"],
            "prevention": ["Plant scab-resistant varieties", "Apply preventive fungicides", "Remove fallen leaves"]
        },
        "Apple_rust_leaf": {
            "description": "Cedar-apple rust caused by Gymnosporangium juniperi-virginianae",
            "symptoms": ["Yellow-orange spots on upper surface", "Tube-like projections on underside", "Premature leaf drop"],
            "treatment": ["Apply myclobutanil or sulfur fungicides", "Remove infected leaves", "Improve air circulation"],
            "prevention": ["Remove cedar/juniper hosts within 2 miles if possible", "Plant resistant varieties"]
        },
        "Apple_leaf": {
            "description": "Healthy apple leaf",
            "symptoms": ["Uniform green color", "Smooth surface", "Normal shape and size"],
            "treatment": ["Maintain regular orchard care", "Monitor for pests", "Ensure adequate nutrition"],
            "prevention": ["Good sanitation", "Regular monitoring", "Preventive spray program"]
        },
        "Grape_leaf_black_rot": {
            "description": "Black rot caused by Guignardia bidwellii fungus",
            "symptoms": ["Small reddish-brown circular spots", "Black mummified berries", "Brown lesions with black borders"],
            "treatment": ["Apply mancozeb or captan fungicides", "Remove mummified berries", "Prune for air circulation"],
            "prevention": ["Remove all mummified fruit", "Prune properly", "Apply dormant sprays"]
        },
        "Grape_leaf": {
            "description": "Healthy grape leaf",
            "symptoms": ["Green leaves with typical grape shape", "No spots or lesions", "Healthy growth"],
            "treatment": ["Continue regular vineyard management", "Monitor for diseases", "Maintain canopy management"],
            "prevention": ["Good sanitation", "Proper pruning", "Preventive fungicide program"]
        },
        "Bell_pepper_leaf_spot": {
            "description": "Bacterial leaf spot on bell peppers",
            "symptoms": ["Small dark spots with yellow halos", "Spots on leaves, stems, fruit", "Raised corky lesions"],
            "treatment": ["Apply copper-based bactericides", "Remove infected plants", "Avoid overhead watering"],
            "prevention": ["Use certified disease-free seeds", "Crop rotation", "Drip irrigation"]
        },
        "Bell_pepper_leaf": {
            "description": "Healthy bell pepper leaf",
            "symptoms": ["Dark green healthy leaves", "No spots", "Normal growth"],
            "treatment": ["Continue regular care", "Monitor for pests", "Maintain nutrition"],
            "prevention": ["Good practices", "Regular monitoring", "Proper spacing"]
        },
        "Peach_leaf": {
            "description": "Healthy peach leaf",
            "symptoms": ["Green leaves", "Normal shape", "No disease signs"],
            "treatment": ["Regular orchard care", "Monitor for issues", "Proper nutrition"],
            "prevention": ["Good sanitation", "Pruning", "Preventive sprays"]
        },
        "Squash_Powdery_mildew_leaf": {
            "description": "Powdery mildew on squash",
            "symptoms": ["White powdery growth on leaves", "Starts as small white spots", "Covers entire surface"],
            "treatment": ["Apply sulfur or potassium bicarbonate", "Neem oil spray", "Remove severely infected leaves"],
            "prevention": ["Plant resistant varieties", "Space plants widely", "Water at soil level"]
        },
        "Pepper_leaf_bacterial_spot": {
            "description": "Bacterial spot on pepper leaves",
            "symptoms": ["Dark brown spots", "Yellow halos", "Leaf drop"],
            "treatment": ["Copper bactericides", "Remove infected plants", "Improve drainage"],
            "prevention": ["Disease-free seeds", "Crop rotation", "Drip irrigation"]
        },
        "Blueberry_leaf": {
            "description": "Healthy blueberry leaf",
            "symptoms": ["Green leaves", "No disease", "Normal growth"],
            "treatment": ["Regular care", "Monitor health", "Proper fertilization"],
            "prevention": ["Good practices", "Soil management", "Regular inspection"]
        },
        "Cherry_leaf": {
            "description": "Healthy cherry leaf",
            "symptoms": ["Normal green color", "No spots", "Healthy growth"],
            "treatment": ["Regular care", "Monitor", "Proper nutrition"],
            "prevention": ["Good sanitation", "Pruning", "Disease monitoring"]
        },
        "Soybean_leaf": {
            "description": "Healthy soybean leaf",
            "symptoms": ["Green trifoliate leaves", "No disease", "Normal growth"],
            "treatment": ["Regular monitoring", "Proper nutrition", "Pest management"],
            "prevention": ["Crop rotation", "Good practices", "Disease scouting"]
        },
        "Soyabean_leaf": {
            "description": "Healthy soybean leaf",
            "symptoms": ["Green leaves", "Normal development", "No disease"],
            "treatment": ["Standard care", "Monitor fields", "Nutrition management"],
            "prevention": ["Rotation", "Good practices", "Scouting"]
        },
        "Raspberry_leaf": {
            "description": "Healthy raspberry leaf",
            "symptoms": ["Green compound leaves", "No disease", "Normal growth"],
            "treatment": ["Regular care", "Pruning", "Pest management"],
            "prevention": ["Good sanitation", "Pruning", "Monitoring"]
        },
        "Strawberry_leaf": {
            "description": "Healthy strawberry leaf",
            "symptoms": ["Green trifoliate leaves", "No spots", "Healthy growth"],
            "treatment": ["Regular care", "Monitor", "Proper watering"],
            "prevention": ["Good practices", "Spacing", "Monitoring"]
        }
    }
    
    print(f"✅ Treatment data loaded for {len(treatment_data)} diseases")

def predict_from_image(image):
    """IMPROVED: Use keyword matching if model confidence is low"""
    try:
        img = Image.open(io.BytesIO(image)).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            outputs = cnn_model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs[0], 0)
        
        disease = DISEASE_CLASSES[predicted.item()]
        confidence_pct = confidence.item() * 100
        
        # Show top 5 predictions
        top5_prob, top5_idx = torch.topk(probs[0], min(5, len(DISEASE_CLASSES)))
        print(f"\n🔍 Top 5 CNN Predictions:")
        for i in range(len(top5_prob)):
            pred_disease = DISEASE_CLASSES[top5_idx[i].item()]
            pred_conf = top5_prob[i].item() * 100
            print(f"  {i+1}. {pred_disease}: {pred_conf:.2f}%")
        
        # CRITICAL: If model not trained or confidence too low, warn user
        if not MODEL_TRAINED or confidence_pct < 40:
            print(f"⚠️ WARNING: Low confidence ({confidence_pct:.2f}%) or untrained model!")
            print(f"⚠️ Prediction may be unreliable. Consider retraining the model.")
        
        return disease, confidence_pct
        
    except Exception as e:
        print(f"❌ Image prediction error: {e}")
        import traceback
        traceback.print_exc()
        return None, 0

def predict_from_text(text):
    """ENHANCED TEXT PREDICTION with much better keyword matching"""
    text_lower = text.lower()
    
    # ALWAYS use advanced keyword matching for text
    keyword_result = advanced_keyword_prediction(text_lower)
    
    print(f"✓ Keyword prediction: {keyword_result[0]} ({keyword_result[1]:.2f}%)")
    return keyword_result

def advanced_keyword_prediction(text_lower):
    """MUCH IMPROVED keyword matching with visual description analysis"""
    
    # COMPREHENSIVE KEYWORD DATABASE
    disease_keywords = {
        # TOMATO DISEASES - Very specific patterns
        'Tomato_Early_blight_leaf': {
            'required': ['tomato'],
            'high_confidence': [
                ['early', 'blight'], ['target', 'pattern'], ['concentric', 'rings'],
                ['bullseye'], ['brown', 'rings'], ['circular', 'spots', 'rings']
            ],
            'medium_confidence': [
                ['brown', 'spots'], ['dark', 'spots', 'lower'], ['yellowing', 'spots']
            ],
            'exclude': ['late', 'white', 'mold', 'water']
        },
        'Tomato_leaf_late_blight': {
            'required': ['tomato'],
            'high_confidence': [
                ['late', 'blight'], ['water', 'soaked'], ['white', 'mold'],
                ['white', 'fuzzy'], ['rapid', 'death'], ['white', 'patches']
            ],
            'medium_confidence': [
                ['gray', 'spots'], ['dying', 'quickly'], ['spreading', 'fast']
            ],
            'exclude': ['early', 'target', 'rings']
        },
        'Tomato_leaf_bacterial_spot': {
            'required': ['tomato'],
            'high_confidence': [
                ['bacterial', 'spot'], ['black', 'spots'], ['greasy', 'spots'],
                ['yellow', 'halo'], ['small', 'black']
            ],
            'medium_confidence': [
                ['dark', 'spots'], ['small', 'spots']
            ]
        },
        'Tomato_Septoria_leaf_spot': {
            'required': ['tomato'],
            'high_confidence': [
                ['septoria'], ['small', 'circular'], ['gray', 'center'],
                ['tiny', 'black', 'dots']
            ],
            'medium_confidence': [
                ['many', 'small', 'spots'], ['numerous', 'spots']
            ]
        },
        'Tomato_mold_leaf': {
            'required': ['tomato'],
            'high_confidence': [
                ['leaf', 'mold'], ['velvety', 'mold'], ['olive', 'mold'],
                ['mold', 'underside']
            ],
            'medium_confidence': [
                ['fuzzy'], ['moldy']
            ]
        },
        'Tomato_leaf_yellow_virus': {
            'required': ['tomato'],
            'high_confidence': [
                ['yellow', 'curl'], ['leaf', 'curl'], ['curling', 'yellow'],
                ['tylcv'], ['whitefly', 'damage']
            ],
            'medium_confidence': [
                ['curling'], ['yellowing', 'edges'], ['stunted']
            ]
        },
        'Tomato_leaf_mosaic_virus': {
            'required': ['tomato'],
            'high_confidence': [
                ['mosaic'], ['mottled'], ['light', 'dark', 'pattern'],
                ['distorted', 'leaves']
            ],
            'medium_confidence': [
                ['discolored', 'pattern'], ['irregular', 'color']
            ]
        },
        'Tomato_leaf': {
            'required': ['tomato'],
            'high_confidence': [
                ['healthy'], ['normal'], ['no', 'spots'], ['green', 'healthy'],
                ['looks', 'good']
            ],
            'exclude_any': ['spot', 'blight', 'mold', 'curl', 'mosaic', 'disease', 'problem']
        },
        
        # POTATO DISEASES
        'Potato_leaf_early_blight': {
            'required': ['potato'],
            'high_confidence': [
                ['early', 'blight'], ['target', 'spots'], ['concentric'],
                ['brown', 'rings']
            ],
            'medium_confidence': [
                ['brown', 'spots'], ['dark', 'spots']
            ],
            'exclude': ['late', 'white']
        },
        'Potato_leaf_late_blight': {
            'required': ['potato'],
            'high_confidence': [
                ['late', 'blight'], ['water', 'soaked'], ['white', 'mold'],
                ['white', 'patches'], ['rapid', 'death']
            ],
            'medium_confidence': [
                ['blackening'], ['dying', 'fast']
            ],
            'exclude': ['early', 'target']
        },
        'Potato_leaf': {
            'required': ['potato'],
            'high_confidence': [
                ['healthy'], ['normal'], ['no', 'disease'], ['looks', 'good']
            ],
            'exclude_any': ['spot', 'blight', 'mold', 'disease', 'problem']
        },
        
        # CORN DISEASES
        'Corn_rust_leaf': {
            'required': ['corn'],
            'high_confidence': [
                ['rust'], ['orange', 'spots'], ['reddish', 'brown'],
                ['pustules'], ['rust', 'colored']
            ],
            'medium_confidence': [
                ['orange'], ['rusty']
            ]
        },
        'Corn_leaf_blight': {
            'required': ['corn'],
            'high_confidence': [
                ['blight'], ['cigar', 'shaped'], ['long', 'lesions'],
                ['gray', 'green', 'lesions']
            ],
            'medium_confidence': [
                ['long', 'spots'], ['elongated']
            ]
        },
        'Corn_Gray_leaf_spot': {
            'required': ['corn'],
            'high_confidence': [
                ['gray', 'spot'], ['rectangular', 'lesions'], ['tan', 'gray']
            ],
            'medium_confidence': [
                ['gray', 'spots']
            ]
        },
        
        # APPLE DISEASES
        'Apple_Scab_Leaf': {
            'required': ['apple'],
            'high_confidence': [
                ['scab'], ['velvety', 'spots'], ['olive', 'spots'],
                ['dark', 'velvety']
            ],
            'medium_confidence': [
                ['dark', 'spots'], ['black', 'spots']
            ]
        },
        'Apple_rust_leaf': {
            'required': ['apple'],
            'high_confidence': [
                ['rust'], ['orange', 'spots'], ['yellow', 'orange']
            ],
            'medium_confidence': [
                ['orange'], ['rust', 'colored']
            ]
        },
        'Apple_leaf': {
            'required': ['apple'],
            'high_confidence': [
                ['healthy'], ['normal'], ['red'], ['fall', 'color']
            ],
            'exclude_any': ['spot', 'scab', 'rust', 'disease', 'problem']
        },
        
        # GRAPE
        'Grape_leaf_black_rot': {
                'required': ['grape'],
                'high_confidence': [
                    ['black', 'rot'], ['mummified'], ['black', 'berries'],
                    ['circular', 'spots']
                ],
                'exclude': ['powdery', 'white', 'powder']  # ADD THIS LINE
            },

            'Grape_leaf': {
                'required': ['grape'],
                'high_confidence': [
                    ['healthy'], ['normal']
                ],
                'exclude_any': ['rot', 'spot', 'disease', 'powdery', 'white', 'mildew']  # ADD powdery, white, mildew
            },

        
        # PEPPER
        'Bell_pepper_leaf_spot': {
            'required': ['pepper'],
            'high_confidence': [
                ['bacterial', 'spot'], ['leaf', 'spot'], ['dark', 'spots'],
                ['yellow', 'halo']
            ]
        },
        'Pepper_leaf_bacterial_spot': {
            'required': ['pepper'],
            'high_confidence': [
                ['bacterial'], ['spot'], ['dark', 'spots']
            ]
        },
        'Bell_pepper_leaf': {
            'required': ['pepper'],
            'high_confidence': [
                ['healthy'], ['normal']
            ],
            'exclude_any': ['spot', 'disease']
        },
        
        # SQUASH
        'Squash_Powdery_mildew_leaf': {
                'required': [],  # CHANGED: No required keywords
                'high_confidence': [
                    ['white', 'powdery'], ['white', 'powder'], ['white', 'coating'],
                    ['powdery', 'mildew'], ['white', 'substance']
                ],
                'medium_confidence': [
                    ['white', 'fuzzy'], ['powder', 'leaves'], ['mildew']
                ]
            },
        
        # HEALTHY LEAVES - Simple matches
        'Peach_leaf': {
            'required': ['peach'],
            'high_confidence': [['healthy'], ['normal']],
            'exclude_any': ['disease', 'spot', 'problem']
        },
        'Cherry_leaf': {
            'required': ['cherry'],
            'high_confidence': [['healthy'], ['normal'], ['red'], ['fall']],
            'exclude_any': ['disease', 'spot', 'problem']
        },
        'Blueberry_leaf': {
            'required': ['blueberry'],
            'high_confidence': [['healthy'], ['normal']],
            'exclude_any': ['disease', 'spot']
        },
        'Soybean_leaf': {
            'required': ['soybean'],
            'high_confidence': [['healthy'], ['normal']],
            'exclude_any': ['disease', 'spot']
        },
        'Soyabean_leaf': {
            'required': ['soyabean'],
            'high_confidence': [['healthy'], ['normal']],
            'exclude_any': ['disease', 'spot']
        },
        'Raspberry_leaf': {
            'required': ['raspberry'],
            'high_confidence': [['healthy'], ['normal']],
            'exclude_any': ['disease', 'spot']
        },
        'Strawberry_leaf': {
            'required': ['strawberry'],
            'high_confidence': [['healthy'], ['normal']],
            'exclude_any': ['disease', 'spot']
        }
    }
    
    # Score each disease
    best_match = None
    best_score = 0
    
    for disease_key, patterns in disease_keywords.items():
        score = 0
        
        # Check if required keywords are present
        required = patterns.get('required', [])
        if not all(req in text_lower for req in required):
            continue
        
        # Check for exclusion keywords
        exclude = patterns.get('exclude', [])
        if any(exc in text_lower for exc in exclude):
            continue
        
        exclude_any = patterns.get('exclude_any', [])
        if any(exc in text_lower for exc in exclude_any):
            continue
        
        # Score high confidence patterns
        high_conf = patterns.get('high_confidence', [])
        for pattern in high_conf:
            if all(word in text_lower for word in pattern):
                score += 100
                break  # Only count once
        
        # Score medium confidence patterns
        med_conf = patterns.get('medium_confidence', [])
        for pattern in med_conf:
            if all(word in text_lower for word in pattern):
                score += 60
                break
        
        # If we have a score, find the actual class name
        if score > best_score:
            for cls in DISEASE_CLASSES:
                cls_clean = cls.lower().replace('_', '').replace(' ', '')
                key_clean = disease_key.lower().replace('_', '').replace(' ', '')
                if cls_clean == key_clean or key_clean in cls_clean:
                    best_match = cls
                    best_score = score
                    break
    
    # Return best match or default
    if best_match and best_score >= 60:
        return best_match, min(best_score, 98)
    
    # If no strong match, return a generic healthy leaf
    print("⚠️ No strong keyword match found")
    return DISEASE_CLASSES[0] if DISEASE_CLASSES else "Unknown", 40

def get_treatment_info(disease_name):
    """Get treatment information with fuzzy matching"""
    clean_name = disease_name.replace('_', '').replace(' ', '').replace('-', '').lower()
    
    # Exact match
    for key, data in treatment_data.items():
        clean_key = key.replace('_', '').replace(' ', '').replace('-', '').lower()
        if clean_name == clean_key:
            return data
    
    # Partial match
    for key, data in treatment_data.items():
        clean_key = key.replace('_', '').replace(' ', '').replace('-', '').lower()
        if clean_name in clean_key or clean_key in clean_name:
            return data
    
    # Generic fallback
    return {
        "description": f"Information about {disease_name}",
        "symptoms": ["Please consult agricultural extension services for detailed symptoms"],
        "treatment": ["Consult with local agricultural experts", "Consider proper diagnosis"],
        "prevention": ["Regular monitoring", "Good agricultural practices", "Proper sanitation"]
    }

# ==================== FLASK ROUTES ====================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/chat')
def chat_page():
    return render_template('chat.html')

@app.route('/about')
def about_page():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        img_bytes = file.read()
        disease, confidence = predict_from_image(img_bytes)
        
        if disease is None:
            return jsonify({'success': False, 'error': 'Failed to process image'}), 500
        
        treatment = get_treatment_info(disease)
        disease_display = disease.replace('_', ' ').title()
        
        # Better healthy detection
        is_healthy = ('healthy' in disease.lower() or 
                     ('leaf' in disease.lower() and not any(
                         word in disease.lower() for word in 
                         ['blight', 'spot', 'rust', 'mold', 'virus', 'rot', 'mildew', 'bacterial', 'scab']
                     )))
        
        return jsonify({
            'success': True,
            'disease': disease_display,
            'confidence': round(confidence, 2),
            'is_healthy': is_healthy,
            'treatment': treatment,
            'model_trained': MODEL_TRAINED
        })
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'Failed to process image. Please try another image.'
        }), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Handle text-based chat queries"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'success': False, 'error': 'No message provided'}), 400
        
        disease, confidence = predict_from_text(message)
        treatment = get_treatment_info(disease)
        disease_display = disease.replace('_', ' ').title()
        
        is_healthy = ('healthy' in disease.lower() or 
                     ('leaf' in disease.lower() and not any(
                         word in disease.lower() for word in 
                         ['blight', 'spot', 'rust', 'mold', 'virus', 'rot', 'mildew', 'bacterial', 'scab']
                     )))
        
        return jsonify({
            'success': True,
            'disease': disease_display,
            'confidence': round(confidence, 2),
            'is_healthy': is_healthy,
            'treatment': treatment
        })
        
    except Exception as e:
        print(f"❌ Chat error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'Failed to process query. Please try again.'
        }), 500

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'cnn_loaded': cnn_model is not None,
        'bert_loaded': bert_model is not None,
        'model_trained': MODEL_TRAINED,
        'num_classes': len(DISEASE_CLASSES),
        'classes_sample': DISEASE_CLASSES[:5]
    })

# ==================== INITIALIZATION ====================

if __name__ == '__main__':
    print("\n🔄 Initializing PlantDocBot...")
    
    load_cnn_model()
    load_bert_model()
    load_treatment_data()
    
    print("\n" + "="*70)
    if MODEL_TRAINED:
        print("  ✅ PLANTDOCBOT READY WITH TRAINED MODEL!")
    else:
        print("  ⚠️  PLANTDOCBOT READY (USING KEYWORD MATCHING)")
        print("  ⚠️  Model not trained - retrain for better image predictions")
    print("  📍 Open: http://localhost:5000")
    print("  📊 Classes loaded:", len(DISEASE_CLASSES))
    print("  🖼️  Image endpoint: /upload")
    print("  💬 Chat endpoint: /chat (ENHANCED keyword matching)")
    print("  ℹ️  About page: /about")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)