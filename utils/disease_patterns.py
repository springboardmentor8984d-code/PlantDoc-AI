"""
Pattern-based disease detection rules.
Supplements BERT model with rule-based pattern matching for characteristic symptoms.
"""

# Disease pattern rules: each pattern is a dict with required keywords and disease name
DISEASE_PATTERNS = [
    {
        "disease": "Tomato Early Blight",
        "confidence_boost": 0.70,  # Add 70% confidence if pattern matches
        "required_all": ["circular", "concentric", "ring", "target"],  # Must have ALL
        "required_any": ["brown", "spot", "lower"],  # Must have at least ONE
        "plant_types": ["tomato"],
        "description": "Circular spots with concentric rings (target pattern) on lower leaves"
    },
    {
        "disease": "Potato Early Blight",
        "confidence_boost": 0.70,
        "required_all": ["circular", "concentric", "ring"],
        "required_any": ["brown", "spot", "lower", "target"],
        "plant_types": ["potato"],
        "description": "Target-spot lesions on lower leaves"
    },
    {
        "disease": "Tomato Late Blight",
        "confidence_boost": 0.85,  # High confidence - very serious disease
        "required_all": [],  # No required ALL - use flexible matching
        "required_any": [
            "water-soaked", "water soaked", "watersoaked",
            "white fungal", "white growth", "fungal growth",
            "rapid spread", "rapidly", "collapse", "wilt",
            "humid weather", "humid"
        ],
        "required_count": 2,  # Must have at least 2 of the above
        "plant_types": ["tomato"],
        "description": "Water-soaked lesions with white fungal growth, rapid spread"
    },
    {
        "disease": "Potato Late Blight",
        "confidence_boost": 0.85,
        "required_all": [],
        "required_any": [
            "water-soaked", "water soaked", "watersoaked",
            "white fungal", "white growth", "fungal growth",
            "rapid spread", "rapidly", "collapse"
        ],
        "required_count": 2,
        "plant_types": ["potato"],
        "description": "Water-soaked lesions spreading rapidly"
    },
    {
        "disease": "Squash Powdery Mildew",
        "confidence_boost": 0.80,
        "required_all": ["white", "powder"],
        "required_any": ["coating", "fuzzy", "leaves"],
        "plant_types": ["squash"],
        "description": "White powdery coating on leaves"
    },
    {
        "disease": "Tomato Septoria Spot",
        "confidence_boost": 0.65,
        "required_all": ["small", "circular", "dark"],
        "required_any": ["spot", "border", "lower"],
        "plant_types": ["tomato"],
        "description": "Small circular spots with dark borders on lower leaves"
    },
    {
        "disease": "Apple Scab",
        "confidence_boost": 0.70,
        "required_all": ["dark", "scab"],
        "required_any": ["olive", "velvety", "leaves", "fruit"],
        "plant_types": ["apple"],
        "description": "Dark, scabby lesions on leaves and fruit"
    },
    {
        "disease": "Grape Black Rot",
        "confidence_boost": 0.75,
        "required_all": ["black", "mummified"],
        "required_any": ["berries", "fruit", "shriveled"],
        "plant_types": ["grape"],
        "description": "Black, mummified berries"
    },
    {
        "disease": "Tomato Powdery Mildew",
        "confidence_boost": 0.85,
        "required_all": ["white", "powdery"],
        "required_any": ["coating", "patches", "spots", "leaves", "growth"],
        "plant_types": ["tomato"],
        "description": "White powdery patches or coating on leaves"
    },
]


def check_pattern_match(symptom_text: str, plant_type: str = None) -> tuple:
    """
    Check if symptom text matches any disease patterns.
    
    Args:
        symptom_text: Accumulated symptom description
        plant_type: Plant type if known
        
    Returns:
        Tuple of (matched_disease, confidence_boost, description) or (None, 0.0, None)
    """
    text_lower = symptom_text.lower()
    
    for pattern in DISEASE_PATTERNS:
        # Check plant type if specified
        if plant_type and plant_type.lower() not in pattern["plant_types"]:
            continue
        
        # Check if ALL required keywords are present (if any)
        if pattern.get("required_all"):
            has_all_required = all(keyword in text_lower for keyword in pattern["required_all"])
            if not has_all_required:
                continue
        
        # Check if at least ONE (or required_count) of the "any" keywords is present
        if pattern.get("required_any"):
            matching_keywords = [keyword for keyword in pattern["required_any"] if keyword in text_lower]
            required_count = pattern.get("required_count", 1)  # Default to 1 if not specified
            
            if len(matching_keywords) < required_count:
                continue
        
        # If we got here, pattern matches!
        return (
            pattern["disease"],
            pattern["confidence_boost"],
            pattern["description"]
        )
    
    return (None, 0.0, None)


def get_all_pattern_diseases():
    """Get list of all diseases that have pattern rules."""
    return [p["disease"] for p in DISEASE_PATTERNS]