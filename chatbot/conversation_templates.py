"""
Conversation templates for natural chatbot interactions.
Contains greetings, questions, and response templates.
"""

import random

# Greeting messages
GREETINGS = [
    "Hello! 🌿 I'm your plant disease assistant. Describe the symptoms you're seeing on your plant, and I'll help diagnose the issue.",
    "Hi there! 👋 Tell me what's wrong with your plant - describe the symptoms, and I'll help identify the problem.",
    "Welcome! 🌱 I can help diagnose plant diseases. What symptoms are you noticing on your plant?",
]

# Clarifying questions organized by category
CLARIFYING_QUESTIONS = {
    "plant_type": [
        "What type of plant is affected? (e.g., tomato, apple, potato)",
        "Which plant are we dealing with?",
        "Can you tell me what kind of plant this is?",
    ],
    "symptom_location": [
        "Where on the plant do you see these symptoms? (leaves, stems, fruit, roots)",
        "Which part of the plant is affected?",
        "Are the symptoms on the leaves, stems, or fruit?",
    ],
    "symptom_color": [
        "What color are the spots or marks? (brown, yellow, white, black, etc.)",
        "Can you describe the color of the affected areas?",
        "What color are the symptoms?",
    ],
    "symptom_pattern": [
        "Do the spots have any particular pattern? (circular, irregular, ring-like)",
        "Are the spots circular with rings, or more irregular in shape?",
        "Can you describe the shape or pattern of the symptoms?",
    ],
    "symptom_spread": [
        "Is the problem spreading or staying in one area?",
        "Are the symptoms getting worse or spreading to other parts?",
        "Is this affecting just one area or spreading across the plant?",
    ],
    "leaf_position": [
        "Are the symptoms on older leaves (bottom of plant) or newer leaves (top)?",
        "Which leaves are affected - the older ones at the bottom or newer ones at the top?",
        "Is this happening on the lower, middle, or upper leaves?",
    ],
    "timing": [
        "When did you first notice these symptoms?",
        "How long has this been going on?",
        "Is this a recent problem or has it been developing for a while?",
    ],
    "texture": [
        "Is there any unusual texture? (powdery, fuzzy, slimy, dry)",
        "Does the affected area feel different? (rough, soft, powdery)",
        "Is there any coating or unusual texture on the symptoms?",
    ],
    "additional_symptoms": [
        "Are there any other symptoms? (wilting, yellowing, curling, holes)",
        "What else are you noticing? Any wilting, discoloration, or deformation?",
        "Are there any other changes to the plant besides what you mentioned?",
    ]
}

# Response templates for different confidence levels
DIAGNOSIS_TEMPLATES = {
    "high_confidence": [
        "Based on your description, I'm {confidence}% confident this is **{disease}**.",
        "This looks like **{disease}** (confidence: {confidence}%).",
        "I'm {confidence}% certain you're dealing with **{disease}**.",
    ],
    "medium_confidence": [
        "This appears to be **{disease}** (confidence: {confidence}%), though I'd like to gather a bit more information to be sure.",
        "Based on what you've told me, this could be **{disease}** ({confidence}% confidence). Let me ask a few more questions to confirm.",
        "I'm {confidence}% confident this is **{disease}**, but let me verify with a couple more questions.",
    ],
    "low_confidence": [
        "I need more specific information to make an accurate diagnosis. Let me ask you some questions.",
        "To give you the best diagnosis, I need a few more details about the symptoms.",
        "I want to make sure I give you the right diagnosis. Can you provide some additional information?",
    ],
    "need_more_info": [
        "Thanks for that information. Let me ask a few more questions to narrow this down.",
        "That's helpful! A few more details will help me make an accurate diagnosis.",
        "Good information. Let me gather a bit more detail to be certain.",
    ]
}

# Follow-up suggestions after diagnosis
FOLLOW_UP_SUGGESTIONS = [
    "\n\nWould you like more information about this disease, or do you have other plants to diagnose?",
    "\n\nIs there anything else you'd like to know about treating this condition?",
    "\n\nDo you have any questions about the treatment or prevention methods?",
    "\n\nCan I help you with anything else related to this plant or other plants?",
]

# Encouragement messages when asking for more info
ENCOURAGEMENT = [
    "Great, that helps!",
    "Thanks for the details!",
    "Perfect, that's useful information.",
    "Excellent, I'm getting a clearer picture.",
    "That's very helpful!",
]

# Image upload suggestions (for low confidence)
IMAGE_UPLOAD_SUGGESTIONS = [
    "\n\n💡 **Tip:** For the most accurate diagnosis, you could also upload a photo in the 'Image Diagnosis' tab.",
    "\n\n📸 **Suggestion:** If you have a photo of the affected plant, uploading it in the 'Image Diagnosis' tab can help confirm the diagnosis.",
    "\n\n💡 **Note:** A photo would help me give you a more accurate diagnosis. Check out the 'Image Diagnosis' tab!",
]

# Healthy plant responses
HEALTHY_RESPONSES = [
    "Good news! Based on your description, your plant appears to be **healthy**. Keep up the good care practices!",
    "Your plant sounds healthy! Continue with your current care routine and monitor regularly for any changes.",
    "Great! It seems your plant is in good health. Maintain proper watering, nutrition, and sanitation to keep it that way.",
]

# Error/confusion responses
CONFUSION_RESPONSES = [
    "I'm not quite sure I understand. Could you describe the symptoms you're seeing on your plant?",
    "I'd love to help! Can you tell me more about what's happening with your plant?",
    "Let's start over. What symptoms are you noticing on your plant?",
]

# Acknowledgment of user providing info
ACKNOWLEDGMENTS = [
    "Got it.",
    "I see.",
    "Understood.",
    "Okay.",
    "Thanks.",
]


def get_random_greeting() -> str:
    """Get a random greeting message."""
    return random.choice(GREETINGS)


def get_clarifying_question(category: str) -> str:
    """Get a random clarifying question from a specific category."""
    if category in CLARIFYING_QUESTIONS:
        return random.choice(CLARIFYING_QUESTIONS[category])
    return "Can you provide more details about the symptoms?"


def get_diagnosis_template(confidence: float) -> str:
    """
    Get appropriate diagnosis template based on confidence level.
    
    Args:
        confidence: Confidence score (0-1)
        
    Returns:
        Template string
    """
    if confidence >= 0.8:
        return random.choice(DIAGNOSIS_TEMPLATES["high_confidence"])
    elif confidence >= 0.6:
        return random.choice(DIAGNOSIS_TEMPLATES["medium_confidence"])
    else:
        return random.choice(DIAGNOSIS_TEMPLATES["low_confidence"])


def get_follow_up_suggestion() -> str:
    """Get a random follow-up suggestion."""
    return random.choice(FOLLOW_UP_SUGGESTIONS)


def get_encouragement() -> str:
    """Get a random encouragement message."""
    return random.choice(ENCOURAGEMENT)


def get_image_upload_suggestion() -> str:
    """Get a random image upload suggestion."""
    return random.choice(IMAGE_UPLOAD_SUGGESTIONS)


def get_healthy_response() -> str:
    """Get a random healthy plant response."""
    return random.choice(HEALTHY_RESPONSES)


def get_confusion_response() -> str:
    """Get a random confusion response."""
    return random.choice(CONFUSION_RESPONSES)


def get_acknowledgment() -> str:
    """Get a random acknowledgment."""
    return random.choice(ACKNOWLEDGMENTS)
