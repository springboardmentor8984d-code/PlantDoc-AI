"""
Chatbot Manager for conversational plant disease diagnosis.
Handles conversation state, symptom extraction, and BERT-based diagnosis.
"""

import re
import torch
import json
from typing import List, Tuple, Optional, Dict
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import numpy as np

from .conversation_templates import (
    get_random_greeting,
    get_clarifying_question,
    get_diagnosis_template,
    get_follow_up_suggestion,
    get_encouragement,
    get_image_upload_suggestion,
    get_healthy_response,
    get_confusion_response,
    get_acknowledgment,
)
from knowledge_base.treatments import format_treatment_message, get_all_disease_names
from knowledge_base.disease_patterns import check_pattern_match
from .followup_templates import get_followup_response


class ConversationState:
    """Manages the state of a conversation."""
    
    def __init__(self):
        self.conversation_history: List[Dict[str, str]] = []
        self.accumulated_symptoms: str = ""
        self.plant_type: Optional[str] = None
        self.diagnosis_made: bool = False
        self.predicted_disease: Optional[str] = None
        self.confidence_score: float = 0.0
        self.questions_asked: List[str] = []
        self.user_message_count: int = 0
        
    def add_message(self, role: str, content: str):
        """Add a message to conversation history."""
        self.conversation_history.append({"role": role, "content": content})
        
    def add_symptoms(self, symptoms: str):
        """Add symptoms to accumulated symptoms."""
        if symptoms.strip():
            if self.accumulated_symptoms:
                self.accumulated_symptoms += " " + symptoms
            else:
                self.accumulated_symptoms = symptoms
                
    def reset(self):
        """Reset conversation state."""
        self.__init__()


class ChatbotManager:
    """Manages chatbot logic and BERT model integration."""
    
    # Confidence thresholds
    HIGH_CONFIDENCE = 0.80
    MEDIUM_CONFIDENCE = 0.60
    LOW_CONFIDENCE = 0.40
    
    # Keywords for symptom extraction
    SYMPTOM_KEYWORDS = [
        'spot', 'spots', 'yellow', 'yellowing', 'brown', 'browning', 'black', 'white',
        'curl', 'curling', 'wilt', 'wilting', 'rot', 'rotting', 'mold', 'moldy',
        'powder', 'powdery', 'rust', 'rusty', 'blight', 'leaf', 'leaves', 'stem',
        'fruit', 'lesion', 'lesions', 'discolor', 'discoloration', 'dying', 'dead',
        'hole', 'holes', 'eaten', 'chewed', 'sticky', 'fuzzy', 'coating', 'ring',
        'rings', 'target', 'circular', 'irregular', 'spreading', 'dried', 'dry',
        'drying', 'turning', 'starting', 'lower', 'upper', 'around', 'few'
    ]
    
    # Keywords to REMOVE (confuse the model)
    NEGATIVE_KEYWORDS = [
        'normal', 'healthy', 'fine', 'good', 'ok', 'okay', 'still look',
        'look normal', 'looks normal', 'appear normal', 'appears normal',
        'no problem', 'no issue', 'not affected', 'unaffected'
    ]
    
    # Plant types
    PLANT_TYPES = [
        'tomato', 'potato', 'apple', 'grape', 'corn', 'pepper', 'bell pepper',
        'cherry', 'peach', 'blueberry', 'raspberry', 'strawberry', 'squash',
        'soybean', 'soyabean'
    ]
    
    def __init__(self, model_path: str, label_mapping_path: str, device: str = "cpu"):
        """
        Initialize chatbot manager.
        
        Args:
            model_path: Path to BERT model file
            label_mapping_path: Path to label mapping JSON
            device: Device to run model on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        
        # Load label mapping
        with open(label_mapping_path, 'r') as f:
            label_map = json.load(f)
        self.label_map = {int(k): v for k, v in label_map.items()}
        
        # Load BERT model
        num_labels = len(self.label_map)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=num_labels
        )
        
        # Load trained weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        self.state = ConversationState()
        
    def generate_response(self, user_message: str) -> str:
        """
        Generate chatbot response to user message.
        
        Args:
            user_message: User's input message
            
        Returns:
            Bot's response
        """
        # Add user message to history
        self.state.add_message("user", user_message)
        self.state.user_message_count += 1
        
        # CRITICAL FIX: Check if diagnosis was already made
        if self.state.diagnosis_made:
            # Post-diagnosis mode: answer follow-up questions
            response = self._handle_followup(user_message)
        
        # First message - greet user
        elif self.state.user_message_count == 1:
            # Extract initial symptoms
            symptoms = self._extract_symptoms(user_message)
            plant_type = self._extract_plant_type(user_message)
            
            if symptoms:
                self.state.add_symptoms(symptoms)
            if plant_type:
                self.state.plant_type = plant_type
                
            # Check if we have enough info to diagnose
            if self._should_diagnose():
                response = self._make_diagnosis()
            else:
                # Ask clarifying questions
                response = self._ask_clarifying_questions()
                
        else:
            # Subsequent messages (still collecting symptoms)
            # Extract symptoms and plant type
            symptoms = self._extract_symptoms(user_message)
            plant_type = self._extract_plant_type(user_message)
            
            if symptoms:
                self.state.add_symptoms(symptoms)
            if plant_type and not self.state.plant_type:
                self.state.plant_type = plant_type
                
            # Check if we should diagnose now
            if self._should_diagnose():
                response = self._make_diagnosis()
            else:
                # Continue asking questions
                response = self._ask_clarifying_questions()
        
        # Add bot response to history
        self.state.add_message("bot", response)
        return response
    
    def _extract_symptoms(self, text: str) -> str:
        """Extract symptom-related text from user message."""
        text_lower = text.lower()
        
        # Check if message contains symptom keywords
        has_symptoms = any(keyword in text_lower for keyword in self.SYMPTOM_KEYWORDS)
        
        if has_symptoms:
            # Remove negative/confusing keywords that suggest health
            cleaned_text = text
            for neg_keyword in self.NEGATIVE_KEYWORDS:
                # Case-insensitive removal
                import re
                pattern = re.compile(re.escape(neg_keyword), re.IGNORECASE)
                cleaned_text = pattern.sub('', cleaned_text)
            
            # Remove extra spaces
            cleaned_text = ' '.join(cleaned_text.split())
            
            # Only return if there's still meaningful content
            if len(cleaned_text.strip()) > 10:
                return cleaned_text
            return text  # Return original if cleaning removed too much
        
        return ""
    
    def _extract_plant_type(self, text: str) -> Optional[str]:
        """Extract plant type from user message."""
        text_lower = text.lower()
        
        for plant in self.PLANT_TYPES:
            if plant in text_lower:
                return plant.title()
        
        return None
    
    def _should_diagnose(self) -> bool:
        """
        Determine if we have enough information to make a diagnosis.
        
        Uses hybrid approach:
        - If we have good symptom description and confidence is high, diagnose
        - If confidence is low, ask more questions
        - After 3-4 exchanges, attempt diagnosis even if confidence is medium
        """
        # Need at least some symptoms
        if not self.state.accumulated_symptoms.strip():
            return False
        
        # Get preliminary prediction
        disease, confidence, _ = self._predict_disease(self.state.accumulated_symptoms)
        
        # High confidence - diagnose immediately
        if confidence >= self.HIGH_CONFIDENCE:
            return True
        
        # After several exchanges, diagnose even with medium confidence
        if self.state.user_message_count >= 4 and confidence >= self.MEDIUM_CONFIDENCE:
            return True
        
        # After many exchanges, diagnose regardless (avoid infinite loop)
        if self.state.user_message_count >= 6:
            return True
        
        # Otherwise, ask more questions
        return False
    
    def _predict_disease(self, symptom_text: str) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Predict disease using BERT model.
        
        Args:
            symptom_text: Text describing symptoms
            
        Returns:
            Tuple of (top_disease, top_confidence, top_3_predictions)
        """
        # Tokenize
        encoding = self.tokenizer(
            symptom_text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs.logits
            probs = softmax(logits, dim=1).cpu().numpy()[0]
        
        # Get top 3 predictions
        top_3_indices = np.argsort(probs)[-3:][::-1]
        top_3_predictions = [
            (self.label_map[idx], float(probs[idx]))
            for idx in top_3_indices
        ]
        
        top_disease = top_3_predictions[0][0]
        top_confidence = top_3_predictions[0][1]
        
        return top_disease, top_confidence, top_3_predictions
    
    def _ask_clarifying_questions(self) -> str:
        """Generate clarifying questions based on what we know."""
        response_parts = []
        
        # Add encouragement if not first message
        if self.state.user_message_count > 1:
            response_parts.append(get_encouragement())
        
        # Determine what questions to ask
        questions_to_ask = []
        
        # Ask about plant type if unknown
        if not self.state.plant_type and 'plant_type' not in self.state.questions_asked:
            questions_to_ask.append(get_clarifying_question('plant_type'))
            self.state.questions_asked.append('plant_type')
        
        # Ask about symptom details
        if 'symptom_color' not in self.state.questions_asked:
            questions_to_ask.append(get_clarifying_question('symptom_color'))
            self.state.questions_asked.append('symptom_color')
        elif 'symptom_location' not in self.state.questions_asked:
            questions_to_ask.append(get_clarifying_question('symptom_location'))
            self.state.questions_asked.append('symptom_location')
        elif 'symptom_pattern' not in self.state.questions_asked:
            questions_to_ask.append(get_clarifying_question('symptom_pattern'))
            self.state.questions_asked.append('symptom_pattern')
        elif 'leaf_position' not in self.state.questions_asked:
            questions_to_ask.append(get_clarifying_question('leaf_position'))
            self.state.questions_asked.append('leaf_position')
        elif 'additional_symptoms' not in self.state.questions_asked:
            questions_to_ask.append(get_clarifying_question('additional_symptoms'))
            self.state.questions_asked.append('additional_symptoms')
        
        # Add questions to response
        if questions_to_ask:
            response_parts.append(" ".join(questions_to_ask))
        else:
            # Fallback - ask general question
            response_parts.append("Can you provide any other details about what you're seeing?")
        
        return "\n\n".join(response_parts)
    
    def _make_diagnosis(self) -> str:
        """Make final diagnosis and provide treatment recommendations."""
        # Get BERT prediction
        disease, confidence, top_3 = self._predict_disease(self.state.accumulated_symptoms)
        
        # NEW: Check for pattern-based detection (characteristic symptoms)
        pattern_disease, pattern_boost, pattern_desc = check_pattern_match(
            self.state.accumulated_symptoms,
            self.state.plant_type
        )
        
        # If we have a pattern match, use it!
        if pattern_disease:
            # Pattern match found - boost confidence significantly
            disease = pattern_disease
            confidence = min(pattern_boost, 0.95)  # Cap at 95%
            print(f"[DEBUG] Pattern match: {pattern_disease} ({confidence*100:.1f}%)")
        else:
            # No pattern match - use BERT prediction with improvements
            
            # IMPROVEMENT: If top prediction is "healthy" but there's a disease prediction close behind,
            # prefer the disease (since user is asking for help, likely has a problem)
            if 'healthy' in disease.lower() and len(top_3) >= 2:
                # Check if second prediction is a disease and close in confidence
                second_disease, second_conf = top_3[1]
                if 'healthy' not in second_disease.lower():
                    # If within 5% confidence, prefer the disease
                    if (confidence - second_conf) < 0.05:
                        disease = second_disease
                        confidence = second_conf
                        # Also check third prediction
                        if len(top_3) >= 3:
                            third_disease, third_conf = top_3[2]
                            if 'healthy' not in third_disease.lower() and third_conf > second_conf:
                                disease = third_disease
                                confidence = third_conf
        
        # Store in state
        self.state.diagnosis_made = True
        self.state.predicted_disease = disease
        self.state.confidence_score = confidence
        
        # Check if it's a healthy plant (after prioritization)
        if 'healthy' in disease.lower():
            return get_healthy_response()
        
        # Format diagnosis message
        confidence_pct = confidence * 100
        template = get_diagnosis_template(confidence)
        diagnosis_msg = template.format(disease=disease, confidence=f"{confidence_pct:.1f}")
        
        # Get treatment information
        treatment_msg = format_treatment_message(disease, confidence)
        
        # Add follow-up suggestion
        follow_up = get_follow_up_suggestion()
        
        # Combine all parts
        full_response = f"{diagnosis_msg}\n\n{treatment_msg}{follow_up}"
        
        # If confidence is not very high, suggest image upload
        if confidence < self.HIGH_CONFIDENCE:
            full_response += get_image_upload_suggestion()
        
        return full_response
    
    def _handle_followup(self, user_message: str) -> str:
        """
        Handle follow-up questions after diagnosis has been made.
        
        Args:
            user_message: User's follow-up question
            
        Returns:
            Appropriate response without repeating diagnosis
        """
        text_lower = user_message.lower()
        
        # Check for thank you / goodbye
        if any(word in text_lower for word in ['thank', 'thanks', 'appreciate', 'bye', 'goodbye']):
            return get_followup_response("thanks")
        
        # Check for new plant / different plant
        if any(phrase in text_lower for phrase in ['new plant', 'another plant', 'different plant', 'other plant']):
            # Reset state for new diagnosis
            self.state.reset()
            return get_followup_response("new_plant")
        
        # Check for timeline / recovery questions (CHECK FIRST - more specific)
        if any(word in text_lower for word in ['long', 'time', 'recover', 'recovery', 'heal', 'timeline']):
            return "Recovery time varies depending on disease severity and how quickly treatment starts. Most fungal diseases show improvement within 1-2 weeks with proper treatment. Keep monitoring your plant and continue treatment as recommended. Would you like more specific guidance?"
        
        # Check for prevention questions (CHECK BEFORE treatment - more specific)
        if any(word in text_lower for word in ['prevent', 'prevention', 'avoid', 'stop', 'future', 'next season', 'returning']):
            disease = self.state.predicted_disease
            if disease:
                from knowledge_base.treatments import TREATMENTS
                # Use disease name directly - TREATMENTS uses original names with spaces
                if disease in TREATMENTS:
                    prevention = TREATMENTS[disease].get("prevention", "Practice good plant hygiene.")
                    return f"🛡️ **Prevention Tips for {disease}:**\n\n{prevention}\n\nAnything else you'd like to know?"
                else:
                    return get_followup_response("general")
            else:
                return get_followup_response("general")
        
        # Check for questions about spreading
        if any(word in text_lower for word in ['spread', 'spreading', 'contagious', 'other plants', 'infect']):
            return "Yes, most plant diseases can spread to nearby plants. I recommend:\n\n1. Isolate the affected plant if possible\n2. Sanitize tools between plants\n3. Avoid overhead watering\n4. Remove and dispose of infected leaves\n5. Monitor nearby plants closely\n\nWould you like more prevention tips?"
        
        
        # Check for treatment/application details (CHECK LAST - most general)
        # Includes questions about specific fungicides or application methods
        if any(phrase in text_lower for phrase in [
            'how to apply', 'how often', 'how do i', 'how should', 'apply the', 'use the',
            'which fungicide', 'what fungicide', 'best fungicide', 'fungicides work',
            'which treatment', 'what treatment', 'recommend', 'suggest'
        ]):
            disease = self.state.predicted_disease
            if disease:
                # Provide more detailed application instructions
                return f"For treating {disease}:\n\n1. **Fungicide Application**: Apply fungicides like chlorothalonil or mancozeb every 7-10 days\n2. **Timing**: Start at first sign of disease or preventively in humid conditions\n3. **Coverage**: Spray thoroughly, covering both top and bottom of leaves\n4. **Safety**: Wear gloves and follow label instructions\n5. **Removal**: Remove and destroy infected leaves before spraying\n\nContinue treatment for 2-3 weeks even if symptoms improve. Need more details?"
            else:
                return get_followup_response("general")
        
        # Default: offer to help further
        return get_followup_response("general")
    
    def reset_conversation(self):
        """Reset the conversation state."""
        self.state.reset()
        
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return self.state.conversation_history