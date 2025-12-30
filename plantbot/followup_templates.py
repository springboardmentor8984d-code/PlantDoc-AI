"""
Follow-up response templates for post-diagnosis conversation.
"""

FOLLOWUP_RESPONSES = {
    "general": [
        "I'm glad I could help diagnose the issue! Is there anything specific you'd like to know about the treatment or prevention?",
        "Happy to help! Would you like more details about how to apply the treatment, or do you have other questions?",
        "I'm here to help further! Do you need clarification on any of the treatment steps, or would you like to know about prevention?",
    ],
    
    "treatment_details": [
        "For the treatment I recommended, it's best to apply it {frequency}. Make sure to {application_tip}.",
        "The treatment should be applied carefully. {detailed_instructions}",
    ],
    
    "prevention": [
        "To prevent this from happening again, focus on: {prevention_tips}",
        "Prevention is key! Here's what you can do: {prevention_tips}",
    ],
    
    "timeline": [
        "Recovery typically takes {timeframe}, depending on the severity and how quickly you start treatment.",
        "You should see improvement within {timeframe} if you follow the treatment plan consistently.",
    ],
    
    "new_plant": [
        "Of course! I'd be happy to help with another plant. What symptoms are you seeing?",
        "Sure! Tell me about the new plant - what's happening with it?",
        "Absolutely! Describe the symptoms you're noticing on this plant.",
    ],
    
    "thanks": [
        "You're very welcome! Feel free to come back if you need help with other plants. Good luck! 🌱",
        "Happy to help! I hope your plant recovers quickly. Don't hesitate to return if you have more questions! 🌿",
        "Glad I could assist! Wishing your plant a speedy recovery! 🌱",
    ],
}


def get_followup_response(category: str = "general") -> str:
    """Get a random follow-up response for the given category."""
    import random
    responses = FOLLOWUP_RESPONSES.get(category, FOLLOWUP_RESPONSES["general"])
    return random.choice(responses)