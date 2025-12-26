from .detailed_treatments import DETAILED_TREATMENTS
"""
Treatment database for plant diseases.
Contains detailed treatment, prevention, and severity information for all supported diseases.
"""

TREATMENTS = {
    "Apple": {
        "description": "General apple tree health issues including spots and defoliation",
        "treatment": "1. **Inspection**: Identify specific symptoms (spots vs. rot).\n2. **Pruning**: Remove any dead or diseased wood.\n3. **Sprays**: Apply a general purpose fruit tree spray schedule.",
        "prevention": "- Plant disease-resistant varieties.\n- Rake up fallen leaves in autumn.\n- Prune annually to open center.",
        "severity": "moderate"
    },
    "Apple Rust": {
        "description": "Fungal disease causing orange/rust-colored spots on leaves",
        "treatment": "1. **Host Removal**: Remove nearby red cedar/juniper trees (alternate hosts).\n2. **Fungicides**: Apply Myclobutanil or sulfur when blossoms show pink.\n3. **Monitoring**: Watch for orange gelatinous galls on junipers in spring.",
        "prevention": "- Plant rust-resistant apple varieties.\n- Remove galls from nearby junipers.\n- Apply protective sprays early.",
        "severity": "moderate"
    },
    "Apple Scab": {
        "description": "Fungal disease causing dark, scabby lesions on leaves and fruit",
        "treatment": "1. **Sanitation**: Rake and destroy ALL fallen leaves (source of spores).\n2. **Fungicides**: Apply captan or sulfur from green tip through petal fall.\n3. **Pruning**: Open up trees to allow leaves to dry quickly.",
        "prevention": "- Plant scab-immune varieties (e.g., 'Liberty', 'Freedom').\n- Keep grass mowed short.\n- Apply urea to fallen leaves to speed decomposition.",
        "severity": "high"
    },
    "Bell Pepper": {
        "description": "General bell pepper plant health issues",
        "treatment": "Ensure roots are not waterlogged. Check for aphids or mites.",
        "prevention": "Water consistently and mulch.",
        "severity": "low"
    },
    "Bell Pepper Spot": {
        "description": "Bacterial or fungal spots on bell pepper leaves and fruit",
        "treatment": "1. **Removal**: Pick off infected leaves immediately.\n2. **Copper**: Spray with fixed copper bactericide.\n3. **Watering**: Do not water from above.",
        "prevention": "- Use hot-water treated seeds.\n- Rotate peppers every 3 years.\n- Control weeds (hosts).",
        "severity": "moderate"
    },
    "Blueberry": {
        "description": "General blueberry plant health issues",
        "treatment": "1. **Soil Check**: Ensure pH is 4.5-5.0.\n2. **Pruning**: Remove oldest canes (>6 years old).\n3. **Mulching**: Maintain 4-6 inches of sawdust mulch.",
        "prevention": "- Acidify soil with sulfur.\n- Plant in raised beds for drainage.\n- Protect from birds.",
        "severity": "moderate"
    },
    "Cherry": {
        "description": "Cherry tree diseases including fungal infections",
        "treatment": "1. **Pruning**: Remove black knot galls (cut 6 inches below knot).\n2. **Sanitation**: Remove all brown rotted fruit.\n3. **Spraying**: Apply fungicide during bloom for brown rot.",
        "prevention": "- Prune in late winter.\n- Apply dormant oil for pests.\n- Harvest fruit promptly.",
        "severity": "moderate"
    },
    "Corn Blight": {
        "description": "Fungal disease causing leaf lesions and reduced yield",
        "treatment": "1. **Fungicide**: Apply if disease reaches ear leaf during tasseling.\n2. **Rotation**: Do not plant corn after corn.\n3. **Tillage**: Bury residue to reduce inoculum.",
        "prevention": "- Plant resistant hybrids.\n- Rotate with soybeans.\n- Control weeds.",
        "severity": "high"
    },
    "Corn Gray Spot": {
        "description": "Fungal disease causing gray rectangular lesions on leaves",
        "treatment": "1. **Threshold**: Treat if lesions cover >5% of ear leaf area.\n2. **Fungicides**: Strobilurins and triazoles are effective.\n3. **Management**: Improve residue management.",
        "prevention": "- Use resistant hybrids (most effective).\n- Rotate away from corn for 2 years.\n- Harvest early if severe.",
        "severity": "moderate"
    },
    "Corn Rust": {
        "description": "Fungal disease with orange-brown pustules on leaves",
        "treatment": "1. **Monitoring**: Check leaves for pustules before pollination.\n2. **Fungicide**: Apply at first sign if weather is cool/humid.\n3. **Cultural**: Avoid high nitrogen late in season.",
        "prevention": "- Plant rust-resistant corn varieties.\n- Apply fungicide early if history of rust.\n- Scout fields weekly.",
        "severity": "moderate"
    },
    "Grape": {
        "description": "General grape vine health issues",
        "treatment": "Check for mildew or rot. Ensure trellis allows good light penetration.",
        "prevention": "Shoot thinning and leaf pulling.",
        "severity": "low"
    },
    "Grape Black Rot": {
        "description": "Fungal disease causing fruit mummification and leaf spots",
        "treatment": "1. **Sanitation**: Remove and destroy all mummified berries and infected shoots (do not compost).\n2. **Fungicide**: Apply effective fungicides (e.g., myclobutanil, mancozeb) starting at bud break.\n3. **Airflow**: Prune heavily to open the canopy and expose fruit to sunlight and air.\n4. **Timing**: Treatment during bloom and early fruit set is most critical.",
        "prevention": "- **Remove Mummies**: In winter, remove all old berries from vines and ground.\n- **Canopy Management**: Keep vines open to reduce humidity.\n- **Resistant Varieties**: Plant resistant cultivars like Norton or Cynthiana.\n- **Weed Control**: Keep area around vines clear to reduce moisture.",
        "severity": "high"
    },
    "Peach": {
        "description": "Peach tree diseases including leaf curl and brown rot",
        "treatment": "1. **Pruning**: Remove infected material immediately.\n2. **Spraying**: Apply fungicide sprays during susceptible periods (pre-bloom and post-bloom).\n3. **Airflow**: Prune centers of trees to facilitate drying.",
        "prevention": "- Apply dormant sprays for leaf curl.\n- Thin fruit to prevent touching.\n- Remove all dropped fruit promptly.",
        "severity": "moderate"
    },
    "Pepper Bell Bacterial Spot": {
        "description": "Bacterial disease causing dark spots on leaves and fruit",
        "treatment": "1. **Removal**: Remove infected plants immediately to stop spread.\n2. **Copper Sprays**: Apply copper-based bactericides early.\n3. **Avoid Water**: Do NOT use overhead irrigation.",
        "prevention": "- Use certified disease-free seeds.\n- Rotate crops (3-4 years).\n- Use resistant varieties like 'Revolution'.",
        "severity": "high"
    },
    "Pepper Bell Healthy": {
        "description": "Healthy bell pepper plant with no disease symptoms",
        "treatment": "Keep up the good work! Maintain consistent watering and fertilization.",
        "prevention": "Continue monitoring for pests and diseases weekly.",
        "severity": "none"
    },
    "Potato Early Blight": {
        "description": "Fungal disease with target-spot lesions on lower leaves",
        "treatment": "1. **Remove Foliage**: Remove lower infected leaves.\n2. **Fungicides**: Apply chlorothalonil or mancozeb when symptoms appear.\n3. **Watering**: Water at the base, not overhead.",
        "prevention": "- Rotate crops every 2-3 years.\n- Maintain plant vigor with proper nitrogen.\n- Mulch to prevent soil splash.",
        "severity": "moderate"
    },
    "Potato Healthy": {
        "description": "Healthy potato plant with no disease symptoms",
        "treatment": "No treatment needed. Ensure hills are well-formed to protect tubers.",
        "prevention": "Monitor for beetles and blight regularly.",
        "severity": "none"
    },
    "Potato Late Blight": {
        "description": "Devastating fungal disease causing rapid plant death",
        "treatment": "1. **Immediate Action**: Destroy infected plants immediately (bag and trash, do not compost).\n2. **Protection**: Apply fungicides to healthy neighboring plants.\n3. **Isolate**: Stop all watering to reduce spread.",
        "prevention": "- Use certified seed potatoes.\n- Destroy volunteer potatoes.\n- Monitor local disease forecasts (Blitecast).",
        "severity": "critical"
    },
    "Raspberry": {
        "description": "Raspberry cane diseases and general health issues",
        "treatment": "1. **Pruning**: Remove old fruiting canes after harvest.\n2. **Airflow**: Thin rows to 1.5-2 feet wide.\n3. **Sanitation**: Remove wild brambles nearby.",
        "prevention": "- Plant certified disease-free stock.\n- Ensure good drainage.\n- Avoid excessive nitrogen.",
        "severity": "moderate"
    },
    "Soyabean": {
        "description": "Soybean leaf diseases and defoliation issues",
        "treatment": "1. **Monitor**: Check threshold for defoliation (apply fungicide if >15% in pod fill).\n2. **Rotation**: Rotate with non-legume crops.",
        "prevention": "- Use resistant varieties.\n- Plant in warm soils.\n- Tillage may help bury residue.",
        "severity": "moderate"
    },
    "Squash Powdery Mildew": {
        "description": "Fungal disease with white powdery coating on leaves",
        "treatment": "1. **Removal**: Remove severely infected leaves.\n2. **Sprays**: Apply neem oil, sulfur, or potassium bicarbonate.\n3. **Coverage**: Spray both top and bottom of leaves.",
        "prevention": "- Space plants for air circulation.\n- Plant resistant varieties (e.g., 'Success PM').\n- Water in morning only.",
        "severity": "moderate"
    },
    "Strawberry": {
        "description": "Strawberry plant diseases including leaf spots and fruit rots",
        "treatment": "1. **Sanitation**: Remove infected leaves and rotting fruit.\n2. **Drainage**: Ensure soil drains well.\n3. **Renovation**: Mow foliage after harvest.",
        "prevention": "- Use straw mulch to keep fruit of soil.\n- Plant resistant varieties.\n- Avoid overhead watering.",
        "severity": "moderate"
    },
    "Tomato": {
        "description": "General tomato plant health issues",
        "treatment": "Inspect carefully. If spots are target-shaped (Early Blight) or water-soaked (Late Blight), see specific treatments.",
        "prevention": "Stake plants, mulch, and water at the base.",
        "severity": "low"
    },
    "Tomato Bacterial Spot": {
        "description": "Bacterial disease causing dark spots with yellow halos",
        "treatment": "1. **Copper**: Apply copper + mancozeb sprays.\n2. **Sanitation**: Remove infected plant parts.\n3. **Dryness**: Avoid working in wet plants.",
        "prevention": "- Use clean seeds.\n- Sanitize tools/stakes.\n- 3-year crop rotation.",
        "severity": "high"
    },
    "Tomato Early Blight": {
        "description": "Fungal disease with target-spot lesions on lower leaves",
        "treatment": "1. **Pruning**: Remove lower 6-10 inches of leaves to improve airflow.\n2. **Fungicides**: Apply chlorothalonil every 7-10 days.\n3. **Mulching**: Apply mulch to stop soil splashing.",
        "prevention": "- Stake or cage plants.\n- Water at base only.\n- Rotate tomato location yearly.",
        "severity": "moderate"
    },
    "Tomato Healthy": {
        "description": "Healthy tomato plant with no disease symptoms",
        "treatment": "No treatment needed. Keep maintaining consistent watering (1-2 inches/week) and support plants with stakes/cages.",
        "prevention": "Inspect weekly for pests like hornworms or early signs of disease.",
        "severity": "none"
    },
    "Tomato Late Blight": {
        "description": "Devastating fungal disease causing rapid plant collapse",
        "treatment": "1. **Emergency**: Remove and bag infected plants immediately to save others.\n2. **Do Not Compost**: Spores survive in compost.\n3. **Protect**: Spray healthy plants with copper or chlorothalonil.",
        "prevention": "- Plant resistant varieties (e.g., 'Mountain Magic').\n- Space plants widely.\n- Water early in the day.",
        "severity": "critical"
    },
    "Tomato Mold": {
        "description": "Fungal disease causing leaf mold and reduced photosynthesis",
        "treatment": "Remove affected tissue; improve ventilation; apply fungicides where indicated and practice sanitation.",
        "prevention": "Ensure good air circulation, avoid high humidity, prune for airflow, use resistant varieties.",
        "severity": "moderate"
    },
    "Tomato Mosaic Virus": {
        "description": "Viral disease causing mottled leaves and stunted growth",
        "treatment": "Use virus-free seed/transplants; rogue infected plants; disinfect tools; control insect vectors.",
        "prevention": "Use certified virus-free seed, wash hands before handling plants, control aphids, remove infected plants.",
        "severity": "high"
    },
    "Tomato Septoria Spot": {
        "description": "Fungal disease with small circular spots with dark borders",
        "treatment": "Remove infected leaves; apply fungicides targeted for Septoria; keep foliage dry and spaced well.",
        "prevention": "Mulch plants, avoid overhead watering, rotate crops, apply fungicides preventively.",
        "severity": "moderate"
    },
    "Tomato Spider Mites Two Spotted Spider Mite": {
        "description": "Pest infestation causing stippling and webbing on leaves",
        "treatment": "Wash plants with water; introduce predatory mites; use miticides if severe; reduce plant stress.",
        "prevention": "Monitor regularly, maintain plant vigor, avoid water stress, encourage beneficial insects.",
        "severity": "moderate"
    },
    "Tomato Target Spot": {
        "description": "Fungal disease with concentric ring patterns on leaves",
        "treatment": "Remove infected tissue; apply fungicide when necessary; maintain crop hygiene.",
        "prevention": "Practice crop rotation, space plants well, apply fungicides in wet conditions.",
        "severity": "moderate"
    },
    "Tomato Tomato Mosaic Virus": {
        "description": "Viral disease causing mosaic patterns and plant distortion",
        "treatment": "Rogue infected plants, sanitize tools, control vectors and use resistant varieties if available.",
        "prevention": "Use virus-free seed, disinfect tools with bleach solution, control aphids and whiteflies.",
        "severity": "high"
    },
    "Tomato Tomato Yellowleaf Curl Virus": {
        "description": "Viral disease transmitted by whiteflies causing leaf curling",
        "treatment": "Control whitefly vectors; remove infected plants and use virus-resistant varieties if possible.",
        "prevention": "Use resistant varieties, control whiteflies with insecticides or yellow sticky traps, use reflective mulch.",
        "severity": "critical"
    },
    "Tomato Yellow Virus": {
        "description": "Viral disease causing yellowing and stunted growth",
        "treatment": "Investigate viral causes; remove affected plants; control insect vectors and practice sanitation.",
        "prevention": "Control insect vectors, use virus-free transplants, remove infected plants promptly.",
        "severity": "high"
    },
    "Tomato Powdery Mildew": {
        "description": "Fungal disease causing white powdery spots on leaves and stems",
        "treatment": "1. **Removal**: Remove infected leaves immediately to reduce spore spread.\n2. **Fungicides**: Apply sulfur, neem oil, or potassium bicarbonate sprays.\n3. **Airflow**: Prune to improve air circulation and reduce humidity.",
        "prevention": "- Space plants properly to ensure good airflow.\n- Plant resistant varieties if available.\n- Water at the base to keep leaves dry.",
        "severity": "moderate"
    }
}

DEFAULT_TREATMENT = {
    "description": "Unknown or unspecified plant disease",
    "treatment": "No specific treatment found. Remove affected parts, maintain good hygiene, improve airflow, and consult local extension for specific chemical controls.",
    "prevention": "Practice good cultural practices, monitor plants regularly, maintain plant vigor.",
    "severity": "unknown"
}


def get_treatment(disease_name: str) -> dict:
    """
    Get treatment information for a specific disease.
    
    Args:
        disease_name: Name of the disease
        
    Returns:
        Dictionary with treatment information
    """
    # Original treatment data
    base_data = TREATMENTS.get(disease_name, DEFAULT_TREATMENT)
    
    # Try to find matching detailed data
    # Normalize keys for matching (lowercase, no extra spaces)
    norm_name = disease_name.lower().strip()
    
    detailed_info = None
    # Check direct match
    if norm_name in DETAILED_TREATMENTS:
        detailed_info = DETAILED_TREATMENTS[norm_name]
    else:
        # Check partial match/variations
        for key in DETAILED_TREATMENTS:
            if norm_name in key or key in norm_name:
                # Basic similarity check
                detailed_info = DETAILED_TREATMENTS[key]
                break
    
    if detailed_info:
        # Create a merged dictionary
        merged = base_data.copy()
        # Add the detailed answer as 'detailed_analysis'
        merged['detailed_analysis'] = detailed_info['answer']
        return merged
        
    return base_data


def format_treatment_message(disease: str, confidence: float) -> str:
    """
    Format a user-friendly treatment message.
    
    Args:
        disease: Disease name
        confidence: Confidence score (0-1)
        
    Returns:
        Formatted message string
    """
    treatment_info = get_treatment(disease)
    confidence_pct = confidence * 100
    
    # Severity emoji
    severity_emoji = {
        "none": "✅",
        "low": "ℹ️",
        "moderate": "⚠️",
        "high": "🔴",
        "critical": "🚨",
        "unknown": "❓"
    }
    emoji = severity_emoji.get(treatment_info["severity"], "ℹ️")
    
    message = f"""
{emoji} **Diagnosis: {disease}**
**Confidence:** {confidence_pct:.1f}%

📋 **Treatment:**
{treatment_info['treatment']}

🛡️ **Prevention:**
{treatment_info['prevention']}
"""
    
    if treatment_info["severity"] in ["high", "critical"]:
        message += "\n⚠️ **Note:** This is a serious disease. Act quickly to prevent spread!"
    
    if treatment_info.get("detailed_analysis"):
        message += f"\n\n🧐 **Detailed Analysis:**\n{treatment_info['detailed_analysis'][:500]}...\n*(Ask for more details!)*"
    
    return message.strip()


def get_all_disease_names() -> list:
    """Get list of all supported disease names."""
    return list(TREATMENTS.keys())