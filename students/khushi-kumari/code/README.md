# 🌿 PlantDocBot - AI Plant Disease Detection

> AI-powered plant disease detection system using Deep Learning  
> **Developed by:** Khushi | **Infosys Virtual Internship 6.0** - AI Domain

[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green)](https://flask.palletsprojects.com/)

---

## 📋 Overview

PlantDocBot identifies plant diseases from leaf images and text descriptions, providing instant diagnosis with treatment recommendations. The system uses MobileNetV2 CNN and DistilBERT NLP models to analyze 28+ disease classes across multiple crops.

**Key Highlights:**
- ⚡ **Fast**: 2-second analysis
- 🎯 **Accurate**: 65.59% validation accuracy
- 💬 **Dual Input**: Image upload + AI chat
- 📱 **Responsive**: Works on any device

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🖼️ **Image Detection** | Upload leaf images with drag-and-drop & webcam support |
| 💬 **AI Chat** | Natural language symptom description with keyword matching |
| 📊 **Diagnostics** | Confidence scores, top-5 predictions, visual metrics |
| 💊 **Treatment Plans** | Detailed symptoms, treatments, and prevention strategies |
| 🎨 **Modern UI** | Beautiful gradient design with smooth animations |

---

## 🌱 Supported Diseases (28 Classes)

**Tomato** (8) • **Potato** (3) • **Corn** (3) • **Apple** (3) • **Grape** (2) • **Bell Pepper** (2) • **Squash** (1) • **Others** (6 healthy leaves)

<details>
<summary>📖 View Complete List</summary>

- **Tomato**: Early Blight, Late Blight, Bacterial Spot, Septoria Leaf Spot, Leaf Mold, Yellow Leaf Curl Virus, Mosaic Virus, Healthy
- **Potato**: Early Blight, Late Blight, Healthy  
- **Corn**: Common Rust, Leaf Blight, Gray Leaf Spot
- **Apple**: Scab, Cedar Apple Rust, Healthy
- **Grape**: Black Rot, Healthy
- **Bell Pepper**: Bacterial Spot, Healthy
- **Other**: Squash, Pepper, Peach, Cherry, Blueberry, Soybean, Raspberry, Strawberry
</details>

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/springboardmentor8984d-code/PlantDoc-AI.git
cd PlantDoc-AI

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

Open browser: **http://localhost:5000**

---

## 📁 Project Structure

```
PlantDocBot/
├── app.py                    # Flask application
├── improved_training.py      # Training script
├── best_model.pth           # Trained model (29.51 MB)
├── class_names.json         # Disease classes
├── requirements.txt         # Dependencies
├── templates/               # HTML files
│   ├── index.html
│   ├── upload.html
│   ├── chat.html
│   └── about.html
└── screenshots/             # Project screenshots
```

---

## 🔧 Technology Stack

**Backend:** Python, Flask, PyTorch, Transformers  
**AI Models:** MobileNetV2 (CNN), DistilBERT (NLP)  
**Frontend:** HTML5, CSS3, JavaScript  

**Key Libraries:**
```
torch>=2.0.1, torchvision>=0.15.2, transformers>=4.32.0
Flask>=2.3.3, Pillow>=10.0.0, numpy>=1.24.3
```

---

## 🧠 Model Details

### CNN Architecture
```
MobileNetV2 (Pre-trained)
└── Custom Classifier
    ├── Linear(1280 → 512) + Dropout(0.4)
    ├── Linear(512 → 256) + Dropout(0.3)
    └── Linear(256 → 28) + Dropout(0.2)
```

**Training:**
- Dataset: PlantDoc (70K+ images)
- Input: 224×224 RGB
- Optimizer: Adam (lr=0.001)
- Validation Accuracy: **65.59%**

---

## 📸 Screenshots

| Homepage | Upload & Analysis | AI Chat |
|----------|-------------------|---------|
| ![Home](screenshots/homepage.png) | ![Upload](screenshots/upload.png) | ![Chat](screenshots/chat.png) |

---

## 💡 Usage Tips

**For Best Results:**
- 📸 Use clear, well-lit images
- 🎯 Focus on affected leaf areas
- 📝 Be specific when describing symptoms
- 🌿 Mention plant type in chat queries

---

## 🔄 Training (Optional)

```bash
python improved_training.py
```

**Prompts:**
- Training directory: `PlantDoc-Dataset/train`
- Validation directory: `PlantDoc-Dataset/test`

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found | Ensure `best_model.pth` exists |
| Port 5000 in use | Change port in `app.py` |
| Low accuracy | Use clear images, retrain model |
| Dataset not found | Run `python dataset_finder.py` |

---

## 🎯 Future Enhancements

- [ ] Mobile app (iOS/Android)
- [ ] Multi-language support (Hindi, Tamil, etc.)
- [ ] Weather-based disease prediction
- [ ] Community forum for farmers
- [ ] PDF report generation

---

## 🙏 Acknowledgments

**Infosys Springboard** • **PlantVillage Dataset** • **PyTorch** • **Hugging Face** • **Flask**

**References:**  
MobileNetV2 • DistilBERT • PlantDoc Dataset

---

## 📞 Contact

**Developer:** Khushi  
**Program:** Infosys Virtual Internship 6.0 - AI Domain  
**Repository:** [PlantDoc-AI](https://github.com/springboardmentor8984d-code/PlantDoc-AI)

---

## 📄 License

Educational project developed for **Infosys Virtual Internship 6.0**  
Not for commercial use without permission.

---

**⭐ If this project helped you, please star it on GitHub!**

---

**Made with ❤️ by Khushi** | *Last Updated: December 2025*