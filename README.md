# 🏥 MediPredict AI — Disease Prediction System

A **Glassmorphism-styled Flask ML Web Application** for predicting diseases based on patient symptoms using three Machine Learning models.

## ✨ Features

- **3 ML Models**: Random Forest, SVM (Support Vector Machine), Naïve Bayes
- **Patient Details**: Name, Age, Date of Birth, Blood Group, Phone Number
- **Symptom Selection**: Interactive checkbox UI for 10 symptoms
- **PDF Report**: Professional downloadable report via ReportLab
- **WhatsApp Integration**: Send results to patient's phone via pywhatkit
- **Glassmorphism Design**: Beautiful, animated, fully responsive UI
- **All Models Comparison**: Confidence bars for all three models

## 📋 Symptoms Covered
Fever, Cough, Headache, Fatigue, Nausea, Vomiting, Rash, Joint Pain, Sore Throat, Shortness of Breath

## 🦠 Diseases Predicted (20 Diseases)
Common Cold, Influenza, Dengue, Malaria, Typhoid, Pneumonia, COVID-19, Asthma, Migraine, Food Poisoning, Bronchitis, Gastritis, Peptic Ulcer, Arthritis, Chickenpox, Tonsillitis, Eczema, Psoriasis, Hepatitis A, Hepatitis B

## 🚀 Setup & Installation

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the Models
```bash
python train_models.py
```
This creates the `models/` folder with trained .pkl files.

### Step 3: Run the App
```bash
python app.py
```

### Step 4: Open in Browser
Navigate to: **http://localhost:5000**

---

## 📱 WhatsApp Feature (pywhatkit)

**Requirements for WhatsApp sending:**
1. Make sure **WhatsApp Web** is already logged in at https://web.whatsapp.com
2. Keep your browser open before clicking "Send via WhatsApp"
3. pywhatkit will open WhatsApp Web and schedule the message 2 minutes ahead
4. The browser must remain open during sending

**Phone number format accepted:**
- `9876543210` (10 digits, auto-prefixes +91)
- `+919876543210` (full international)
- `09876543210` (leading 0, converts to +91)

---

## 📄 PDF Report Contents
- Patient Information (Name, Age, DOB, Blood Group, Phone, Report ID)
- Selected Symptoms list
- Predicted Disease with emoji
- Severity Level
- All 3 Models Comparison with confidence scores
- Medical Recommendations
- Disclaimer

---

## 📁 Project Structure
```
disease_predictor/
├── app.py               # Main Flask application
├── train_models.py      # ML model training script
├── disease.csv          # Training dataset
├── requirements.txt     # Python dependencies
├── models/
│   ├── random_forest.pkl
│   ├── svm.pkl
│   ├── naive_bayes.pkl
│   ├── label_encoder.pkl
│   └── symptoms.pkl
├── templates/
│   └── index.html       # Main Glassmorphism UI
└── static/
    ├── css/style.css    # Glassmorphism styles
    └── js/main.js       # Frontend logic
```

---

## ⚠️ Disclaimer
This application is for **educational purposes only**. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.
