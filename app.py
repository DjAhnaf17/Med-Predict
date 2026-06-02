from flask import Flask, render_template, request, jsonify, send_file
import pickle
import os
import io
import datetime
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

app = Flask(__name__)

# ── Load ML Models ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

with open(os.path.join(MODEL_DIR, 'random_forest.pkl'), 'rb') as f:
    rf_model = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'svm.pkl'), 'rb') as f:
    svm_model = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'naive_bayes.pkl'), 'rb') as f:
    nb_model = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'rb') as f:
    le = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'symptoms.pkl'), 'rb') as f:
    SYMPTOMS = pickle.load(f)

DISEASES = le.classes_.tolist()

# Disease info dictionary
DISEASE_INFO = {
    "Common Cold": {"severity": "Mild", "recommendation": "Rest, hydrate, and take OTC cold medicine. Consult doctor if symptoms persist beyond 10 days.", "emoji": "🤧"},
    "Influenza": {"severity": "Moderate", "recommendation": "Rest, stay hydrated. Antiviral medications may help if taken early. Consult a doctor.", "emoji": "🤒"},
    "Dengue": {"severity": "High", "recommendation": "Seek immediate medical attention. Avoid aspirin. Monitor platelet count closely.", "emoji": "🦟"},
    "Malaria": {"severity": "High", "recommendation": "Urgent medical care required. Antimalarial medications needed. Do not delay treatment.", "emoji": "🦠"},
    "Typhoid": {"severity": "High", "recommendation": "Requires antibiotic treatment. Consult a doctor immediately. Maintain hygiene and hydration.", "emoji": "⚠️"},
    "Pneumonia": {"severity": "High", "recommendation": "Immediate medical attention required. May need antibiotics or antiviral medication.", "emoji": "🫁"},
    "COVID-19": {"severity": "Moderate-High", "recommendation": "Isolate immediately, get tested. Monitor oxygen levels. Seek emergency care if breathing is difficult.", "emoji": "😷"},
    "Asthma": {"severity": "Moderate", "recommendation": "Use prescribed inhaler. Avoid triggers. Consult pulmonologist for long-term management.", "emoji": "💨"},
    "Migraine": {"severity": "Moderate", "recommendation": "Rest in dark, quiet room. Over-the-counter pain relievers. Consult neurologist for recurring migraines.", "emoji": "🧠"},
    "Food Poisoning": {"severity": "Moderate", "recommendation": "Stay hydrated. Avoid solid food initially. Seek medical care if vomiting persists over 24 hours.", "emoji": "🤢"},
    "Bronchitis": {"severity": "Mild-Moderate", "recommendation": "Rest, stay hydrated, use a humidifier. Consult doctor for prescribed medications.", "emoji": "😮‍💨"},
    "Gastritis": {"severity": "Mild-Moderate", "recommendation": "Avoid spicy foods and alcohol. Take antacids. Consult gastroenterologist for H. pylori testing.", "emoji": "🫃"},
    "Peptic Ulcer": {"severity": "Moderate", "recommendation": "Avoid NSAIDs and alcohol. Consult doctor for acid-reducing medication and H. pylori treatment.", "emoji": "⚕️"},
    "Arthritis": {"severity": "Chronic", "recommendation": "Physical therapy, anti-inflammatory medications. Consult rheumatologist for management plan.", "emoji": "🦴"},
    "Chickenpox": {"severity": "Mild", "recommendation": "Rest, apply calamine lotion, avoid scratching. Keep away from vulnerable individuals.", "emoji": "🔴"},
    "Tonsillitis": {"severity": "Mild-Moderate", "recommendation": "Rest, gargle warm salt water, pain relievers. Consult doctor for antibiotics if bacterial.", "emoji": "👅"},
    "Eczema": {"severity": "Chronic", "recommendation": "Moisturize regularly, avoid triggers. Consult dermatologist for topical treatments.", "emoji": "🩹"},
    "Psoriasis": {"severity": "Chronic", "recommendation": "Consult dermatologist for medicated creams, light therapy, or systemic treatments.", "emoji": "🩺"},
    "Hepatitis A": {"severity": "Moderate-High", "recommendation": "Rest, avoid alcohol, stay hydrated. Usually resolves on its own; consult doctor for monitoring.", "emoji": "🧪"},
    "Hepatitis B": {"severity": "High", "recommendation": "Immediate medical evaluation required. Antiviral treatment may be needed. Avoid alcohol.", "emoji": "🔬"},
}

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', symptoms=SYMPTOMS)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Extract patient info
    patient = {
        'name': data.get('name', ''),
        'age': data.get('age', ''),
        'dob': data.get('dob', ''),
        'blood_group': data.get('blood_group', ''),
        'phone': data.get('phone', ''),
    }
    
    selected_symptoms = data.get('symptoms', [])
    model_choice = data.get('model', 'random_forest')
    
    # Build feature vector
    feature_vector = pd.DataFrame([[1 if s in selected_symptoms else 0 for s in SYMPTOMS]], columns=SYMPTOMS)
    
    # Select model
    model_map = {
        'random_forest': rf_model,
        'svm': svm_model,
        'naive_bayes': nb_model
    }
    model = model_map.get(model_choice, rf_model)
    
    # Predict
    pred_encoded = model.predict(feature_vector)[0]
    disease = le.inverse_transform([pred_encoded])[0]
    
    # Get probabilities for all models
    results = {}
    for mname, mobj in model_map.items():
        try:
            proba = mobj.predict_proba(feature_vector)[0]
            pred_idx = mobj.predict(feature_vector)[0]
            pred_disease = le.inverse_transform([pred_idx])[0]
            confidence = round(float(proba[pred_idx]) * 100, 1)
            results[mname] = {'disease': pred_disease, 'confidence': confidence}
        except Exception:
            results[mname] = {'disease': le.inverse_transform([mobj.predict(feature_vector)[0]])[0], 'confidence': 0}
    
    info = DISEASE_INFO.get(disease, {"severity": "Unknown", "recommendation": "Consult a doctor.", "emoji": "🏥"})
    
    return jsonify({
        'success': True,
        'disease': disease,
        'model_used': model_choice,
        'all_results': results,
        'info': info,
        'patient': patient,
        'symptoms_selected': selected_symptoms,
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    data = request.get_json()
    patient = data.get('patient', {})
    disease = data.get('disease', 'Unknown')
    model_used = data.get('model_used', '')
    all_results = data.get('all_results', {})
    symptoms_selected = data.get('symptoms_selected', [])
    info = data.get('info', {})
    timestamp = data.get('timestamp', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=0.75*inch, leftMargin=0.75*inch,
        topMargin=0.75*inch, bottomMargin=0.75*inch
    )
    
    styles = getSampleStyleSheet()
    story = []
    
    # ── Custom Styles
    title_style = ParagraphStyle('Title', parent=styles['Normal'],
        fontSize=22, textColor=colors.HexColor('#1a1a2e'),
        alignment=TA_CENTER, fontName='Helvetica-Bold', spaceAfter=8, leading=28)
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'],
        fontSize=11, textColor=colors.HexColor('#4a4a8a'),
        alignment=TA_CENTER, fontName='Helvetica', spaceAfter=6, leading=16)
    section_header = ParagraphStyle('SectionHeader', parent=styles['Normal'],
        fontSize=13, textColor=colors.white,
        fontName='Helvetica-Bold', spaceBefore=14, spaceAfter=10,
        backColor=colors.HexColor('#4a4a8a'), leftIndent=-10, rightIndent=-10,
        borderPad=8, leading=16)
    normal_bold = ParagraphStyle('NormalBold', parent=styles['Normal'],
        fontSize=10, fontName='Helvetica-Bold',
        textColor=colors.HexColor('#1a1a2e'), leading=14)
    normal_text = ParagraphStyle('NormalText', parent=styles['Normal'],
        fontSize=10, fontName='Helvetica',
        textColor=colors.HexColor('#333333'), leading=15, spaceAfter=6)
    disease_style = ParagraphStyle('DiseaseStyle', parent=styles['Normal'],
        fontSize=20, textColor=colors.HexColor('#e63946'),
        alignment=TA_CENTER, fontName='Helvetica-Bold', spaceAfter=10, leading=26)
    footer_style = ParagraphStyle('Footer', parent=styles['Normal'],
        fontSize=8, textColor=colors.HexColor('#999999'),
        alignment=TA_CENTER, fontName='Helvetica-Oblique', leading=12)
    
    # ── Header
    story.append(Paragraph("🏥 MediPredict AI", title_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Disease Prediction Report", subtitle_style))
    story.append(Spacer(1, 2))
    story.append(Paragraph(f"Generated: {timestamp}", subtitle_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#4a4a8a'), spaceAfter=16))
    
    # ── Patient Info
    story.append(Paragraph("  Patient Information", section_header))
    story.append(Spacer(1, 10))
    
    patient_data = [
        ['Patient Name:', patient.get('name', 'N/A'), 'Age:', patient.get('age', 'N/A')],
        ['Date of Birth:', patient.get('dob', 'N/A'), 'Blood Group:', patient.get('blood_group', 'N/A')],
        ['Phone Number:', patient.get('phone', 'N/A'), 'Report ID:', f"MP-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"],
    ]
    
    t = Table(patient_data, colWidths=[1.5*inch, 2.5*inch, 1.5*inch, 1.5*inch])
    t.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME', (2,0), (2,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0,0), (0,-1), colors.HexColor('#4a4a8a')),
        ('TEXTCOLOR', (2,0), (2,-1), colors.HexColor('#4a4a8a')),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.HexColor('#f8f9fa'), colors.white]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e0e0e0')),
        ('PADDING', (0,0), (-1,-1), 8),
        ('ROWHEIGHT', (0,0), (-1,-1), 28),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t)
    story.append(Spacer(1, 16))
    
    # ── Symptoms
    story.append(Paragraph("  Selected Symptoms", section_header))
    story.append(Spacer(1, 10))
    
    sym_display = ', '.join([s.replace('_', ' ') for s in symptoms_selected]) if symptoms_selected else 'None selected'
    sym_text = f"<b>Reported Symptoms ({len(symptoms_selected)}):</b>\n{sym_display}"
    story.append(Paragraph(sym_text, normal_text))
    story.append(Spacer(1, 16))
    
    # ── Diagnosis Result
    story.append(Paragraph("  Diagnosis Result", section_header))
    story.append(Spacer(1, 10))
    
    emoji = info.get('emoji', '🏥')
    story.append(Paragraph(f"{emoji}  {disease}", disease_style))
    story.append(Spacer(1, 10))
    
    sev_color = {'Mild': '#2ecc71', 'Moderate': '#f39c12', 'High': '#e74c3c',
                 'Chronic': '#9b59b6', 'Moderate-High': '#e67e22', 'Mild-Moderate': '#f39c12'}.get(
                     info.get('severity', ''), '#333')
    
    diag_data = [
        ['Predicted Disease:', disease],
        ['Severity Level:', info.get('severity', 'Unknown')],
        ['Model Used:', model_used.replace('_', ' ').title()],
    ]
    t2 = Table(diag_data, colWidths=[2.5*inch, 4.5*inch])
    t2.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 11),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0,0), (0,-1), colors.HexColor('#4a4a8a')),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.HexColor('#fff5f5'), colors.HexColor('#f8f9fa')]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e0e0e0')),
        ('PADDING', (0,0), (-1,-1), 10),
        ('ROWHEIGHT', (0,0), (-1,-1), 30),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t2)
    story.append(Spacer(1, 16))
    
    # ── All Models Comparison
    story.append(Paragraph("  Model Comparison", section_header))
    story.append(Spacer(1, 10))
    
    model_headers = [['Model', 'Predicted Disease', 'Confidence']]
    model_rows = []
    model_names = {'random_forest': 'Random Forest', 'svm': 'Support Vector Machine', 'naive_bayes': 'Naïve Bayes'}
    for mkey, mdata in all_results.items():
        conf = mdata.get('confidence', 0)
        model_rows.append([model_names.get(mkey, mkey), mdata.get('disease', ''), f"{conf}%"])
    
    t3 = Table(model_headers + model_rows, colWidths=[2.2*inch, 3*inch, 1.8*inch])
    t3.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#4a4a8a')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f8f9fa'), colors.white]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cccccc')),
        ('PADDING', (0,0), (-1,-1), 10),
        ('ROWHEIGHT', (0,0), (-1,-1), 28),
        ('ALIGN', (2,0), (2,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t3)
    story.append(Spacer(1, 16))
    
    # ── Recommendation
    story.append(Paragraph("  Medical Recommendation", section_header))
    story.append(Spacer(1, 10))
    
    rec_text = info.get('recommendation', 'Please consult a qualified medical professional.')
    story.append(Paragraph(f"⚕️\n{rec_text}", normal_text))
    story.append(Spacer(1, 18))
    
    # ── Disclaimer
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cccccc'), spaceAfter=8))
    disclaimer = ("⚠️  <b>Disclaimer:</b> This report is generated by an AI-based prediction system for informational "
                  "purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. "
                  "Always consult a qualified healthcare provider for medical concerns.")
    story.append(Paragraph(disclaimer, footer_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("MediPredict AI  •  Powered by Machine Learning  •  For Educational Use Only", footer_style))
    
    doc.build(story)
    buffer.seek(0)
    
    filename = f"MediPredict_{patient.get('name','Patient').replace(' ','_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    
    return send_file(
        buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=filename
    )

@app.route('/send_whatsapp', methods=['POST'])
def send_whatsapp():
    data = request.get_json()
    phone = data.get('phone', '')
    patient_name = data.get('patient_name', 'Patient')
    disease = data.get('disease', 'Unknown')
    recommendation = data.get('recommendation', '')
    severity = data.get('severity', '')
    model_used = data.get('model_used', '')
    
    if not phone:
        return jsonify({'success': False, 'message': 'Phone number is required'})
    
    # Format phone number
    phone_clean = ''.join(filter(str.isdigit, phone))
    if phone_clean.startswith('0'):
        phone_clean = '91' + phone_clean[1:]
    elif not phone_clean.startswith('91') and len(phone_clean) == 10:
        phone_clean = '91' + phone_clean
    
    message = (
        f"🏥 *MediPredict AI — Health Report*\n\n"
        f"Hello {patient_name},\n\n"
        f"Your disease prediction result is ready:\n\n"
        f"🦠 *Predicted Disease:* {disease}\n"
        f"⚠️ *Severity:* {severity}\n"
        f"🤖 *Model Used:* {model_used.replace('_',' ').title()}\n\n"
        f"💊 *Recommendation:*\n{recommendation}\n\n"
        f"⚕️ _This is an AI-based prediction. Please consult a qualified doctor for medical advice._\n\n"
        f"— MediPredict AI Team"
    )
    
    try:
        import pywhatkit
        now = datetime.datetime.now()
        send_hour = now.hour
        send_min = now.minute + 2
        if send_min >= 60:
            send_min -= 60
            send_hour += 1
        if send_hour >= 24:
            send_hour = 0
        
        pywhatkit.sendwhatmsg(f'+{phone_clean}', message, send_hour, send_min, 15, True, 4)
        return jsonify({'success': True, 'message': f'WhatsApp message scheduled for +{phone_clean} at {send_hour:02d}:{send_min:02d}'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'WhatsApp sending failed: {str(e)}. Ensure WhatsApp Web is open in your browser.'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
