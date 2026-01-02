# Enhanced Hospital Disease Prediction System with WhatsApp Integration (Text + PDF)
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from fpdf import FPDF
import base64
import qrcode
from datetime import datetime
import plotly.graph_objects as go
import os
from PIL import Image
import io
import time
import pywhatkit  # ✅ Text WhatsApp
import re

# -------------------- PAGE CONFIG --------------------in
st.set_page_config(page_title="🏥 MedPredict - AI Disease Predictor",
                   layout="wide", page_icon="🧬")

# -------------------- PATHS --------------------
BASE_DIR = os.path.dirname(__file__)
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

logo_path = os.path.join(ASSETS_DIR, "logo.png")
doctor_path = os.path.join(ASSETS_DIR, "temp_photo.png")
sign_path = os.path.join(ASSETS_DIR, "sign.png")
temp_photo_path = os.path.join(ASSETS_DIR, "temp_photo.png")

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
:root{--brand-1:#00695c; --brand-2:#00bfa5; --muted:#f4f9f9; --card:#ffffff}
.main-wrap{background: linear-gradient(180deg, var(--muted) 0%, #ffffff 100%); padding: 18px 20px;}
.animated-title{font-size:34px; font-weight:800; text-align:center; color:var(--brand-1);
  background: linear-gradient(90deg, #004d40, #00796b, #00bfa5);
  -webkit-background-clip: text; background-clip: text; color: transparent;
  animation: slideGradient 6s ease infinite;}
@keyframes slideGradient{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
.card{background:var(--card); border-radius:14px; padding:18px; box-shadow: 0 8px 30px rgba(2,48,43,0.06);}
.result-box{border-left:6px solid var(--brand-1); padding:14px; border-radius:10px; background:linear-gradient(90deg,#e8faf6, #ffffff);}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-wrap"><div class="animated-title">🏥 HealthSense — AI Disease Prediction (Pro)</div></div>', unsafe_allow_html=True)

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("../datasets/Training.csv").dropna(axis=1)
    return df

df = load_data()
encoder = LabelEncoder()
df["prognosis"] = encoder.fit_transform(df["prognosis"])
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

# -------------------- TRAIN MODELS --------------------
@st.cache_data
def train_models():
    svm = SVC(probability=True)
    svm.fit(X_train, y_train)
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    rf = RandomForestClassifier(random_state=18)
    rf.fit(X_train, y_train)
    return svm, nb, rf

svm_model, nb_model, rf_model = train_models()

symptom_index = {" ".join(i.capitalize() for i in val.split("_")): idx for idx, val in enumerate(X.columns)}
data_dict = {"symptom_index": symptom_index, "predictions_classes": encoder.classes_}

# -------------------- DISEASE INFO --------------------
disease_info = {
    "Fungal infection": {"precautions": ["Keep area dry", "Use antifungal cream", "Avoid tight clothes"], "doctor": "Dermatologist"},
    "Hypertension": {"precautions": ["Low salt diet", "Regular exercise", "Monitor BP"], "doctor": "Cardiologist"},
    "Diabetes": {"precautions": ["Low sugar diet", "Regular insulin check", "Daily walking"], "doctor": "Endocrinologist"},
    "Heart disease": {"precautions": ["Avoid stress", "Quit smoking", "Eat heart-healthy food"], "doctor": "Cardiologist"},
    "default": {"precautions": ["Consult a doctor immediately."], "doctor": "General Physician"}
}

# -------------------- LIFESTYLE TIPS --------------------
lifestyle_tips = {
    "Diabetes": ["🍏 Eat more vegetables", "🚶 Exercise 30 min daily", "💧 Stay hydrated"],
    "Hypertension": ["🧘 Practice yoga", "🛌 Sleep 7-8 hours", "🥗 Reduce sodium intake"],
    "Heart disease": ["❤️ Avoid smoking", "🏃 Maintain healthy weight", "🥦 Eat heart-friendly diet"],
    "default": ["🥗 Maintain balanced diet", "🚶 Stay active", "💧 Drink plenty of water"]
}

# -------------------- PREDICT FUNCTION --------------------
def predict_disease(symptoms):
    symptoms = symptoms.split(",")
    input_data = [0] * len(symptom_index)
    for s in symptoms:
        s = s.strip().capitalize()
        if s in symptom_index:
            input_data[symptom_index[s]] = 1
    input_data = np.array(input_data).reshape(1, -1)

    rf_pred = rf_model.predict(input_data)[0]
    nb_pred = nb_model.predict(input_data)[0]
    svm_pred = svm_model.predict(input_data)[0]

    rf_proba = max(rf_model.predict_proba(input_data)[0])
    nb_proba = max(nb_model.predict_proba(input_data)[0])
    svm_proba = max(svm_model.predict_proba(input_data)[0])

    votes = [rf_pred, nb_pred, svm_pred]
    final_pred = max(set(votes), key=votes.count)

    return {
        "rf_model_prediction": encoder.classes_[rf_pred],
        "naive_bayes_prediction": encoder.classes_[nb_pred],
        "svm_model_prediction": encoder.classes_[svm_pred],
        "final_prediction": encoder.classes_[final_pred],
        "confidence_rf": round(rf_proba * 100, 2),
        "confidence_nb": round(nb_proba * 100, 2),
        "confidence_svm": round(svm_proba * 100, 2),
    }

# -------------------- RISK LEVEL --------------------
def risk_level(conf):
    if conf > 80:
        return "🔴 High Risk"
    elif conf > 50:
        return "🟠 Medium Risk"
    else:
        return "🟢 Low Risk"

# -------------------- QR CODE --------------------
def generate_qr_code(url):
    qr = qrcode.make(url)
    qr_path = os.path.join(ASSETS_DIR, "qr_health_info.png")
    qr.save(qr_path)
    return qr_path

# -------------------- DOCTOR INFO --------------------
doctor_info = {
    "name": "Dr. Jawwad Ahnaf",
    "role": "General Physician",
    "sign": sign_path
}

# -------------------- PDF REPORT --------------------
def create_pdf(pred, patient_info, extra_info, patient_photo=None):
    pdf = FPDF()
    pdf.add_page()

    pdf.image(logo_path, x=10, y=8, w=30)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, "MedPredict Health Clinic", ln=1, align='C')
    pdf.set_font("Arial", '', 9)
    pdf.cell(200, 5, "123 XYZ Street, Vaniyambadi - 635751", ln=1, align='C')
    pdf.cell(200, 5, "Phone: +91 9876543210 | Email: contact@medpredict.com", ln=1, align='C')
    pdf.line(10, 35, 200, 35)
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 11)
    pdf.cell(200, 8, "Patient Information", ln=1)
    pdf.set_font("Arial", '', 9)
    pdf.cell(100, 6, f"Name: {patient_info['name']}", ln=0)
    pdf.cell(100, 6, f"Age: {patient_info['age']}", ln=1)
    pdf.cell(100, 6, f"Gender: {patient_info['gender']}", ln=0)
    pdf.cell(100, 6, f"Date: {datetime.today().strftime('%Y-%m-%d')}", ln=1)
    pdf.cell(100, 6, f"BMI: {patient_info['bmi']}", ln=1)

    if patient_photo:
        image = Image.open(patient_photo).convert("RGB")
        image.save(temp_photo_path, format="PNG")
        pdf.image(temp_photo_path, x=160, y=50, w=30, h=30)

    pdf.ln(10)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(200, 8, "Diagnosis Report", ln=1)
    pdf.set_font("Arial", '', 9)
    for k, v in pred.items():
        label = k.replace('_', ' ').capitalize()
        pdf.cell(95, 6, f"{label}:", border=0)
        pdf.cell(95, 6, str(v), ln=1, border=0)

    pdf.ln(4)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(200, 8, "Doctor Recommendation & Precautions", ln=1)
    pdf.set_font("Arial", '', 9)
    pdf.cell(200, 6, f"Doctor: {extra_info['doctor']}", ln=1)
    for p in extra_info['precautions']:
        pdf.multi_cell(0, 6, f"- {p}")

    qr_path = generate_qr_code("https://www.webmd.com")
    pdf.ln(4)
    pdf.cell(200, 6, "Scan QR Code for more info:", ln=1)
    pdf.image(qr_path, x=80, y=None, w=35)

    pdf.ln(15)
    # pdf.image(doctor_info["photo"], x=20, y=None, w=25)
    pdf.image(doctor_info["sign"], x=180, y=None, w=35)
    pdf.cell(200, 8, doctor_info["name"], ln=1, align='R')
    pdf.cell(200, 5, doctor_info["role"], ln=1, align='R')

    return pdf.output(dest="S").encode("latin-1")


# -------------------- VALIDATE PHONE NUMBER --------------------
def validate_phone_number(phone):
    # Basic validation for international phone numbers
    pattern = re.compile(r'^\+[1-9]\d{1,14}$')
    return pattern.match(phone) is not None

# -------------------- SESSION STATE --------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

# -------------------- UI --------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🧑 Patient Info", "🩺 Symptoms", "📊 Results & Report", "📜 History", "🤖 AI Chatbot"])

# -------- Patient Info --------
with tab1:
    patient_name = st.text_input("Full Name")
    patient_age = st.number_input("Age", 0, 120, 25)
    patient_gender = st.radio("Gender", ["Male", "Female", "Other"])
    patient_phone = st.text_input("WhatsApp Number (with country code, e.g. +919876543210)")
    patient_photo = st.file_uploader("Upload Patient Photo", type=["jpg", "jpeg", "png"])
    height = st.number_input("Height (cm)", 100, 220, 170)
    weight = st.number_input("Weight (kg)", 20, 200, 70)
    bmi = round(weight / ((height/100)**2), 2)
    st.write(f"📊 Your BMI: **{bmi}**")

# -------- Symptoms --------
with tab2:
    selected_symptoms = st.multiselect("Choose symptoms:", sorted(symptom_index.keys()))
    manual_input = st.text_input("Or type manually (comma-separated):")
    symptoms_input = manual_input if manual_input else ", ".join(selected_symptoms)

# -------- Results & Report --------
with tab3:
    if st.button("🚀 Predict Disease"):
        if not symptoms_input or not patient_name:
            st.warning("⚠️ Please provide patient info and symptoms.")
        elif not patient_phone or not validate_phone_number(patient_phone):
            st.warning("⚠️ Please provide a valid WhatsApp number with country code (e.g. +919876543210).")
        else:
            result = predict_disease(symptoms_input)
            final_disease = result["final_prediction"]
            extra_info = disease_info.get(final_disease, disease_info["default"])
            tips = lifestyle_tips.get(final_disease, lifestyle_tips["default"])
            avg_conf = np.mean([result["confidence_rf"], result["confidence_nb"], result["confidence_svm"]])
            risk = risk_level(avg_conf)

            st.markdown(f"<div class='result-box'>✅ Final Prediction: <strong>{final_disease}</strong><br>Risk Level: {risk}</div>", unsafe_allow_html=True)

            fig = go.Figure(go.Indicator(mode="gauge+number", value=avg_conf, title={'text': "Average Confidence"}, gauge={'axis': {'range': [0, 100]}}))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📄 Download Professional Report")
            patient_info = {"name": patient_name, "age": patient_age, "gender": patient_gender, "bmi": bmi}
            pdf_bytes = create_pdf(result, patient_info, extra_info, patient_photo)
            pdf_path = "Patient_Report.pdf"
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)

            b64 = base64.b64encode(pdf_bytes).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="Patient_Report.pdf">📥 Download Report (PDF)</a>'
            st.markdown(href, unsafe_allow_html=True)

            # -------------------- WHATSAPP MESSAGING --------------------
            st.subheader("📲 Sending Report via WhatsApp")
            
            # Send Text Message
            with st.spinner("Sending WhatsApp text message..."):
                now = datetime.now()
                message = f"""🏥 MedPredict Health Report
------------------------
👤 Patient: {patient_name}
📅 Date: {now.strftime('%Y-%m-%d')}
🕒 Time: {now.strftime('%H:%M')}
🧬 Diagnosis: {final_disease}
⚖️ Risk Level: {risk}

👨‍⚕️ Doctor: {doctor_info['name']} ({doctor_info['role']})

✅ Please consult your doctor for further guidance.
"""
                try:
                    pywhatkit.sendwhatmsg_instantly(
                        patient_phone,
                        message,
                        wait_time=10,
                        tab_close=False
                    )
                    st.success("✅ Text report sent successfully on WhatsApp!")
                except Exception as e:
                    st.error(f"❌ Failed to send text message: {e}")
            
            # Send PDF
            # with st.spinner("Sending PDF report via WhatsApp..."):
            #     status = send_pdf_via_whatsapp(patient_phone, pdf_path)
            #     st.info(status)
            
            # Add to history
            st.session_state["history"].append({
                "Name": patient_name,
                "Date": datetime.today().strftime('%Y-%m-%d'),
                "Diagnosis": final_disease,
                "Risk": risk
            })

# -------- History --------
with tab4:
    st.subheader("📜 Patient History")
    if st.session_state["history"]:
        history_df = pd.DataFrame(st.session_state["history"])
        st.table(history_df)
    else:
        st.info("No history yet. Make a prediction first.")

# -------- AI Chatbot --------
with tab5:
    st.subheader("Ask AI Health Assistant")
    query = st.text_input("Type your health-related question:")
    if query:
        query = query.lower()

        if "diabetes" in query:
            st.write("🍎 For **Diabetes**: Maintain a low sugar diet, monitor glucose regularly, take insulin/medications as prescribed, and exercise daily.")
        elif "heart" in query or "cardiac" in query:
            st.write("❤️ For **Heart Health**: Avoid smoking, eat less oily food, manage stress, exercise moderately, and get regular ECG checkups.")
        elif "bp" in query or "hypertension" in query or "blood pressure" in query:
            st.write("🧘 For **Hypertension**: Reduce sodium intake, maintain a healthy weight, practice yoga/meditation, and monitor blood pressure daily.")
        elif "fever" in query:
            st.write("🤒 For **Fever**: Stay hydrated, take adequate rest, monitor temperature, and consult a doctor if it persists beyond 3 days.")
        elif "cold" in query or "cough" in query or "flu" in query:
            st.write("🤧 For **Cold & Flu**: Drink warm fluids, use steam inhalation, take vitamin C, and consult a doctor if symptoms worsen.")
        elif "asthma" in query or "breathing" in query:
            st.write("🌬️ For **Asthma**: Always carry an inhaler, avoid dust/pollution, practice breathing exercises, and consult a pulmonologist.")
        elif "skin" in query or "rash" in query or "allergy" in query:
            st.write("🌿 For **Skin Issues**: Keep skin clean and dry, avoid allergens, apply soothing creams, and consult a dermatologist for chronic rashes.")
        elif "obesity" in query or "weight" in query:
            st.write("⚖️ For **Obesity/Weight Management**: Maintain a calorie-deficit diet, exercise 45 min daily, avoid junk foods, and track BMI regularly.")
        elif "kidney" in query or "urine" in query:
            st.write("💧 For **Kidney Health**: Drink 2–3L water daily, reduce salt intake, avoid overuse of painkillers, and get kidney function tests if needed.")
        elif "mental" in query or "stress" in query or "depression" in query:
            st.write("🧠 For **Mental Health**: Practice meditation, talk to trusted people, maintain good sleep hygiene, and seek a counselor if stress is overwhelming.")
        elif "liver" in query:
            st.write("🍵 For **Liver Health**: Avoid alcohol, eat a balanced diet, exercise regularly, and get liver function tests if you have persistent issues.")
        elif "covid" in query or "corona" in query:
            st.write("🦠 For **COVID-19**: Wear a mask in crowded areas, wash hands often, maintain distance, get vaccinated, and isolate if symptoms appear.")
        else:
            st.write("🩺 Please consult a specialist for personalized advice. I can only provide general health tips.")
