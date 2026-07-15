import streamlit as st
import pickle
import numpy as np
import os

# Set up page configurations
st.set_page_config(
    page_title="Thyroid AI Clinic",
    page_icon="🩺",
    layout="centered"
)

# App Title & Branding
st.title("🩺 Thyroid AI Clinic")
st.subheader("Autonomous Predictive Diagnostic Dashboard")
st.markdown("---")

# Load the trained model safely
MODEL_PATH = 'model/thyroid_model.pkl'

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None

model = load_model()

# Handle case where model isn't trained yet
if model is None:
    st.error("⚠️ Pre-trained model file (`thyroid_model.pkl`) not found!")
    st.info("Please run `python model/train_model.py` first to generate and train the model framework.")
else:
    st.sidebar.header("📋 Patient Meta-Information")
    patient_id = st.sidebar.text_input("Patient ID/Name", value="PT-8801")
    age = st.sidebar.slider("Patient Age", 1, 100, 35)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])

    st.header("🧪 Clinical Laboratory Inputs")
    st.write("Enter the patient's thyroid panel metrics down below:")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        tsh = st.number_input("TSH Level (µIU/mL)", min_value=0.0, max_value=50.0, value=1.8, step=0.1)
    with col2:
        t3 = st.number_input("Total T3 Level (ng/dL)", min_value=0.0, max_value=10.0, value=1.2, step=0.1)
    with col3:
        t4 = st.number_input("Total T4 Level (µg/dL)", min_value=0.0, max_value=30.0, value=8.5, step=0.1)

    st.markdown("---")

    # Run AI Diagnostics Evaluation
    if st.button("Analyze & Run Diagnostics", type="primary"):
        # Format the features exactly as expected by the Random Forest model
        features = np.array([[age, tsh, t3, t4]])
        
        # Predictions & Confidence Scores
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = probabilities[int(prediction)] * 100

        st.subheader("📊 Diagnostic Outcome")
        
        # Format mapping for output classes
        classes = {0: "Normal (Euthyroid)", 1: "Hyperthyroidism", 2: "Hypothyroidism"}
        result_text = classes.get(prediction, "Unknown Classification")

        if prediction == 0:
            st.success(f"**Classification:** {result_text} (Confidence: {confidence:.2f}%)")
            st.info("💡 **Clinical Recommendation:** Patient metrics look standard. Continue with annual routines. No immediate surgical evaluation requested.")
        elif prediction == 1:
            st.warning(f"**Classification:** {result_text} (Confidence: {confidence:.2f}%)")
            st.error("🚨 **Surgical Risk Assessment:** Elevated risk. Recommended for an ultrasound or nuclear thyroid scan to evaluate nodule status before surgical planning.")
        elif prediction == 2:
            st.warning(f"**Classification:** {result_text} (Confidence: {confidence:.2f}%)")
            st.info("💊 **Clinical Recommendation:** Hormonal therapy stabilization advised. Monitor systemic levels over a 6-week timeframe.")

st.markdown("<br><br><center><small>Powered by OMG CREATOR Systems</small></center>", unsafe_allow_html=True)
