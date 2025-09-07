import streamlit as st
import os
import re
import inspect
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import ml_model
import importlib
importlib.reload(ml_model)

# Load the dataset from the CSV file
try:
    df_hospital_readmission = pd.read_csv("datasets/hospital_readmission.csv")
except FileNotFoundError:
    st.error("Dataset file not found. Please ensure the CSV file is in the datasets folder.")
    st.stop()

# Initialize and train the model
try:
    predictor = ml_model.HospitalReadmissionPredictor()
    
    # Preprocess the data
    X, y = predictor.preprocess_data(df_hospital_readmission)
    X_test, y_test, y_pred, y_pred_proba = predictor.train_model(X, y, 'random_forest')
except Exception as e:
    st.error(f"Error training model: {str(e)}")
    st.stop()

st.title("Healthcare AI Project")
st.write("Welcome! This app predicts the risk of hospital readmission.")

# Patient Demographics Section
st.header("Patient Demographics")

age = st.number_input("Patient Age", min_value=0, max_value=120, value=50, step=1)
gender = st.selectbox("Patient Gender", ["Male", "Female"])
insurance_type = st.selectbox("Insurance Type", ["Medicare", "Medicaid", "Private", "Uninsured"])
living_situation = st.selectbox("Living Situation", ["Alone", "With Family", "Assisted Living"])


# Age is now directly input as a number


# Clinical History Section
st.header("Clinical History")

# Diagnosis options with search functionality
diagnosis_options = [
    "Septicemia (sepsis)",
    "Heart Failure",
    "Diabetes Mellitus with Complication",
    "Acute and Unspecified Renal Failure",
    "Schizophrenia Spectrum & Other Psychotic Disorders",
    "Pneumonia (except TB)",
    "COVID-19 (index year 2020)",
    "Cardiac Dysrhythmias",
    "Chronic Obstructive Pulmonary Disease (COPD) & Bronchiectasis",
    "Respiratory Failure / Insufficiency / Arrest",
    "Acute Myocardial Infarction (AMI)",
    "Alcohol-Related Disorders",
    "Urinary Tract Infections (UTI)",
    "Fluid & Electrolyte Disorders",
    "Complication of Select Surgical or Medical Care / Injury (Initial Encounter)",
    "Cerebral Infarction (Ischemic Stroke)",
    "Depressive Disorders",
    "Gastrointestinal Hemorrhage",
    "Skin & Subcutaneous Tissue Infections",
    "Chronic Kidney Disease (CKD)"
]

primary_diagnosis = st.selectbox(
    "Primary Diagnosis (start typing to search)",
    options=diagnosis_options
)

# Additional clinical fields
chronic_conditions = st.number_input("Number of Chronic Conditions", min_value=0, max_value=15, value=2, step=1)
comorbidity_score = st.number_input("Charlson Comorbidity Index (CCI)", min_value=0, max_value=15, value=3, step=1)
prev_hospitalizations = st.number_input("Previous Hospitalizations (12 months)", min_value=0, max_value=20, value=2, step=1)
recent_readmission = st.selectbox("Readmission in Last 30 Days?", ["Yes", "No"])



# Hospital Stay Section
st.header("Hospital Stay")
length_of_stay = st.slider("Length of Stay (days)", 1, 50, 10)
num_procedures = st.slider("Number of Procedures", 0, 10, 2)
num_medications = st.slider("Number of Medications Prescribed", 1, 20, 5)
emergency_admission = st.selectbox("Emergency Admission?", ["Yes", "No"])

# Labs & Vitals at Discharge Section
st.header("Labs & Vitals at Discharge")
hemoglobin = st.slider("Hemoglobin Level (g/dL)", 0, 20, 10)
creatinine = st.slider("Creatinine Level (mg/dL)", 0, 10, 5)
glucose = st.slider("Glucose Level (mg/dL)", 0, 300, 100)
albumin = st.slider("Albumin Level (g/dL)", 0, 6, 3)
wbc = st.slider("White Blood Cell Count (x10^9/L)", 0, 30, 7)

# Discharge & Follow-Up Section
st.header("Discharge & Follow-Up")
discharge_destination = st.selectbox("Discharge Destination", ["Home", "Skilled Nursing Facility", "Rehabilitation", "Hospice", "Against Medical Advice"])
follow_up = st.selectbox("Follow-Up Appointment Within 7 Days?", ["Yes", "No"])
discharge_instructions = st.selectbox("Discharge Instructions Provided?", ["Yes", "No"])

# Convert Yes/No responses to numeric values for ML model
emergency_admission_numeric = 1 if emergency_admission == "Yes" else 0
recent_readmission_numeric = 1 if recent_readmission == "Yes" else 0
follow_up_numeric = 1 if follow_up == "Yes" else 0
discharge_instructions_numeric = 1 if discharge_instructions == "Yes" else 0

# Collected the data from the user (only features the model was trained on)
custom_patient = {
    'age': age,                    
    'gender': gender,    
    'primary_diagnosis': primary_diagnosis,
    'length_of_stay': length_of_stay,
    'num_medications': num_medications,
    'num_procedures': num_procedures,
    'emergency_admission': emergency_admission_numeric,     
    'insurance_type': insurance_type,
    'prev_hospitalizations': prev_hospitalizations,   
    'chronic_conditions': chronic_conditions,      
    'discharge_destination': discharge_destination, 
    'hemoglobin': hemoglobin,           
    'creatinine': creatinine,            
    'glucose': glucose
}

# Make prediction
prediction, probability = predictor.predict_new_patient(custom_patient)

if st.button("Predict Readmission Risk"):
    st.write("=" * 50)
    st.write("**PATIENT PROFILE**")
    st.write("=" * 50)
    
    # Demographics
    st.write("**Demographics:**")
    st.write(f"   • Age: {custom_patient['age']} years")
    st.write(f"   • Gender: {custom_patient['gender']}")
    st.write(f"   • Insurance Type: {custom_patient['insurance_type']}")
    st.write(f"   • Living Situation: {living_situation}")
    
    # Clinical History
    st.write("\n**Clinical History:**")
    st.write(f"   • Primary Diagnosis: {custom_patient['primary_diagnosis']}")
    st.write(f"   • Chronic Conditions: {custom_patient['chronic_conditions']}")
    st.write(f"   • Charlson Comorbidity Index: {custom_patient['comorbidity_score']}")
    st.write(f"   • Previous Hospitalizations (12 months): {custom_patient['prev_hospitalizations']}")
    st.write(f"   • Recent Readmission (30 days): {recent_readmission}")
    
    # Hospital Stay
    st.write("\n**Hospital Stay:**")
    st.write(f"   • Length of Stay: {custom_patient['length_of_stay']} days")
    st.write(f"   • Number of Procedures: {custom_patient['num_procedures']}")
    st.write(f"   • Medications Prescribed: {custom_patient['num_medications']}")
    st.write(f"   • Emergency Admission: {emergency_admission}")
    
    # Labs & Vitals
    st.write("\n**Labs & Vitals at Discharge:**")
    st.write(f"   • Hemoglobin: {custom_patient['hemoglobin']} g/dL")
    st.write(f"   • Creatinine: {custom_patient['creatinine']} mg/dL")
    st.write(f"   • Glucose: {custom_patient['glucose']} mg/dL")
    st.write(f"   • Albumin: {custom_patient['albumin']} g/dL")
    st.write(f"   • White Blood Cell Count: {custom_patient['wbc']} x10^9/L")
    
    # Discharge & Follow-Up
    st.write("\n**Discharge & Follow-Up:**")
    st.write(f"   • Discharge Destination: {custom_patient['discharge_destination']}")
    st.write(f"   • Follow-Up Within 7 Days: {follow_up}")
    st.write(f"   • Discharge Instructions Provided: {discharge_instructions}")
    
    st.write("=" * 50)
    st.write("Prediction Results:")
    st.write("=" * 50)
    st.write(f"Predicted readmission risk score: {prediction:.2f}")
    st.write(f"Predicted readmission probability: {probability:.1%}")
    st.write("=" * 50)
    
    if prediction == 1:
        st.write("=" * 50)
        st.write("Readmission Risk: High")
        st.write("=" * 50)
        st.write("Recommendations:")
        st.write("=" * 50)
        st.write("1. Enhanced discharge planning")
        st.write("2. Follow-up appointment within 7 days")
        st.write("3. Medication reconciliation")
        st.write("4. Care coordination with primary care")
        st.write("5. Home health services if needed") 
        st.write("=" * 50)
    else:
        st.write("=" * 50)
        st.write("Readmission Risk: Low")
        st.write("=" * 50)
        st.write("Recommendations:")
        st.write("=" * 50)
        st.write("1. Regular discharge planning")
        st.write("2. Standard follow-up appointment")
        st.write("3. Patient education materials")
        st.write("=" * 50)
        
   
        
        