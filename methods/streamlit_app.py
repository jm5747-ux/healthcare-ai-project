import streamlit as st
import os
import re
import inspect
import numpy as np
import pandas as pd
import seaborn as sns
import ruptures as rpt
import matplotlib.pyplot as plt

import ml_model
import importlib
importlib.reload(ml_model)


# Load the dataset from the CSV file
df_hospital_readmission = pd.read_csv("datasets/hospital_readmission.csv")

# Initialize and train the model
print("Training machine learning model...")
predictor = ml_model.HospitalReadmissionPredictor()

# Preprocess the data
X, y = predictor.preprocess_data(df_hospital_readmission)
X_test, y_test, y_pred, y_pred_proba = predictor.train_model(X, y, 'random_forest')


# Load the model
#predictor = ml_model.load_model()

st.title("Healthcare AI Project")
st.write("Welcome! This app predicts the risk of hospital readmission.")

# Example inputs (replace with your actual model and logic)
age = st.slider("Patient Age", 0, 100, 50)
gender = st.selectbox("Patient Gender", ["Male", "Female"])
primary_diagnosis = st.selectbox("Primary Diagnosis", ["Heart Failure", "Pneumonia", "COPD", "Diabetes", "Kidney Disease", "Stroke", "Hip/Knee Surgery", "Coronary Artery Disease"])
length_of_stay = st.slider("Length of Stay", 1, 30, 10)
num_medications = st.slider("Number of Medications", 1, 20, 5)
num_procedures = st.slider("Number of Procedures", 0, 8, 2)
emergency_admission = st.selectbox("Emergency Admission", [0, 1])
insurance_type = st.selectbox("Insurance Type", ["Medicare", "Medicaid", "Private", "Uninsured"])
prev_hospitalizations = st.slider("Previous Hospitalizations", 0, 10, 2)
chronic_conditions = st.slider("Number of Chronic Conditions", 0, 8, 2)
discharge_destination = st.selectbox("Discharge Destination", ["Home", "Skilled Nursing Facility", "Rehabilitation", "Hospice"])
hemoglobin = st.slider("Hemoglobin Level", 0, 20, 10)
creatinine = st.slider("Creatinine Level", 0, 10, 5)
glucose = st.slider("Glucose Level", 0, 300, 100)

custom_patient = {
    'age': age,                    # Age (18-95)
    'gender': gender,    
    'primary_diagnosis': primary_diagnosis,
    'length_of_stay': length_of_stay,
    'num_medications': num_medications,
    'num_procedures': num_procedures,
    'emergency_admission': emergency_admission,     # Emergency admission (1=Yes, 0=No)
    'insurance_type': insurance_type,
    'prev_hospitalizations': prev_hospitalizations,   # Previous hospitalizations in last year (0-10)
    'chronic_conditions': chronic_conditions,      # Number of chronic conditions (0-8)
    'discharge_destination': discharge_destination, # 'Home', 'Skilled Nursing Facility', 'Rehabilitation', 'Hospice'
    'hemoglobin': hemoglobin,           # Hemoglobin level
    'creatinine': creatinine,            # Creatinine level
    'glucose': glucose              # Glucose level
}

# Make prediction
prediction, probability = predictor.predict_new_patient(custom_patient)

if st.button("Predict Readmission Risk"):
    st.write("=" * 50)
    st.write(f"Patient Profile:")
    st.write(f"   Age: {custom_patient['age']} years")
    st.write(f"   Gender: {custom_patient['gender']}")
    st.write(f"   Diagnosis: {custom_patient['primary_diagnosis']}")
    st.write(f"   Length of Stay: {custom_patient['length_of_stay']} days")
    st.write(f"   Medications: {custom_patient['num_medications']}")
    st.write(f"   Procedures: {custom_patient['num_procedures']}")
    st.write(f"   Emergency Admission: {'Yes' if custom_patient['emergency_admission'] else 'No'}")
    st.write(f"   Insurance Type: {custom_patient['insurance_type']}")
    st.write(f"   Previous Hospitalizations: {custom_patient['prev_hospitalizations']}")
    st.write(f"   Chronic Conditions: {custom_patient['chronic_conditions']}")
    st.write(f"   Discharge Destination: {custom_patient['discharge_destination']}")
    st.write(f"   Hemoglobin Level: {custom_patient['hemoglobin']}")
    st.write(f"   Creatinine Level: {custom_patient['creatinine']}")
    st.write(f"   Glucose Level: {custom_patient['glucose']}")
    
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
        
   
        
        