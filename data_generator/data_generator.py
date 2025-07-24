import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_hospital_data(n_samples=1000):
    """
    Generate realistic dummy data for hospital readmission prediction.
    Based on real-world factors that influence readmission risk.
    """
    np.random.seed(42)
    random.seed(42)
    
    # Patient demographics
    ages = np.random.normal(65, 15, n_samples).astype(int)
    ages = np.clip(ages, 18, 95)
    
    genders = np.random.choice(['Male', 'Female'], n_samples)
    
    # Medical conditions (common causes of readmission)
    conditions = ['Heart Failure', 'Pneumonia', 'COPD', 'Diabetes', 'Kidney Disease', 
                  'Stroke', 'Hip/Knee Surgery', 'Coronary Artery Disease']
    
    primary_diagnosis = np.random.choice(conditions, n_samples)
    
    # Length of stay (days)
    length_of_stay = np.random.exponential(5, n_samples).astype(int)
    length_of_stay = np.clip(length_of_stay, 1, 30)
    
    # Number of medications prescribed
    num_medications = np.random.poisson(8, n_samples)
    num_medications = np.clip(num_medications, 1, 20)
    
    # Number of procedures during stay
    num_procedures = np.random.poisson(2, n_samples)
    num_procedures = np.clip(num_procedures, 0, 8)
    
    # Emergency admission (higher readmission risk)
    emergency_admission = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    
    # Insurance type
    insurance_types = ['Medicare', 'Medicaid', 'Private', 'Uninsured']
    insurance = np.random.choice(insurance_types, n_samples, p=[0.4, 0.2, 0.3, 0.1])
    
    # Previous hospitalizations in last year
    prev_hospitalizations = np.random.poisson(1.5, n_samples)
    prev_hospitalizations = np.clip(prev_hospitalizations, 0, 10)
    
    # Chronic conditions count
    chronic_conditions = np.random.poisson(2, n_samples)
    chronic_conditions = np.clip(chronic_conditions, 0, 8)
    
    # Discharge destination
    discharge_destinations = ['Home', 'Skilled Nursing Facility', 'Rehabilitation', 'Hospice']
    discharge_destination = np.random.choice(discharge_destinations, n_samples, p=[0.6, 0.2, 0.15, 0.05])
    
    # Lab values (simplified)
    hemoglobin = np.random.normal(12, 2, n_samples)
    creatinine = np.random.normal(1.2, 0.5, n_samples)
    glucose = np.random.normal(140, 40, n_samples)
    
    # Create realistic readmission risk based on features
    readmission_risk = (
        (ages > 70) * 0.3 +
        (length_of_stay > 7) * 0.2 +
        (num_medications > 10) * 0.15 +
        (emergency_admission == 1) * 0.2 +
        (prev_hospitalizations > 2) * 0.25 +
        (chronic_conditions > 3) * 0.2 +
        (discharge_destination != 'Home') * 0.15 +
        (hemoglobin < 10) * 0.1 +
        (creatinine > 2) * 0.1 +
        (glucose > 200) * 0.1
    )
    
    # Add some randomness
    readmission_risk += np.random.normal(0, 0.1, n_samples)
    readmission_risk = np.clip(readmission_risk, 0, 1)
    
    # Generate readmission outcome (30-day readmission)
    readmitted_30_days = (readmission_risk > 0.5).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'patient_id': range(1, n_samples + 1),
        'age': ages,
        'gender': genders,
        'primary_diagnosis': primary_diagnosis,
        'length_of_stay': length_of_stay,
        'num_medications': num_medications,
        'num_procedures': num_procedures,
        'emergency_admission': emergency_admission,
        'insurance_type': insurance,
        'prev_hospitalizations': prev_hospitalizations,
        'chronic_conditions': chronic_conditions,
        'discharge_destination': discharge_destination,
        'hemoglobin': hemoglobin,
        'creatinine': creatinine,
        'glucose': glucose,
        'readmitted_30_days': readmitted_30_days
    })
    
    return data

