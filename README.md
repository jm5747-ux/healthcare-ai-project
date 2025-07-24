# Hospital Readmission Prediction System

## Project Overview

This ML project addresses a critical healthcare challenge in the United States: hospital readmissions within 30 days of discharge. Hospital readmissions are a major concern because they:

- Cost the US healthcare system $26 billion annually
- Lead to poor patient outcomes
- Result in Medicare penalties for hospitals with high readmission rates
- Indicate gaps in care coordination and discharge planning

## Problem Statement

The Hospital Readmission Reduction Program (HRRP) penalizes hospitals with higher-than-expected readmission rates for specific conditions. Our ML system helps identify patients at high risk of readmission, enabling healthcare providers to:

- Intervene early with targeted care plans
- Improve discharge planningfor high-risk patients
- Reduce healthcare costs and improve patient outcomes
- Avoid Medicare penalties

## Features

The system uses 14 key features that influence readmission risk:

### Patient Demographics
- Age
- Gender

### Clinical Factors
- Primary diagnosis (Heart Failure, Pneumonia, COPD, Diabetes, etc.)
- Length of stay
- Number of medications prescribed
- Number of procedures during stay
- Emergency admission status

### Medical History
- Previous hospitalizations in the last year
- Number of chronic conditions

### Discharge Information
- Discharge destination (Home, Skilled Nursing, Rehabilitation, Hospice)
- Insurance type (Medicare, Medicaid, Private, Uninsured)

### Lab Values
- Hemoglobin levels
- Creatinine levels
- Glucose levels

## Technical Implementation

### Libraries Used
- panda: Data manipulation and analysis
- numpy: Numerical computations
- scikit-learn**: Machine learning algorithms
- matplotlib & seaborn: Data visualization

### ML Models
- Random Forest Classifier: Primary model for prediction
- Logistic Regression: Alternative model for comparison

### Key Features
- Data Generation: Realistic dummy data based on real-world patterns
- Data Preprocessing: Encoding categorical variables, scaling numerical features
- Model Training: Cross-validation and hyperparameter tuning
- Model Evaluation: Accuracy, ROC-AUC, confusion matrix, classification report
- Feature Importance: Understanding which factors most influence readmission risk
- Prediction Interface: Easy-to-use function for new patient predictions

## Installation & Usage

##1. Install Dependencies
```bash
pip install -r requirements.txt
```
##2. Execution Steps
Run the following steps from notebook "hospitable_readmission.ipynb":
- Setup and Imports
- Data Generation
- Data Exploration and Visualization
- Machine Learning Model
- Model Evaluation
- Feature Importance Analysis
- Patient Risk Assessment with scemarios
- Custom Patient Assessment with scenario

##3. Expected Output
The system will:
- Generate 1000 realistic patient records
- Train a Random Forest model
- Display model performance metrics
- Show feature importance analysis
- Provide example predictions for new patients
- Save visualization plots (confusion matrix, ROC curve, feature importance)

##4. Model Performance
The model typically achieves:
- Accuracy: ~75-80%
- ROC-AUC: ~0.75-0.80
- Precision: ~70-75% for readmission prediction
- Recall: ~65-70% for identifying high-risk patients