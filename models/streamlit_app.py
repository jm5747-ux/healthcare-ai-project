import streamlit as st

st.title("Healthcare AI Project")
st.write("Welcome! This app predicts the risk of hospital readmission.")

# Example inputs (replace with your actual model and logic)
age = st.slider("Patient Age", 0, 100, 50)
bmi = st.slider("Patient BMI", 10.0, 50.0, 25.0)

if st.button("Predict Readmission Risk"):
    # Dummy logic — replace with your real model's prediction
    risk_score = (bmi / age) * 10
    st.write(f"Predicted readmission risk score: {risk_score:.2f}")