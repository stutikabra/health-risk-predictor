import joblib
import numpy as np
import streamlit as st

# Load model
model = joblib.load('diabetes_model_logistic.pkl')

st.title("Diabetes Risk Predictor (Logistic Model)")


# Inputs
gender = st.selectbox('Gender', ['Female', 'Male', 'Other'])
age = st.number_input('Age', min_value=0, max_value=120, value=30)     
hypertension = st.selectbox('Hypertension (0 = No, 1 = Yes)', [0, 1])  
heart_disease = st.selectbox('Heart Disease (0 = No, 1 = Yes)', [0, 1])
smoking_history = st.selectbox('Smoking History', ['No Info', 'never', 
'ever', 'not current', 'former', 'current'])
bmi = st.number_input('BMI', min_value=10.0, max_value=60.0, value=25.0) 
hba1c = st.number_input('HbA1C Level (%)', min_value=3.0, max_value=20.0, 
value=5.5)
glucose = st.number_input('Blood Glucose (mg/dL)', min_value=50, 
max_value=300, value=100)

# One-hot encode gender, drop_first=True means 'Female' is baseline, so:
gender_Male = 1 if gender == 'Male' else 0
gender_Other = 1 if gender == 'Other' else 0

# One-hot encode smoking_history as you did in training (drop_first=True)
# This depends on how your training data got dummified; example:

smoking_ever = 1 if smoking_history == 'ever' else 0
smoking_never = 1 if smoking_history == 'never' else 0
smoking_not_current = 1 if smoking_history == 'not current' else 0
smoking_former = 1 if smoking_history == 'former' else 0
smoking_current = 1 if smoking_history == 'current' else 0
# 'No Info' and 'never' are baseline (all zeros)

features = np.array([[age, hypertension, heart_disease, bmi, hba1c, 
glucose,
                      gender_Male, gender_Other, smoking_current, 
smoking_ever,
                      smoking_former, smoking_never, 
smoking_not_current]])

# Predict
if st.button('Predict Risk'):
    risk = model.predict_proba(features)[0][1]
    st.write(f"Predicted risk of diabetes: {risk:.2%}")

