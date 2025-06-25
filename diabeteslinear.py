import numpy as np
import streamlit as st
import joblib

# Load the saved linear regression model
model = joblib.load('diabetes_model_linear.pkl')

st.title("Diabetes Risk Predictor (Linear Model)")

# Inputs from user
gender = st.selectbox('Gender', ['Female', 'Male', 'Other'])
age = st.number_input('Age', min_value=0, max_value=120, value=30)
hypertension = st.selectbox('Hypertension (0 = No, 1 = Yes)', [0, 1])
heart_disease = st.selectbox('Heart Disease (0 = No, 1 = Yes)', [0, 1])
smoking_history = st.selectbox('Smoking History', ['never', 'current', 
'former', 'not current'])
bmi = st.number_input('BMI', min_value=10.0, max_value=60.0, value=25.0)
hba1c = st.number_input('HbA1C Level (%)', min_value=3.0, max_value=20.0, 
value=5.5)
glucose = st.number_input('Blood Glucose (mg/dL)', min_value=50, 
max_value=300, value=100)

gender_Male = 1 if gender == 'Male' else 0
gender_Other = 1 if gender == 'Other' else 0

# Encode smoking history (single numeric input)
smoking_map = {
    'never': 0.0,    'ever': 0.0, 
    'former': 0.75,
    'not current': 0.75,
    'current': 1.0
}
smoking_encoded = smoking_map[smoking_history]

# Features in the correct order as used in training
features = np.array([[age, hypertension, 
heart_disease, smoking_encoded, bmi, hba1c, glucose, gender_Male, 
gender_Other]])

if st.button("Predict Risk"):
    risk = model.predict(features)[0]
    st.write(f"Predicted risk of diabetes: {risk:.2%}")

