import joblib
import numpy as np
import streamlit as st

# Load model
model = joblib.load('diabetes_model_rf.pkl')

st.title("Diabetes Risk Predictor (Random Forest)")

# User Inputs
gender = st.selectbox('Gender', ['Female', 'Male', 'Other'])
age = st.number_input('Age', min_value=0, max_value=120, value=30)
hypertension = st.selectbox('Hypertension (0 = No, 1 = Yes)', [0, 1])
heart_disease = st.selectbox('Heart Disease (0 = No, 1 = Yes)', [0, 1])
smoking_history = st.selectbox(
    'Smoking History',
    ['No Info', 'never', 'ever', 'not current', 'former', 'current']
)
bmi = st.number_input('BMI', min_value=10.0, max_value=60.0, value=25.0)
hba1c = st.number_input('HbA1C Level (%)', min_value=3.0, max_value=20.0, 
value=5.5)
glucose = st.number_input('Blood Glucose (mg/dL)', min_value=50, 
max_value=300, value=100)

# One-hot encoding (assuming drop_first=True in training)

gender_Male = 1 if gender == 'Male' else 0
gender_Other = 1 if gender == 'Other' else 0

smoking_map = {
    "No Info": 0.0,
    "never": 0.0,
    "ever": 0.0,
    "not current": 0.5,
    "former": 0.75,
    "current": 1.0
}
smoking_encoded = smoking_map[smoking_history]


# Match feature order used in training
features = np.array([[age, hypertension, heart_disease, smoking_encoded, 
bmi, hba1c, 
glucose, gender_Male, gender_Other]])

# Prediction
if st.button('Predict Risk'):
    risk = model.predict_proba(features)[0][1]
    st.write(f"Predicted risk of diabetes: {risk:.2%}")
