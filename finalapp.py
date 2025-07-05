import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.title("Health Risk Predictor")

# Load models
diabetes_models = {
    "Linear Regression": joblib.load("diabetes_model_linear.pkl"),
    "Logistic Regression": joblib.load("diabetes_model_logistic.pkl"),
    "Polynomial Regression": joblib.load("diabetes_model_polynomial.pkl"),
    "Random Forest": joblib.load("diabetes_model_rf.pkl"),
    "Support Vector Regression": joblib.load("diabetes_model_svr.pkl")
}

heart_models = {
    "Support Vector Machine": joblib.load("heart_model_svm.pkl"),
    "Naive Bayes": joblib.load("heart_model_nb.pkl")
}

# Load transformers
poly = joblib.load("poly.pkl")  # for Polynomial Regression (diabetes)
scaler = joblib.load("scaler.pkl")  # for SVR and Heart Disease inputs

# Disease selection
disease = st.selectbox("Select Disease to Predict", ["Diabetes", "Heart Disease"])

if disease == "Diabetes":
    model_choice = st.selectbox("Select Diabetes Model", 
list(diabetes_models.keys()))
    model = diabetes_models[model_choice]

    # Diabetes inputs
    gender = st.selectbox('Gender', ['Female', 'Male', 'Other'])
    age = st.number_input('Age', min_value=0, max_value=120, value=30)
    hypertension = st.selectbox('Hypertension (0 = No, 1 = Yes)', [0, 1])
    heart_disease = st.selectbox('Heart Disease (0 = No, 1 = Yes)', [0, 
1])
    smoking_history = st.selectbox(
        'Smoking History',
        ['No Info', 'never', 'ever', 'not current', 'former', 'current']
    )
    bmi = st.number_input('BMI', min_value=10.0, max_value=60.0, 
value=25.0)
    hba1c = st.number_input('HbA1C Level (%)', min_value=3.0, 
max_value=20.0, value=5.5)
    glucose = st.number_input('Blood Glucose (mg/dL)', min_value=50, 
max_value=300, value=100)

    # One-hot encoding
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

    # Assemble features
    features = np.array([[age, hypertension, heart_disease, 
smoking_encoded,
                          bmi, hba1c, glucose, gender_Male, 
gender_Other]])

    # Transform features if needed
    if model_choice == "Polynomial Regression":
        features = poly.transform(features)
    if model_choice == "Support Vector Regression":
        features = scaler.transform(features)

    if st.button('Predict Diabetes Risk'):
        if model_choice in ["Logistic Regression", "Random Forest"]:
            risk = model.predict_proba(features)[0][1]
        else:
            risk = model.predict(features)[0]
            risk = np.clip(risk, 0, 1)
        st.write(f"Predicted risk of diabetes: **{risk:.2%}**")

elif disease == "Heart Disease":
    model_choice = st.selectbox("Select Heart Disease Model", 
list(heart_models.keys()))
    model = heart_models[model_choice]

    age = st.number_input("Age", 1, 120, 50)
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", options=[0, 1])
    cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", 
    options=[0, 1])
    restecg = st.selectbox("Resting ECG Results (restecg)", options=[0, 1, 
    2])
    thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 
    1])
    oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope of ST Segment (slope)", options=[0, 1, 2])
    ca = st.selectbox("Number of Major Vessels Colored (ca)", options=[0, 
    1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3])

    input_dict = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

    input_df = pd.DataFrame([input_dict])

    input_df['sex_0'] = (input_df['sex'] == 0).astype(int)
    input_df['sex_1'] = (input_df['sex'] == 1).astype(int)
    input_df = input_df.drop(columns=['sex'])

    feature_order = ['age', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'sex_0', 
    'sex_1']

    input_df = input_df[feature_order]
    input_scaled = scaler.transform(input_df)

    if st.button("Predict Heart Disease Risk"):
        prob = model.predict_proba(input_scaled)[0][1]
        st.write(f"Chance of Heart Disease: **{prob*100:.2f}%**")

