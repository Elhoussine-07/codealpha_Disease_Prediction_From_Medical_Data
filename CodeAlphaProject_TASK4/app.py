import streamlit as st
import joblib
import numpy as np

# Charger le modèle
model = joblib.load('models/heart_disease_best_model.pkl')

st.title("Prédiction de Maladie Cardiaque")

# Exemple de champs (adaptez selon les features de votre modèle)
age = st.number_input("Âge", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sexe", ["Homme", "Femme"])
cp = st.selectbox("Type de douleur thoracique (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Pression artérielle au repos (trestbps)", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholestérol (chol)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Glycémie à jeun > 120 mg/dl (fbs)", [0, 1])
restecg = st.selectbox("Résultat électrocardiogramme au repos (restecg)", [0, 1, 2])
thalach = st.number_input("Fréquence cardiaque max atteinte (thalach)", min_value=60, max_value=220, value=150)
exang = st.selectbox("Angine induite par l’exercice (exang)", [0, 1])
oldpeak = st.number_input("Dépression ST induite par l’exercice (oldpeak)", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Pente du segment ST (slope)", [0, 1, 2])
ca = st.selectbox("Nombre de vaisseaux colorés (ca)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thal (thal)", [0, 1, 2, 3])

# Encodage du sexe
sex_encoded = 1 if sex == "Homme" else 0

# Prédiction
if st.button("Prédire"):
    features = np.array([[age, sex_encoded, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(features)[0]
    if prediction == 1:
        st.error("Le patient est susceptible d'avoir une maladie cardiaque.")
    else:
        st.success("Le patient n'est PAS susceptible d'avoir une maladie cardiaque.")

