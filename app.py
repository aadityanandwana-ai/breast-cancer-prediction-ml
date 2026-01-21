import streamlit as st
import joblib
import numpy as np

# Load trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Breast Cancer Survival Prediction (Full Feature Model)")

st.write("Enter patient details:")

# 34 inputs (numerical / encoded)
inputs = []

feature_names = [
    "Patient ID",
    "Age at Diagnosis",
    "Type of Breast Surgery",
    "Cancer Type",
    "Cancer Type Detailed",
    "Cellularity",
    "Chemotherapy",
    "Pam50 + Claudin-low subtype",
    "Cohort",
    "ER status measured by IHC",
    "ER Status",
    "Neoplasm Histologic Grade",
    "HER2 status measured by SNP6",
    "HER2 Status",
    "Tumor Other Histologic Subtype",
    "Hormone Therapy",
    "Inferred Menopausal State",
    "Integrative Cluster",
    "Primary Tumor Laterality",
    "Lymph nodes examined positive",
    "Mutation Count",
    "Nottingham prognostic index",
    "Oncotree Code",
    "Overall Survival (Months)",
    "Overall Survival Status",
    "PR Status",
    "Radio Therapy",
    "Relapse Free Status (Months)",
    "Relapse Free Status",
    "Sex",
    "3-Gene classifier subtype",
    "Tumor Size",
    "Tumor Stage",
    "10yr_mortality"
]

for feature in feature_names:
    value = st.number_input(feature, value=0.0)
    inputs.append(value)

if st.button("Predict"):
    input_array = np.array(inputs).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("Patient likely to survive")
    else:
        st.error("High mortality risk detected")
