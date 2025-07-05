
# Example: How to load and use the saved models for new patient prediction

import pickle
import pandas as pd
import numpy as np

# Load the models
with open('heart_failure_models.pkl', 'rb') as f:
    models = pickle.load(f)

# Extract components
kmeans_model = models['kmeans_model']
scaler = models['scaler']
clustering_features = models['clustering_features']

# Example: Predict cluster for a new patient
def predict_patient_cluster(patient_data):
    """
    Predict cluster assignment for a new patient

    patient_data: dict with keys matching clustering_features
    """
    # Convert to DataFrame
    patient_df = pd.DataFrame([patient_data])

    # Select and scale features
    patient_features = patient_df[clustering_features]
    patient_scaled = scaler.transform(patient_features)

    # Predict cluster
    cluster = kmeans_model.predict(patient_scaled)[0]

    return cluster

# Example usage:
new_patient = {
    'age': 65,
    'ejection_fraction': 35,
    'serum_creatinine': 1.8,
    'serum_sodium': 135,
    'creatinine_phosphokinase': 250,
    'platelets': 200000
}

predicted_cluster = predict_patient_cluster(new_patient)
print(f"Patient assigned to Cluster: {predicted_cluster}")

# Load clinical profiles to get risk information
with open('heart_failure_data.pkl', 'rb') as f:
    data = pickle.load(f)

clinical_profiles = data['clinical_profiles']
patient_risk = clinical_profiles.loc[predicted_cluster, 'DEATH_EVENT']
print(f"Patient risk level: {patient_risk:.1%} mortality rate")
