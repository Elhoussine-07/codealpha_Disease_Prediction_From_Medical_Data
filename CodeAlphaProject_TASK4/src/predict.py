import joblib
import numpy as np

def predict_disease(input_data, model_path="models/heart_disease_best_model.pkl"):
    model = joblib.load(model_path)
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction
