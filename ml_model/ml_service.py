import joblib
import numpy as np

model = joblib.load("ml_model/model.pkl")
scaler = joblib.load("ml_model/scaler.pkl")
encoder = joblib.load("ml_model/encoder.pkl")

def predict_attack(features: list):
    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)

    pred = model.predict(features)[0]
    label = encoder.inverse_transform([pred])[0]
    confidence = max(model.predict_proba(features)[0])

    if label.lower() in ["normal", "benign"]:
        return "normal", confidence
    else:
        return "attack", confidence