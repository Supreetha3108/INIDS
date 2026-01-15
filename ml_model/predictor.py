import joblib

model = joblib.load("ml_model/model.pkl")
scaler = joblib.load("ml_model/scaler.pkl")
encoder = joblib.load("ml_model/encoder.pkl")

def predict_attack(features: list):
    features_scaled = scaler.transform([features])
    pred = model.predict(features_scaled)[0]
    label = encoder.inverse_transform([pred])[0]

    if label.upper() == "BENIGN":
        return "normal"
    else:
        return "attack"