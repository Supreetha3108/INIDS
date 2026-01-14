import pandas as pd
import joblib
import os

# =========================
# PATHS
# =========================
DATA_PATH = "dataset/processed/final_data.csv"
MODEL_PATH = "models/saved_models/intrusion_model.pkl"
SCALER_PATH = "models/saved_models/scaler.pkl"
ENCODER_PATH = "models/saved_models/label_encoder.pkl"

# =========================
# LOAD FILES
# =========================
print("🔄 Loading files...")

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)

print("✅ Dataset and model loaded successfully")

# =========================
# SPLIT FEATURES & LABEL
# =========================
X = df.drop("Label", axis=1)
y = df["Label"]

X_scaled = scaler.transform(X)

# =========================
# ATTACK BASE RISK MAP
# =========================
ATTACK_RISK_MAP = {
    "Normal": 5,
    "DoS": 90,
    "DDoS": 95,
    "PortScan": 40,
    "Brute Force": 65,
    "Infiltration": 80,
    "Bot": 85,
    "Web Attack": 70
}

# =========================
# RISK LEVEL FUNCTION
# =========================
def get_risk_level(score):
    if score >= 80:
        return "CRITICAL 🔴"
    elif score >= 60:
        return "HIGH 🟠"
    elif score >= 30:
        return "MEDIUM 🟡"
    else:
        return "LOW 🟢"

# =========================
# PREDICTION + RISK ENGINE
# =========================
print("\n🚨 REAL-TIME RISK ANALYSIS REPORT\n")

predictions = model.predict(X_scaled)
probabilities = model.predict_proba(X_scaled)
from collections import Counter

attack_names = label_encoder.inverse_transform(predictions)

risk_levels_all = []
attack_counter = Counter()

for i in range(len(attack_names)):
    attack = attack_names[i]
    confidence = max(probabilities[i])
    base_risk = ATTACK_RISK_MAP.get(attack, 50)
    risk_score = base_risk * confidence

    level = get_risk_level(risk_score)
    risk_levels_all.append(level)
    attack_counter[attack] += 1

print("\n📊 ========== FULL DATASET ANALYSIS ==========\n")

print("🔹 Total records analyzed :", len(attack_names), "\n")

print("🔹 Attack distribution:")
for k, v in attack_counter.items():
    print(f"   {k:15s} : {v}")

print("\n🔹 Risk level distribution:")
print(Counter(risk_levels_all))

critical_count = risk_levels_all.count("CRITICAL 🔴")
high_count = risk_levels_all.count("HIGH 🟠")

print(f"\n🚨 High risk traffic (HIGH + CRITICAL): {high_count + critical_count}")
print("=============================================\n")

# only first 10 samples (to avoid huge output)
for i in range(10):
    pred_label = predictions[i]
    attack_type = label_encoder.inverse_transform([pred_label])[0]
    confidence = max(probabilities[i])

    base_risk = ATTACK_RISK_MAP.get(attack_type, 50)
    risk_score = round(base_risk * confidence, 2)
    risk_level = get_risk_level(risk_score)

    print("====================================")
    print("Sample :", i+1)
    print("Attack Type :", attack_type)
    print("Confidence  :", round(confidence, 3))
    print("Risk Score  :", risk_score)
    print("Risk Level  :", risk_level)

print("\n✅ Risk analysis completed")