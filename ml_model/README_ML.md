ML MODULE – INIDS

Files:
model.pkl – trained Random Forest model
scaler.pkl – StandardScaler
label_encoder.pkl – label encoder
predictor.py – prediction interface
feature_order.txt – exact feature order
encoding_details.txt – preprocessing info

Usage:
from predictor import predict_attack
predict_attack([f1, f2, ..., fn])

Output:
"normal" or "attack"