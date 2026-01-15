from predictor import predict_attack
import joblib
import pandas as pd

df = pd.read_csv("dataset/processed/final_data.csv")

X = df.drop("Label", axis=1)
sample = X.iloc[0].tolist()

print("Sample input length:", len(sample))
print("Prediction:", predict_attack(sample))