import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

print(" Loading dataset...")

file_path = "dataset/raw/friday.csv.csv"   # change if needed
df = pd.read_csv(file_path)

print("Initial shape:", df.shape)

# Clean column names
df.columns = df.columns.str.strip()

# Replace infinite values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with missing values
df.dropna(inplace=True)

print("After removing NaN & Inf:", df.shape)

# Encode labels
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])

# Save label encoder
os.makedirs("models/saved_models", exist_ok=True)
joblib.dump(le, "models/saved_models/label_encoder.pkl")

# Separate features and target
X = df.drop('Label', axis=1)
y = df['Label']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, "models/saved_models/scaler.pkl")

# Final processed dataset
final_df = pd.DataFrame(X_scaled, columns=X.columns)
final_df['Label'] = y.values

# Save processed file
os.makedirs("dataset/processed", exist_ok=True)
final_df.to_csv("dataset/processed/final_data.csv", index=False)

print("✅ Preprocessing completed")
print("✅ Clean dataset saved to dataset/processed/final_data.csv")