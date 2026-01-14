import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("➡️ Loading processed dataset...")

df = pd.read_csv("dataset/processed/final_data.csv")

X = df.drop("Label", axis=1)
y = df["Label"]

print("Dataset shape:", df.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)

# Model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

print("➡️ Training Random Forest model...")
model.fit(X_train, y_train)

print("✅ Training completed")

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - INIDS")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

os.makedirs("results", exist_ok=True)
plt.savefig("results/confusion_matrix.png")
plt.show()

# Save model
os.makedirs("models/saved_models", exist_ok=True)
joblib.dump(model, "models/saved_models/intrusion_model.pkl")

print("✅ Model saved to models/saved_models/intrusion_model.pkl")
os.makedirs("models/saved_models", exist_ok=True)

joblib.dump(model, "models/saved_models/intrusion_model.pkl")

print("✅ Trained model saved successfully.")