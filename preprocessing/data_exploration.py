import pandas as pd

print("➡️ Program started...")
file_path = "dataset/raw/friday.csv.csv"   # keep your filename
print("➡️ Loading dataset...")

df = pd.read_csv(file_path)

print("✅ Dataset loaded successfully")

print("\n📊 Shape (rows, columns):")
print(df.shape)

print("\n🔹 First 5 rows:")
print(df.head())

print("\n🔹 Column names:")
print(df.columns)

print("\n🔹 Missing values (top 10 columns):")
print(df.isnull().sum().head(10))

print("\n🔹 Attack distribution:")
print(df[' Label'].value_counts())