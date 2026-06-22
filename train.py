import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder

# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv("dataset/Salary Data.csv")

# =========================
# 2. Clean Data
# =========================
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# =========================
# 3. Encode categorical columns
# =========================

# Gender
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

# Education
edu_map = {
    "Bachelor's": 0,
    "Master's": 1,
    "PhD": 2
}
df["Education Level"] = df["Education Level"].map(edu_map)

# Job Title → MUST ENCODE (THIS WAS MISSING)
from sklearn.preprocessing import LabelEncoder

job_encoder = LabelEncoder()
df["Job Title"] = job_encoder.fit_transform(df["Job Title"])

joblib.dump(job_encoder, "models/job_encoder.pkl")
# =========================
# 4. Features & Target
# =========================
X = df.drop("Salary", axis=1)
y = df["Salary"]

# Save column structure (VERY IMPORTANT)
os.makedirs("models", exist_ok=True)
joblib.dump(X.columns, "models/columns.pkl")

# =========================
# 5. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 6. Feature Scaling
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.pkl")

# =========================
# 7. Models
# =========================
models = {
    "linear_regression": LinearRegression(),
    "decision_tree": DecisionTreeRegressor(random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "knn": KNeighborsRegressor(n_neighbors=5)
}

# =========================
# 8. Train & Save Models
# =========================
results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    r2 = model.score(X_test_scaled, y_test)
    results[name] = r2

    joblib.dump(model, f"models/{name}.pkl")

# =========================
# 9. Print Results
# =========================
print("\nMODEL PERFORMANCE:")
for k, v in results.items():
    print(f"{k}: {v*100:.2f}% Accuracy")

print("\nTraining Completed Successfully!")