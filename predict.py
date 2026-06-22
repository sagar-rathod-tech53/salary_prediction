import os
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models
model = joblib.load(os.path.join(BASE_DIR, "models", "random_forest.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
columns = joblib.load(os.path.join(BASE_DIR, "models", "columns.pkl"))

print("💼 Salary Prediction System (CLI)")

# Inputs
age = float(input("Enter Age: "))
experience = float(input("Enter Years of Experience: "))

gender = input("Enter Gender (Male/Female): ")
education = input("Enter Education (Bachelor/Master/PhD): ")

# Encoding
gender_val = 1 if gender == "Male" else 0

edu_map = {"Bachelor": 0, "Master": 1, "PhD": 2}
education_val = edu_map.get(education, 0)

# Create empty vector
input_data = np.zeros(len(columns))
input_dict = dict(zip(columns, input_data))

# Fill values
if "Age" in input_dict:
    input_dict["Age"] = age

if "Years of Experience" in input_dict:
    input_dict["Years of Experience"] = experience

if "Gender" in input_dict:
    input_dict["Gender"] = gender_val

if "Education Level" in input_dict:
    input_dict["Education Level"] = education_val

# Final input
final_input = np.array(list(input_dict.values())).reshape(1, -1)

# Scale
final_scaled = scaler.transform(final_input)

# Predict
prediction = model.predict(final_scaled)

print("\n💰 Predicted Salary:", round(prediction[0], 2))