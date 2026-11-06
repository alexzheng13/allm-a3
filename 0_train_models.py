# ===========================
# STEP 0: Train & Save Models
# ===========================
import pandas as pd
import numpy as np
import uuid
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

os.makedirs("model", exist_ok=True)

# Diabetes Model
print("Training Diabetes Model...")
diabetes = pd.read_csv("data/pima-indians-diabetes.data.csv",
                       names=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
                              "DiabetesPedigreeFunction", "Age", "Outcome"])
X_d, y_d = diabetes.drop("Outcome", axis=1), diabetes["Outcome"]

clf_diabetes = RandomForestClassifier().fit(X_d, y_d)
joblib.dump(clf_diabetes, "model/diabetes.pkl")

# Breast Cancer Model
print("Training Breast Cancer Model...")
breast = load_breast_cancer()
X_b = pd.DataFrame(breast.data, columns=breast.feature_names)
y_b = breast.target  # 0 = malignant, 1 = benign

clf_breast = RandomForestClassifier().fit(X_b, y_b)
joblib.dump(clf_breast, "model/breast.pkl")

# Heart Model
print("Training Heart Model...")
heart = pd.read_csv("data/heart.csv")
X_h, y_h = heart.drop("target", axis=1), heart["target"]

clf_heart = RandomForestClassifier().fit(X_h, y_h)
joblib.dump(clf_heart, "model/heart.pkl")

print("Models trained and saved in 'model/' folder.")