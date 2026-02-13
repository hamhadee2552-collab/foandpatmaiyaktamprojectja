import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =====================================
# 1) LOAD DATASET
# =====================================

csv_path = "potato_blight_environment_dataset.csv"

print("\n===== LOADING DATA =====")

if not os.path.exists(csv_path):
    raise FileNotFoundError("CSV file not found!")

df = pd.read_csv(csv_path)

print("Dataset shape:", df.shape)
print(df.head())

# =====================================
# 2) CHECK DATA QUALITY
# =====================================

print("\n===== DATA SUMMARY =====")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

# =====================================
# 3) CLASS DISTRIBUTION
# =====================================

print("\n===== CLASS DISTRIBUTION =====")
print(df["Risk"].value_counts())

df["Risk"].value_counts().plot(kind="bar", color="pink")
plt.title("Risk Class Distribution")
plt.show()

# =====================================
# 4) SPLIT FEATURES / LABEL
# =====================================

X = df[["Temp_C","Humidity_%","Rain_mm","LeafWet_hours","PlantAge_days"]]
y = df["Risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# =====================================
# 5) TRAIN MODEL
# =====================================

print("\n===== TRAINING MODEL =====")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =====================================
# 6) MODEL PERFORMANCE
# =====================================

print("\n===== MODEL PERFORMANCE =====")

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =====================================
# 7) CONFUSION MATRIX
# =====================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="pink")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =====================================
# 8) CROSS VALIDATION
# =====================================

print("\n===== CROSS VALIDATION =====")

cv_scores = cross_val_score(model, X, y, cv=5)

print("CV Scores:", cv_scores)
print("Mean CV:", cv_scores.mean())

# =====================================
# 9) FEATURE IMPORTANCE
# =====================================

importance = model.feature_importances_

plt.figure(figsize=(6,4))
plt.barh(X.columns, importance, color="deeppink")
plt.title("Feature Importance")
plt.show()

# =====================================
# 10) SAMPLE EXPLAINABILITY
# =====================================

print("\n===== SAMPLE PREDICTION =====")

sample = X_test.iloc[0:1]

print("Input:")
print(sample)

print("Prediction:", model.predict(sample))

# =====================================
# 11) SAVE MODEL
# =====================================

joblib.dump(model, "potato_blight_model.pkl")

print("\nModel saved successfully!")
print("\n🎉 PROGRAM FINISHED 🎉")
