import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Load your features and labels (replace with your actual loading method)
# Example:
# features = [[...], [...], ...]
# labels = ["Oval", "Heart", "Round", ...]

# Assuming your CSV looks like: feature1, feature2, ..., featureN, label
df = pd.read_csv("face_shape_dataset.csv")

# Separate features and labels
features = df.drop(columns=["label"]).values
labels = df["label"].values

# 🔹 Standardize features for better performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Save the scaler for use during prediction
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# 🔹 Split dataset for training/testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, labels, test_size=0.2, random_state=42
)

# 🔹 Train Random Forest model with improved parameters
model = RandomForestClassifier(
    n_estimators=200,         # 💡 Increase trees for stability
    max_depth=20,             # 💡 Limit depth to prevent overfitting
    min_samples_split=4,      # 💡 More conservative splits
    class_weight='balanced',  # 💡 Handle imbalanced classes
    random_state=42
)
model.fit(X_train, y_train)

# Save trained model
with open("face_shape_model.pkl", "wb") as f:
    pickle.dump(model, f)

# 🔎 Evaluate model
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("✅ Training Accuracy:", accuracy_score(y_train, y_pred_train))
print("✅ Testing Accuracy:", accuracy_score(y_test, y_pred_test))
print("\n📊 Classification Report on Test Set:\n", classification_report(y_test, y_pred_test))

# 💡 Optional: Display feature importance
import matplotlib.pyplot as plt

importances = model.feature_importances_
plt.figure(figsize=(10, 4))
plt.title("Feature Importances")
plt.bar(range(len(importances)), importances)
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
