import csv
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
X = []
y = []

with open("face_shape_dataset.csv", newline='') as file:
    reader = csv.DictReader(file)
    for row in reader:
        X.append([
            float(row['jaw_width']),
            float(row['face_height']),
            float(row['cheekbone_width']),
            float(row['chin_angle'])
        ])
        y.append(row['label'])

# Split into training/testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train k-NN classifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"[✅] Accuracy: {accuracy:.2f}")

# Save the model
with open("face_shape_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("[✅] Model saved to face_shape_model.pkl")
