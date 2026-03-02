import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------------------------------
# STEP 1: Generate Synthetic Dataset
# ---------------------------------------------------

np.random.seed(42)
n = 500

data = {
    "CGPA": np.random.uniform(5.0, 9.5, n),
    "Internships": np.random.randint(0, 4, n),
    "Projects": np.random.randint(1, 6, n),
    "AptitudeScore": np.random.randint(40, 100, n),
    "CommunicationSkill": np.random.randint(1, 10, n),
    "TechnicalSkill": np.random.randint(1, 10, n),
}

df = pd.DataFrame(data)

# Create weighted score (realistic logic)

score = (
    0.30 * (df["CGPA"] / 10) +
    0.25 * (df["TechnicalSkill"] / 10) +
    0.20 * (df["AptitudeScore"] / 100) +
    0.15 * (df["CommunicationSkill"] / 10) +
    0.10 * ((df["Internships"] + df["Projects"]) / 10)
)

# Add randomness (noise)
noise = np.random.normal(0, 0.05, n)

probability = score + noise

# Convert to binary outcome
df["Placed"] = (probability > 0.6).astype(int)

print("Dataset Created ✅")

# ---------------------------------------------------
# STEP 2: Features & Target
# ---------------------------------------------------

X = df.drop("Placed", axis=1)
y = df["Placed"]

feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------
# STEP 3: Train Multiple Models
# ---------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

results = {}
conf_matrices = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    results[name] = acc
    conf_matrices[name] = cm
    
    print(f"{name} Accuracy: {round(acc*100,2)}%")

# ---------------------------------------------------
# STEP 4: Select Best Model
# ---------------------------------------------------

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print(f"\nBest Model: {best_model_name}")

# ---------------------------------------------------
# STEP 5: Save Everything
# ---------------------------------------------------

model_data = {
    "best_model_name": best_model_name,
    "best_model": best_model,
    "all_accuracies": results,
    "confusion_matrices": conf_matrices,
    "features": feature_names
}

pickle.dump(model_data, open("model/model.pkl", "wb"))

print("\nMulti-Model Training Completed ✅")