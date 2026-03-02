import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# -----------------------------
# STEP 1: Generate Synthetic Data
# -----------------------------

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

# Placement logic (rule-based)
df["Placed"] = (
    (df["CGPA"] > 7) &
    (df["AptitudeScore"] > 60) &
    (df["TechnicalSkill"] > 5)
).astype(int)

print("Dataset Created ✅")
print(df.head())

# -----------------------------
# STEP 2: Define Features & Target
# -----------------------------

X = df.drop("Placed", axis=1)
y = df["Placed"]

# -----------------------------
# STEP 3: Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# STEP 4: Train Model
# -----------------------------

model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------
# STEP 5: Evaluate Model
# -----------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nModel Trained Successfully ✅")
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)

# -----------------------------
# STEP 6: Save Model
# -----------------------------

pickle.dump(model, open("model/model.pkl", "wb"))

print("\nModel Saved Successfully ✅")