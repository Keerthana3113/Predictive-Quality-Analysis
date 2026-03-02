import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# ===============================
# STEP 1: Load Dataset
# ===============================
df = pd.read_csv("dataset33.csv")

print("\n--- Dataset Preview ---")
print(df.head())

# ===============================
# STEP 2: Split Features & Target
# (Last column assumed as defect type)
# ===============================
X = df.iloc[:, :-1]   # Input parameters
y = df.iloc[:, -1]    # Defect type

# ===============================
# STEP 3: Encode Defect Names
# ===============================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nDefect Classes:")
for i, defect in enumerate(label_encoder.classes_):
    print(f"{i} -> {defect}")

# ===============================
# STEP 4: Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# ===============================
# STEP 5: Train Model
# ===============================
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# ===============================
# STEP 6: Evaluate Model
# ===============================
y_pred = model.predict(X_test)

print("\n--- Model Accuracy ---")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# ===============================
# STEP 7: Save Model
# ===============================
with open("model.pkl", "wb") as f:
    pickle.dump((model, label_encoder), f)

print("\nModel saved as model.pkl")

# ===============================
# STEP 8: Sample Prediction
# ===============================
# CHANGE VALUES ACCORDING TO YOUR DATASET
sample_input = [X.iloc[0].values]  # taking first row as example

prediction = model.predict(sample_input)
predicted_defect = label_encoder.inverse_transform(prediction)

print("\n--- Sample Prediction ---")
print("Input Parameters:", sample_input)
print("Predicted Defect Type:", predicted_defect[0]