import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# ── Synthetic dataset (replace with real CSV via pd.read_csv) ──────────────────
np.random.seed(42)
N = 2000

data = pd.DataFrame({
    "Gender":              np.random.choice(["M", "F"], N),
    "Car_Owner":           np.random.choice(["Y", "N"], N),
    "Property_Owner":      np.random.choice(["Y", "N"], N),
    "Annual_Income":       np.random.uniform(20000, 200000, N),
    "Income_Type":         np.random.choice(["Working", "Commercial associate", "Pensioner", "State servant"], N),
    "Education_Type":      np.random.choice(["Higher education", "Secondary", "Incomplete higher", "Lower secondary"], N),
    "Family_Status":       np.random.choice(["Married", "Single", "Separated", "Widow"], N),
    "Housing_Type":        np.random.choice(["House / apartment", "Rented apartment", "With parents"], N),
    "Employment_Duration": np.random.uniform(-20, 0, N),   # negative = years employed (DAYS format)
    "Occupation":          np.random.choice(["Laborers", "Core staff", "Managers", "Drivers", "Sales staff"], N),
    "Family_Members":      np.random.randint(1, 6, N),
    "Num_Children":        np.random.randint(0, 4, N),
    "Num_Credit_Inquiries":np.random.randint(0, 10, N),
    "Payment_Status":      np.random.choice([0, 1, 2, 3, 4, 5], N),  # 0=no debt, 1-5=overdue months
})

# Feature engineering: binary risk label from payment status
data["High_Risk"] = (data["Payment_Status"] >= 2).astype(int)

# Target: approval (1) or rejection (0)
approval_score = (
    (data["Annual_Income"] > 60000).astype(int) +
    (data["Employment_Duration"] < -2).astype(int) +
    (data["Num_Credit_Inquiries"] < 4).astype(int) +
    (data["High_Risk"] == 0).astype(int) +
    (data["Property_Owner"] == "Y").astype(int)
)
data["Approved"] = (approval_score >= 3).astype(int)

# ── Preprocessing ──────────────────────────────────────────────────────────────
CATEGORICAL = ["Gender", "Car_Owner", "Property_Owner", "Income_Type",
               "Education_Type", "Family_Status", "Housing_Type", "Occupation"]

encoders = {}
for col in CATEGORICAL:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

FEATURES = ["Gender", "Car_Owner", "Property_Owner", "Annual_Income", "Income_Type",
            "Education_Type", "Family_Status", "Housing_Type", "Employment_Duration",
            "Occupation", "Family_Members", "Num_Children", "Num_Credit_Inquiries",
            "High_Risk"]

X = data[FEATURES]
y = data["Approved"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Train all four classifiers ─────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost":             XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=42),
}

results = {}
for name, model in models.items():
    X_tr = X_train_sc if name == "Logistic Regression" else X_train
    X_te = X_test_sc  if name == "Logistic Regression" else X_test
    model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"\n{'='*40}\n{name}  —  Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=["Rejected", "Approved"]))

# ── Select best model ──────────────────────────────────────────────────────────
best_name = max(results, key=results.get)
best_model = models[best_name]
print(f"\nBest model: {best_name} ({results[best_name]:.4f})")

# ── Save artefacts ─────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(scaler,     "models/scaler.pkl")
joblib.dump(encoders,   "models/encoders.pkl")
joblib.dump(FEATURES,   "models/features.pkl")
joblib.dump(best_name,  "models/best_model_name.pkl")
print("Artefacts saved to models/")
