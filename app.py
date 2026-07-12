from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model    = joblib.load("models/best_model.pkl")
scaler   = joblib.load("models/scaler.pkl")
encoders = joblib.load("models/encoders.pkl")
features = joblib.load("models/features.pkl")
model_name = joblib.load("models/best_model_name.pkl")

CATEGORICAL = ["Gender", "Car_Owner", "Property_Owner", "Income_Type",
               "Education_Type", "Family_Status", "Housing_Type", "Occupation"]

def preprocess(form):
    row = {
        "Gender":               form["Gender"],
        "Car_Owner":            form["Car_Owner"],
        "Property_Owner":       form["Property_Owner"],
        "Annual_Income":        float(form["Annual_Income"]),
        "Income_Type":          form["Income_Type"],
        "Education_Type":       form["Education_Type"],
        "Family_Status":        form["Family_Status"],
        "Housing_Type":         form["Housing_Type"],
        "Employment_Duration":  float(form["Employment_Duration"]),
        "Occupation":           form["Occupation"],
        "Family_Members":       int(form["Family_Members"]),
        "Num_Children":         int(form["Num_Children"]),
        "Num_Credit_Inquiries": int(form["Num_Credit_Inquiries"]),
        "Payment_Status":       int(form["Payment_Status"]),
    }
    row["High_Risk"] = 1 if row["Payment_Status"] >= 2 else 0

    for col in CATEGORICAL:
        le = encoders[col]
        val = row[col]
        if val not in le.classes_:
            val = le.classes_[0]
        row[col] = int(le.transform([val])[0])

    X = np.array([[row[f] for f in features]])

    if model_name == "Logistic Regression":
        X = scaler.transform(X)

    return X


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", model_name=model_name)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        X = preprocess(request.form)
        pred  = int(model.predict(X)[0])
        proba = float(model.predict_proba(X)[0][pred]) * 100
        result = "Approved ✅" if pred == 1 else "Rejected ❌"
        return render_template("index.html", result=result, confidence=f"{proba:.1f}",
                               model_name=model_name, form_data=request.form)
    except Exception as e:
        return render_template("index.html", error=str(e), model_name=model_name)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON endpoint for programmatic / Watson integration."""
    try:
        data = request.get_json(force=True)
        X    = preprocess(data)
        pred  = int(model.predict(X)[0])
        proba = float(model.predict_proba(X)[0][pred]) * 100
        return jsonify({"prediction": pred,
                        "label": "Approved" if pred == 1 else "Rejected",
                        "confidence": round(proba, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
