# Credit Card Approval Prediction

An end-to-end machine learning project that automates credit card approval decisions using four classification algorithms, served through a Flask web application, and deployable to IBM Watson Machine Learning for cloud-based real-time predictions.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Business Scenarios](#business-scenarios)
- [Project Structure](#project-structure)
- [Dataset & Features](#dataset--features)
- [Feature Engineering](#feature-engineering)
- [Machine Learning Models](#machine-learning-models)
- [Model Results](#model-results)
- [Web Application](#web-application)
- [API Endpoint](#api-endpoint)
- [IBM Watson ML Deployment](#ibm-watson-ml-deployment)
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)

---

## Project Overview

Banks and financial institutions receive thousands of credit card applications every day. A significant portion are rejected due to factors such as high existing loan balances, insufficient income, or excessive credit inquiries. Manually reviewing each application is time-consuming and prone to human error.

This project automates the credit card approval decision using machine learning. By training a predictive model on historical applicant data, the system evaluates financial and demographic inputs to determine whether an applicant is likely to be **approved** or **rejected** — just as real banks do.

---

## Business Scenarios

### Scenario 1 — Automated Credit Card Application Screening
A bank credit analyst enters a new applicant's financial profile including income type, employment duration, and credit history into the web application. The model returns an instant approval or rejection prediction, enabling the analyst to prioritize applications that require further review.

### Scenario 2 — High-Risk Applicant Identification and Compliance Review
A financial compliance officer uses the platform to batch-screen applicants with past-due loan records. The feature engineering pipeline converts multi-class payment status codes into binary labels, allowing the model to clearly classify high-risk applicants as ineligible for a new credit card.

### Scenario 3 — Customer Self-Service Credit Card Eligibility Check
A prospective customer uses the web application to enter personal and financial details such as income level, employment status, and credit history before submitting a formal credit card application. The system instantly predicts the likelihood of approval, helping the customer understand their eligibility and reducing unnecessary application rejections.

---

## Project Structure

```
Credit Card Approval Prediction/
│
├── models/                  # Saved model artefacts (generated after training)
│   ├── best_model.pkl       # Best performing trained model
│   ├── scaler.pkl           # StandardScaler for Logistic Regression
│   ├── encoders.pkl         # LabelEncoders for categorical features
│   ├── features.pkl         # Ordered feature list
│   └── best_model_name.pkl  # Name of the best model
│
├── templates/
│   └── index.html           # Flask web UI
│
├── train_model.py           # ML training pipeline
├── app.py                   # Flask web application
├── watson_deploy.py         # IBM Watson ML deployment script
├── requirements.txt         # Python dependencies
└── .gitignore
```

---

## Dataset & Features

The dataset contains **2000 applicant records** with the following 14 input features:

| Feature | Type | Description |
|---|---|---|
| `Gender` | Categorical | M / F |
| `Car_Owner` | Categorical | Y / N — owns a car |
| `Property_Owner` | Categorical | Y / N — owns property |
| `Annual_Income` | Numerical | Annual income in USD |
| `Income_Type` | Categorical | Working / Commercial associate / Pensioner / State servant |
| `Education_Type` | Categorical | Higher education / Secondary / Incomplete higher / Lower secondary |
| `Family_Status` | Categorical | Married / Single / Separated / Widow |
| `Housing_Type` | Categorical | House/apartment / Rented apartment / With parents |
| `Employment_Duration` | Numerical | Years employed (negative value, e.g. -5 = 5 years) |
| `Occupation` | Categorical | Laborers / Core staff / Managers / Drivers / Sales staff |
| `Family_Members` | Numerical | Total number of family members |
| `Num_Children` | Numerical | Number of children |
| `Num_Credit_Inquiries` | Numerical | Number of credit inquiries in last 12 months |
| `Payment_Status` | Numerical | 0 = no debt, 1–5 = months overdue |

**Target variable:** `Approved` — 1 (Approved) or 0 (Rejected)

> To use a real dataset, replace the synthetic data block in `train_model.py` with:
> ```python
> data = pd.read_csv("your_dataset.csv")
> ```

---

## Feature Engineering

A key transformation applied before training:

```python
data["High_Risk"] = (data["Payment_Status"] >= 2).astype(int)
```

`Payment_Status` (0–5 months overdue) is converted into a binary `High_Risk` flag:
- `0` → Low risk (0 or 1 month overdue)
- `1` → High risk (2+ months overdue)

This directly supports **Scenario 2** — compliance officers can screen high-risk applicants based on past payment behaviour.

**Approval logic** is based on a scoring system:
- Annual income > $60,000
- Employment duration > 2 years
- Credit inquiries < 4
- Not high risk
- Owns property

Applicants scoring 3 or more out of 5 are labelled as **Approved**.

---

## Machine Learning Models

Four classification algorithms are trained and evaluated:

| Model | Description |
|---|---|
| **Logistic Regression** | Linear baseline classifier; features are scaled with StandardScaler |
| **Decision Tree** | Non-linear tree with max depth of 6 to prevent overfitting |
| **Random Forest** | Ensemble of 100 decision trees; robust to noise |
| **XGBoost** | Gradient boosting with 100 estimators; typically highest accuracy |

The best model by accuracy is automatically selected and saved to `models/best_model.pkl`.

---

## Model Results

Results on 20% held-out test set (400 samples):

| Model | Accuracy |
|---|---|
| Logistic Regression | 88.25% |
| Decision Tree | 98.50% |
| Random Forest | 98.25% |
| **XGBoost** | **99.75%** |

**XGBoost** was selected as the best model with **99.75% accuracy**.

```
XGBoost Classification Report:
              precision    recall  f1-score
   Rejected       1.00      0.99      1.00
   Approved       1.00      1.00      1.00
   accuracy                           1.00
```

---

## Web Application

The Flask web app provides an intuitive form-based UI for credit card approval prediction.

**Routes:**

| Route | Method | Description |
|---|---|---|
| `/` | GET | Renders the applicant input form |
| `/predict` | POST | Processes form input and returns prediction |
| `/api/predict` | POST | JSON API endpoint for programmatic access |

**How it works:**
1. Applicant details are entered into the form
2. Categorical fields are encoded using saved `LabelEncoders`
3. `Payment_Status` is converted to `High_Risk` binary flag
4. Features are passed to the loaded model
5. Prediction (Approved / Rejected) and confidence % are displayed

**Screenshot preview:**

The UI displays:
- Active model name badge
- Input form with all 14 fields
- Colour-coded result: green for Approved, red for Rejected
- Confidence percentage

---

## API Endpoint

The `/api/predict` endpoint accepts JSON for programmatic or Watson integration:

**Request:**
```json
POST /api/predict
Content-Type: application/json

{
  "Gender": "M",
  "Car_Owner": "Y",
  "Property_Owner": "Y",
  "Annual_Income": 85000,
  "Income_Type": "Working",
  "Education_Type": "Higher education",
  "Family_Status": "Married",
  "Housing_Type": "House / apartment",
  "Employment_Duration": -5.0,
  "Occupation": "Managers",
  "Family_Members": 3,
  "Num_Children": 1,
  "Num_Credit_Inquiries": 2,
  "Payment_Status": 0
}
```

**Response:**
```json
{
  "prediction": 1,
  "label": "Approved",
  "confidence": 97.43
}
```

---

## IBM Watson ML Deployment

`watson_deploy.py` deploys the best trained model to IBM Watson Machine Learning for scalable cloud-based predictions.

**Steps performed by the script:**
1. Authenticates with IBM Cloud using API key
2. Stores the trained model in the WML repository
3. Creates an online deployment endpoint
4. Tests the scoring endpoint with a sample payload

**Required environment variables:**

```bash
set WML_API_KEY=<your_ibm_cloud_api_key>
set WML_URL=https://us-south.ml.cloud.ibm.com
set WML_SPACE_ID=<your_deployment_space_guid>
```

**Run deployment:**
```bash
python watson_deploy.py
```

**Output:**
```
Model stored — UID: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
Deployment UID : xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
Scoring URL    : https://us-south.ml.cloud.ibm.com/ml/v4/deployments/.../predictions
Test prediction: [[1, 'Approved']]
```

---

## Installation & Setup

**Prerequisites:**
- Python 3.9+
- Git

**1. Clone the repository:**
```bash
git clone https://github.com/prasanna8118/Credit-Card-Approval-Prediction.git
cd Credit-Card-Approval-Prediction
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## How to Run

**Step 1 — Train the model:**
```bash
python train_model.py
```
This trains all 4 classifiers, prints evaluation reports, selects the best model, and saves all artefacts to the `models/` folder.

**Step 2 — Start the web app:**
```bash
python app.py
```
Open your browser at → `http://127.0.0.1:5000`

**Step 3 — (Optional) Deploy to IBM Watson ML:**
```bash
set WML_API_KEY=<your_key>
set WML_URL=https://us-south.ml.cloud.ibm.com
set WML_SPACE_ID=<your_space_id>
python watson_deploy.py
```

---

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.13 |
| ML Library | scikit-learn, XGBoost |
| Web Framework | Flask |
| Data Processing | pandas, NumPy |
| Model Persistence | joblib |
| Cloud Deployment | IBM Watson Machine Learning |
| Frontend | HTML5, CSS3 (Jinja2 templates) |
| Version Control | Git / GitHub |
