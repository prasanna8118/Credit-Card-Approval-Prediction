"""
IBM Watson Machine Learning – Deployment Pipeline
--------------------------------------------------
Prerequisites:
  pip install ibm-watson-machine-learning joblib scikit-learn xgboost

Set environment variables before running:
  WML_API_KEY   – IBM Cloud API key
  WML_URL       – Watson ML service URL  (e.g. https://us-south.ml.cloud.ibm.com)
  WML_SPACE_ID  – Deployment space GUID
"""

import os
import joblib
from ibm_watson_machine_learning import APIClient

API_KEY  = os.environ["WML_API_KEY"]
WML_URL  = os.environ["WML_URL"]
SPACE_ID = os.environ["WML_SPACE_ID"]

wml_credentials = {"apikey": API_KEY, "url": WML_URL}
client = APIClient(wml_credentials)
client.set.default_space(SPACE_ID)

# ── Store the trained model ────────────────────────────────────────────────────
model      = joblib.load("models/best_model.pkl")
model_name = joblib.load("models/best_model_name.pkl")

SW_SPEC_NAME = "scikit-learn_1.1-py3.9"   # adjust to your WML runtime
sw_spec_id   = client.software_specifications.get_id_by_name(SW_SPEC_NAME)

model_meta = {
    client.repository.ModelMetaNames.NAME:              f"CreditCard-{model_name}",
    client.repository.ModelMetaNames.TYPE:              "scikit-learn_1.1",
    client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sw_spec_id,
}

stored = client.repository.store_model(model=model, meta_props=model_meta)
model_uid = client.repository.get_model_id(stored)
print(f"Model stored — UID: {model_uid}")

# ── Create online deployment ───────────────────────────────────────────────────
deploy_meta = {
    client.deployments.ConfigurationMetaNames.NAME:        "CreditCard-Approval-Online",
    client.deployments.ConfigurationMetaNames.ONLINE:      {},
}

deployment   = client.deployments.create(model_uid, meta_props=deploy_meta)
deploy_uid   = client.deployments.get_uid(deployment)
scoring_url  = client.deployments.get_scoring_href(deployment)

print(f"Deployment UID : {deploy_uid}")
print(f"Scoring URL    : {scoring_url}")

# ── Test scoring endpoint ──────────────────────────────────────────────────────
features = joblib.load("models/features.pkl")

sample_payload = {
    "input_data": [{
        "fields": features,
        "values": [[1, 1, 1, 85000, 0, 0, 0, 0, -5.0, 2, 3, 1, 2, 0]]
    }]
}

response = client.deployments.score(deploy_uid, sample_payload)
print("Test prediction:", response["predictions"][0]["values"])
