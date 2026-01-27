from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

# Simple Flask app to expose a prediction endpoint
app = Flask(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "artifacts/preprocessor.joblib")

# Lazy-load artifacts
model = None
preprocessor = None

def load_artifacts():
    global model, preprocessor
    if model is None and os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    if preprocessor is None and os.path.exists(PREPROCESSOR_PATH):
        preprocessor = joblib.load(PREPROCESSOR_PATH)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    load_artifacts()
    if model is None or preprocessor is None:
        return jsonify({"error": "Model or preprocessor not loaded"}), 503

    payload = request.get_json(force=True)
    features = payload.get("features")
    if features is None:
        return jsonify({"error": "Missing 'features' in request body"}), 400

    try:
        X = np.array(features)
        X_proc = preprocessor.transform(X)
        preds = model.predict_proba(X_proc)[:, 1]
        return jsonify({"fraud_probability": preds.tolist()})
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
