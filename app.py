from flask import Flask, request, jsonify, render_template
from utils import load_model_from_pkl
import numpy as np

app = Flask(__name__)

model, scaler = load_model_from_pkl("ecg_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json.get("data")
    if data is None:
        return jsonify({"error": "Aucune donnée fournie"}), 400

    arr = np.array(data, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    arr = scaler.transform(arr)
    preds = model.predict(arr)
    classes = (preds >= 0.5).astype(int).flatten().tolist()
    return jsonify({"probabilities": preds.flatten().tolist(), "predictions": classes})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
