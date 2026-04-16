from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load dict
data = pickle.load(open("final_leakage_free_model.pkl", "rb"))

model = data["model"]
features_list = data["features"]

@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.json

    try:
        # Build input in correct order
        features = []

        for f in features_list:
            features.append(float(input_data.get(f, 0)))

        prediction = model.predict([features])[0]

        probability = 0.75
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([features])[0]
            probability = float(np.max(probs))

        return jsonify({
            "risk": str(prediction),
            "confidence": probability
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)