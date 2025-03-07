import os
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model using full path
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get("study_hours")  
    if data is None:
        return jsonify({"error": "Missing 'study_hours' in request"}), 400
    
    prediction = model.predict([[data]])[0]
    return jsonify({"study_hours": data, "pass_exam": int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
