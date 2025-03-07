from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API is working!"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    # Your ML model prediction logic here
    return jsonify({"prediction": "example result"})

if __name__ == "__main__":
    app.run(debug=True)
