from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return "Welcome to PayPal Fraud Detection API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(input_features)[0]
    result = "Fraud" if prediction == 1 else "Legit"
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
