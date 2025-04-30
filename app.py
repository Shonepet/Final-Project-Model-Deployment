from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Absolute path fix
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(model_path)

@app.route('/')
def home():
    import os
    model_exists = os.path.exists(os.path.join(os.path.dirname(__file__), 'model.pkl'))
    return f"Model found: {model_exists}"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(input_features)[0]
    result = "Fraud" if prediction == 1 else "Legit"
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)

