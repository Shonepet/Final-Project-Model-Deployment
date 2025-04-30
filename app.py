from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model using an absolute path
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(model_path)

# Homepage route
@app.route('/')
def home():
    return "Welcome to PayPal Fraud Detection API!"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Define the expected feature names in order
    columns = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

    try:
        # Convert input list to DataFrame
        input_df = pd.DataFrame([data['features']], columns=columns)

        # Make prediction
        prediction = model.predict(input_df)[0]
        result = "Fraud" if prediction == 1 else "Legit"
        return jsonify({'prediction': result})

    except Exception as e:
        # Catch and return any errors
        return jsonify({'error': str(e)}), 500

# Local run (ignored on Render)
if __name__ == '__main__':
    app.run(debug=True)
