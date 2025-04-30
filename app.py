from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load model
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(model_path)

@app.route('/')
def home():
    return "Welcome to PayPal Fraud Detection API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Correct and complete list of features
    columns = [
        'step', 'amount', 'oldbalanceorg', 'newbalanceorig', 'oldbalancedest', 'newbalancedest',
        'isflaggedfraud', 'type_encoded', 'error_balance_orig', 'error_balance_dest',
        'orig_error_flag', 'dest_error_flag'
    ]

    try:
        input_df = pd.DataFrame([data['features']], columns=columns)
        prediction = model.predict(input_df)[0]
        result = "Fraud" if prediction == 1 else "Legit"
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
