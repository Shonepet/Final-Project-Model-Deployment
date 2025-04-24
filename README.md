# PayPal Fraud Detection API

This is a Flask-based API deployed on Render.com to detect fraudulent transactions using a pre-trained XGBoost model.

## Endpoints

- `/` - Health check
- `/predict` - POST endpoint that accepts transaction features and returns prediction (Fraud or Legit)

## Input Format (JSON)
```json
{
  "features": [feature1, feature2, ..., featureN]
}
```

## Output Format (JSON)
```json
{
  "prediction": "Fraud" or "Legit"
}
```

## Deployment

Deployed using Render.com with the following command:
```
web: gunicorn app:app
```