import json
import joblib


def extract_features(event):
    # Extract features based on the structure of your incoming event data
    # Modify this according to your actual data structure
    features = event.get('features', [])

    return [features]  # Assuming features is a list or array

def lambda_handler(event, context):
    model_logreg = joblib.load('models/model_logreg.joblib')
    model_rf = joblib.load('models/model_rf.joblib')
    model_xgb = joblib.load('models/model_xgb.joblib')

    features = extract_features(event)  # Define extract_features function based on your data

    prediction_logreg = model_logreg.predict(features)
    prediction_rf = model_rf.predict_proba(features)[:, 1]
    prediction_xgb = model_xgb.predict_proba(features)[:, 1]

    return {
        'statusCode': 200,
        'body': json.dumps({
            'prediction_logreg': prediction_logreg.tolist(),
            'prediction_rf': prediction_rf.tolist(),
            'prediction_xgb': prediction_xgb.tolist()
        })
    }
