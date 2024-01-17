# app.py (Flask Application)
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load Linear Regression model
linear_reg_model = joblib.load('models/model_linear_reg.joblib')

# Extract features from the form data
def extract_features_from_form(form_data):
    # Assume the form contains fields 'strain_gauge_reading', 'temperature', 'humidity', 'pressure', 'vibration'
    features = [
        float(form_data.get('strain_gauge_reading')),
        float(form_data.get('temperature')),
        float(form_data.get('humidity')),
        float(form_data.get('pressure')),
        float(form_data.get('vibration'))
    ]
    return [features]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from the form data
    features = extract_features_from_form(request.form)

    # Make predictions using Linear Regression model
    prediction_linear_reg = linear_reg_model.predict(features)

    return render_template('result.html', prediction_linear_reg=prediction_linear_reg[0])

if __name__ == '__main__':
    app.run(debug=True)
