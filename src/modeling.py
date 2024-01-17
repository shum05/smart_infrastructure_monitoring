from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_models(sensor_data, maintenance_records):
    X = sensor_data[['rolling_avg', 'temperature', 'humidity']]
    y = maintenance_records['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Logistic Regression
    logreg_model = LogisticRegression()
    logreg_model.fit(X_train, y_train)

    # Random Forest
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)

    # XGBoost
    xgb_model = XGBClassifier()
    xgb_model.fit(X_train, y_train)

    # Save models
    joblib.dump(logreg_model, 'models/model_logreg.joblib')
    joblib.dump(rf_model, 'models/model_rf.joblib')
    joblib.dump(xgb_model, 'models/model_xgb.joblib')

    return logreg_model, rf_model, xgb_model

def evaluate_models(models, X_test, y_test):
    for model in models:
        predictions = model.predict(X_test)
        print(f"{model.__class__.__name__} Accuracy:", accuracy_score(y_test, predictions))
        print(f"Classification Report ({model.__class__.__name__}):\n", classification_report(y_test, predictions))
