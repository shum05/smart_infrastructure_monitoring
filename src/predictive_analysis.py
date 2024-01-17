import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load your dataset
sensor_data_path = '../data/sensor_data.csv'
maintenance_records_path = '../data/maintenance_records.csv'
sensor_df = pd.read_csv(sensor_data_path)
maintenance_df = pd.read_csv(maintenance_records_path)

# Convert timestamp columns to datetime format
sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
maintenance_df['datetime_column'] = pd.to_datetime(maintenance_df['datetime_column'])

# Merge the datasets on timestamp and datetime_column
merged_df = pd.merge(sensor_df, maintenance_df, left_on='timestamp', right_on='datetime_column')

# Assuming 'maintenance_type' is the column to be used as the target variable
# Adjust 'maintenance_type' to the actual target column name
y = merged_df['maintenance_type']

# Extract features (X) from the merged dataset
X = merged_df.drop('maintenance_type', axis=1, errors='ignore')  # Adjust 'maintenance_type' to the actual target column name

# Extract hour and minute from timestamp
X['hour'] = pd.to_datetime(X['timestamp']).dt.hour
X['minute'] = pd.to_datetime(X['timestamp']).dt.minute

# Drop the original timestamp column
X = X.drop('datetime_column', axis=1)
X = X.drop('timestamp', axis=1)

# Create a rolling average feature
sensor_reading_columns = ['strain_gauge_reading', 'temperature', 'humidity', 'pressure', 'vibration']
your_window_size = 5

for column in sensor_reading_columns:
    X[f'{column}_rolling_avg'] = X[column].rolling(window=5).mean()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Rest of the code remains the same...
