import pandas as pd

def load_data():
    sensor_data = pd.read_csv('data/sensor_data.csv')
    maintenance_records = pd.read_csv('data/maintenance_records.csv')
    return sensor_data, maintenance_records
