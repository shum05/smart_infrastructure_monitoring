def perform_feature_engineering(sensor_data):
    sensor_data['rolling_avg'] = sensor_data['strain_gauge_reading'].rolling(window=7).mean()
