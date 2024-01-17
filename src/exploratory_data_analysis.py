import seaborn as sns
import matplotlib.pyplot as plt

def perform_eda(sensor_data):
    selected_features = ['strain_gauge_reading', 'temperature', 'humidity', 'maintenance_indicator']
    sns.pairplot(sensor_data[selected_features], hue='maintenance_indicator')
    plt.show()

    correlation_matrix = sensor_data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()
