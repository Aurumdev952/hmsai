import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Mock sensor data generation
np.random.seed(42)

# Generating synthetic time-series sensor data
def generate_mock_sensor_data(num_samples, seq_length):
    """
    Generates mock time-series sensor data for classification tasks.
    """
    data = []
    labels = []
    for _ in range(num_samples):
        base_temp = np.random.uniform(20, 30)  # Normal temperature range
        base_vibration = np.random.uniform(0.1, 0.3)  # Normal vibration range
        
        # Generating a time-series of sensor readings
        temp_series = base_temp + np.random.normal(0, 0.5, seq_length)
        vibration_series = base_vibration + np.random.normal(0, 0.05, seq_length)
        
        # Randomly introduce an anomaly
        if np.random.rand() > 0.7:
            anomaly_index = np.random.randint(seq_length // 2, seq_length)
            temp_series[anomaly_index:] += np.random.uniform(5, 10)  # Spike in temperature
            vibration_series[anomaly_index:] += np.random.uniform(0.5, 1.0)  # Spike in vibration
            labels.append(1)  # Anomaly
        else:
            labels.append(0)  # Normal
        
        # Combine features and store
        series_features = np.column_stack((temp_series, vibration_series))
        flattened_features = series_features.flatten()  # Flatten time-series for ML input
        data.append(flattened_features)
    
    return np.array(data), np.array(labels)

# Generate synthetic data
num_samples = 1000
seq_length = 50  # Length of each time-series
data, labels = generate_mock_sensor_data(num_samples, seq_length)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Validate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

accuracy, classification_rep