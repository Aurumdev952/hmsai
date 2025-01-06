import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import mlflow
import os

os.environ['MLFLOW_TRACKING_URI'] = 'http://127.0.0.1:5000'
os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://127.0.0.1:9000'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'


# Calculate samples per week (7 days * 24 hours * 12 samples per hour)
SAMPLES_PER_WEEK = 7 * 24 * 12
THRESHOLD_FACTOR = 1.5

# def analyze_weekly_patterns(data_scaled, reconstructed_data):
#     # Reshape data into weeks
#     num_complete_weeks = len(data_scaled) // SAMPLES_PER_WEEK

#     # Truncate data to complete weeks
#     truncated_length = num_complete_weeks * SAMPLES_PER_WEEK
#     data_weeks = data_scaled[:truncated_length].reshape(num_complete_weeks, SAMPLES_PER_WEEK, -1)
#     reconstructed_weeks = reconstructed_data[:truncated_length].reshape(num_complete_weeks, SAMPLES_PER_WEEK, -1)

#     # Calculate weekly pattern similarity
#     weekly_differences = []
#     for week in range(num_complete_weeks):
#         # Calculate pattern difference for the week
#         week_diff = np.mean(np.abs(data_weeks[week] - reconstructed_weeks[week]))
#         weekly_differences.append(week_diff)

#     # Convert to array and identify anomalous weeks
#     weekly_differences = np.array(weekly_differences)
#     threshold = np.mean(weekly_differences) + (2 * np.std(weekly_differences))
#     anomalous_weeks = weekly_differences > threshold

#     # Convert weekly anomalies back to original time series
#     anomalies = np.repeat(anomalous_weeks, SAMPLES_PER_WEEK)

#     return anomalies, weekly_differences, threshold

# Load and process your data as before
INPUT_FILE = "preprocessed_features.csv"
new_data = pd.read_csv(INPUT_FILE, index_col=0)
scaler = MinMaxScaler()
new_data_scaled = scaler.fit_transform(new_data)

# Load and predict with model
model = mlflow.keras.load_model(model_uri="models:/DeployedModel/latest")
model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
new_data_reconstructed = model.predict(new_data_scaled)

reconstruction_errors = np.mean(np.square(new_data_scaled - new_data_reconstructed), axis=1)
threshold = np.mean(reconstruction_errors) + THRESHOLD_FACTOR * np.std(reconstruction_errors)
anomalies = reconstruction_errors > threshold
print("Anomalies using data from training set", np.sum(anomalies))
# Detect weekly pattern anomalies
# anomalies, weekly_differences, threshold = analyze_weekly_patterns(new_data_scaled, new_data_reconstructed)

ANOMALY_DURATION = 30  # Duration of anomalies in terms of time steps
ANOMALY_MAGNITUDE = 3  # Magnitude of the anomaly spikes

# Load and process your data as before
INPUT_FILE = "preprocessed_features.csv"
new_data = pd.read_csv(INPUT_FILE, index_col=0)
# Split data into training and testing sets

new_data_scaled = scaler.fit_transform(new_data)
X_train = new_data_scaled  # Use the entire dataset as training data to introduce anomalies
X_test = X_train  # In this case, we simulate anomalies in the training set itself

# Introduce time-based anomalies into the training data (spikes that happen over time)
num_anomalies = 2  # Number of anomaly events to create
for _ in range(num_anomalies):
    anomaly_start = np.random.randint(0, len(X_train) - ANOMALY_DURATION)  # Randomly select a start time
    anomaly_end = anomaly_start + ANOMALY_DURATION  # Define the end of the anomaly duration

    # Create spikes (anomalies) for a range of features
    anomaly_spike = np.random.uniform(ANOMALY_MAGNITUDE, ANOMALY_MAGNITUDE + 2, (anomaly_end - anomaly_start, X_train.shape[1]))

    # Add anomalies to the data (simulating spikes over time)
    X_train[anomaly_start:anomaly_end] += anomaly_spike
# Load and predict with model
model = mlflow.keras.load_model(model_uri="models:/DeployedModel/latest")
model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
new_data_reconstructed = model.predict(X_train)

# Detect weekly pattern anomalies
# anomalies, weekly_differences, threshold = analyze_weekly_patterns(new_data_scaled, new_data_reconstructed)
reconstruction_errors = np.mean(np.square(new_data_scaled - new_data_reconstructed), axis=1)
threshold = np.mean(reconstruction_errors) + THRESHOLD_FACTOR * np.std(reconstruction_errors)
anomalies = reconstruction_errors > threshold


print("Anomalies using data from augmented with artificial anomalies", np.sum(anomalies))

