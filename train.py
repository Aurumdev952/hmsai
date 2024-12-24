# %pip install pandas scikit-learn numpy matplotlib seaborn torch requests tensorflow torch plotly

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Generate time-series data with 1-minute intervals for a year
def generate_time_series_data(start_time, num_minutes, anomaly_probability=0.2):
    timestamps = pd.date_range(start=start_time, periods=num_minutes, freq='T')

    # Extract seasonal information
    day_of_year = timestamps.dayofyear
    hour_of_day = timestamps.hour

    # Define seasonal trends
    seasonal_temp = 10 * np.sin(2 * np.pi * day_of_year / 365.25) + np.random.uniform(-2, 2, num_minutes)  # Summer peak, winter low
    seasonal_volt = 0.5 * np.sin(2 * np.pi * day_of_year / 365.25) + np.random.uniform(-0.2, 0.2, num_minutes)  # Subtle seasonal effect
    seasonal_pitch = 1 * np.sin(2 * np.pi * hour_of_day / 24) + np.random.uniform(-0.5, 0.5, num_minutes)  # Daily pattern
    seasonal_roll = 1 * np.sin(2 * np.pi * hour_of_day / 24) + np.random.uniform(-0.5, 0.5, num_minutes)  # Daily pattern

    # Simulate normal ranges for each feature
    normal_temp = 70 + seasonal_temp  # Adjusted base for temperature
    normal_volt = 13 + seasonal_volt  # Adjusted base for voltage
    normal_pitch = seasonal_pitch  # Daily pattern
    normal_roll = seasonal_roll  # Daily pattern

    # Generate anomalies
    anomalies = np.random.rand(num_minutes) < anomaly_probability
    temp_anomaly = np.random.uniform(85, 100, num_minutes)
    volt_anomaly = np.random.uniform(16, 18, num_minutes)
    pitch_anomaly = np.random.uniform(-15, -10, num_minutes)
    roll_anomaly = np.random.uniform(10, 15, num_minutes)

    # Apply anomalies to data
    temp_data = np.where(anomalies, temp_anomaly, normal_temp)
    volt_data = np.where(anomalies, volt_anomaly, normal_volt)
    pitch_data = np.where(anomalies, pitch_anomaly, normal_pitch)
    roll_data = np.where(anomalies, roll_anomaly, normal_roll)

    # Create DataFrame
    data = pd.DataFrame({
        'Time': timestamps,
        'Internal Temperature (F)': temp_data,
        'Volt (V)': volt_data,
        'Pitch (deg)': pitch_data,
        'Roll (deg)': roll_data,
        'Label': anomalies.astype(int)
    })

    return data

# Generate data
start_time = "2023-01-01 00:00:00"
num_minutes = 525600  # Number of minutes in a year
data = generate_time_series_data(start_time, num_minutes)

# Split data into training and testing datasets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['Label'])

# Prepare features and labels
features = ['Internal Temperature (F)', 'Volt (V)', 'Pitch (deg)', 'Roll (deg)']
X_train = train_data[features]
X_test = test_data[features]
y_train = train_data['Label']
y_test = test_data['Label']

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape for LSTM
X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Define LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]), return_sequences=False))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model (will use GPU automatically if available in Colab)
model.fit(X_train_reshaped, y_train, epochs=2, batch_size=64, validation_data=(X_test_reshaped, y_test), verbose=1)

# Validate the model
y_pred_prob = model.predict(X_test_reshaped)
y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary predictions

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Output results
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_rep)

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Prepare test data for visualization
test_data['Predicted_Label'] = y_pred  # Add predicted labels to the test data
test_data['Time'] = pd.to_datetime(test_data['Time'])

# Resample to daily intervals for better visualization
daily_actual = test_data.resample('D', on='Time')['Label'].sum()  # Count the actual anomalies per day
daily_predicted = test_data.resample('D', on='Time')['Predicted_Label'].sum()  # Count predicted anomalies per day

# Normalize by the number of data points per day (if needed)
daily_actual = daily_actual / test_data.resample('D', on='Time').size()
daily_predicted = daily_predicted / test_data.resample('D', on='Time').size()

# Create a figure with subplots
fig = make_subplots(rows=1, cols=1, subplot_titles=["Daily Anomaly Predictions vs Actual"])

# Add actual anomalies trace
fig.add_trace(
    go.Scattergl(
        x=daily_actual.index,
        y=daily_actual.values,
        mode='lines',
        name='Actual Anomalies',
        line=dict(color='red')
    ),
    row=1,
    col=1
)

# Add predicted anomalies trace
fig.add_trace(
    go.Scattergl(
        x=daily_predicted.index,
        y=daily_predicted.values,
        mode='lines',
        name='Predicted Anomalies',
        line=dict(color='orange')
    ),
    row=1,
    col=1
)

# Update layout for better visualization
fig.update_layout(
    title="Daily Anomaly Predictions vs Actual Anomalies",
    xaxis_title="Date",
    yaxis_title="Anomaly Proportion",
    legend=dict(orientation="h", y=-0.2),
    height=600,
    width=1000
)

fig.show()

data.head()

import plotly.graph_objects as go

# Generate new data to test the model
new_data = generate_time_series_data("2024-01-01 00:00:00", num_minutes=1440)  # 1 day of data

# Split data into features and labels
features = ['Internal Temperature (F)', 'Volt (V)', 'Pitch (deg)', 'Roll (deg)']
X_new = new_data[features]
y_new = new_data['Label']

# Standardize the new data using the same scaler
scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X_new)

# Reshape the data for LSTM (add a time-step dimension)
X_new_reshaped = X_new_scaled.reshape((X_new_scaled.shape[0], 1, X_new_scaled.shape[1]))

# Load the trained model (assuming it's saved as a .h5 file)
# model = load_model('path_to_your_trained_model.h5')

# Predict anomalies on new data using the trained model
y_pred_prob = model.predict(X_new_reshaped)
y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary predictions

# Evaluate predictions
accuracy = np.mean(y_pred.flatten() == y_new.values)  # Calculate accuracy
print(f"Accuracy on new data: {accuracy:.4f}")

# Plot actual vs predicted anomalies using Plotly
fig = go.Figure()

# Actual anomalies plot
fig.add_trace(go.Scatter(
    x=new_data['Time'],
    y=y_new,
    mode='lines',
    name='Actual Anomalies',
    line=dict(color='blue', width=2, dash='dot'),
))

# Predicted anomalies plot
fig.add_trace(go.Scatter(
    x=new_data['Time'],
    y=y_pred.flatten(),
    mode='lines',
    name='Predicted Anomalies',
    line=dict(color='red', width=2, dash='solid'),
))

# Update layout
fig.update_layout(
    title='Actual vs Predicted Anomalies',
    xaxis_title='Time',
    yaxis_title='Anomaly (0 or 1)',
    legend_title='Legend',
    hovermode='x unified',
    template='plotly_dark'
)

# Show plot
fig.show()

import plotly.graph_objects as go

# Plot actual vs predicted anomalies on test data
# Assume that we have test data and predictions from your trained model

# Actual anomalies
y_test_pred = (model.predict(X_test_reshaped) > 0.5).astype(int)  # Predicted anomalies on test data

# Create a figure
fig = go.Figure()

# Plot Actual Anomalies
fig.add_trace(go.Scattergl(
    x=test_data['Time'],
    y=y_test,
    mode='lines',
    name='Actual Anomalies',
    line=dict(color='blue', width=2, dash='dot'),
))

# Plot Predicted Anomalies
fig.add_trace(go.Scattergl(
    x=test_data['Time'],
    y=y_test_pred.flatten(),
    mode='lines',
    name='Predicted Anomalies',
    line=dict(color='red', width=2, dash='solid'),
))

# Plot Internal Temperature (F) as an example feature
fig.add_trace(go.Scattergl(
    x=test_data['Time'],
    y=test_data['Internal Temperature (F)'],
    mode='lines',
    name='Internal Temperature (F)',
    line=dict(color='green', width=1)
))

# Optional: Plot other features like Volt, Pitch, and Roll if you want
# Plot Volt (V)
fig.add_trace(go.Scattergl(
    x=test_data['Time'],
    y=test_data['Volt (V)'],
    mode='lines',
    name='Volt (V)',
    line=dict(color='purple', width=1)
))

# Plot Pitch (deg)
fig.add_trace(go.Scattergl(
    x=test_data['Time'],
    y=test_data['Pitch (deg)'],
    mode='lines',
    name='Pitch (deg)',
    line=dict(color='orange', width=1)
))

# Plot Roll (deg)
fig.add_trace(go.Scattergl(
    x=test_data['Time'],
    y=test_data['Roll (deg)'],
    mode='lines',
    name='Roll (deg)',
    line=dict(color='cyan', width=1)
))

# Update layout
fig.update_layout(
    title='Test Data with Actual vs Predicted Anomalies',
    xaxis_title='Time',
    yaxis_title='Value',
    legend_title='Legend',
    hovermode='x unified',
    template='plotly_dark',
)

# Show plot
fig.show()
