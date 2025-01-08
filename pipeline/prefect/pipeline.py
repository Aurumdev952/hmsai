from prefect import task, flow
from datetime import datetime, timedelta
import boto3
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from statsmodels.tsa.seasonal import STL
from rainflow import count_cycles
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
import mlflow
import mlflow.keras
import requests
from fetch_data import get_week_data
from data_processing import process_data_and_feature_enginnering
from mlflow.models import infer_signature
import os
# S3 Configuration
S3_BUCKET = 'pipeline'
S3_FEATURES_KEY = 'features/latest_features.csv'
S3_MODEL_KEY = 'models/latest_model.h5'
AWS_REGION = 'your-region'
S3_ENDPOINT_URL = 'http://host.docker.internal:9000'
S3_ACCESS_KEY = 'minioadmin'
S3_SECRET_KEY = 'minioadmin'
EPOCHS = 10
BATCH_SIZE = 256
TEST_SIZE = 0.2
VALIDATION_SPLIT = 0.1
THRESHOLD_FACTOR = 1.5
VALIDATION_THRESHOLD = 0.2

def get_s3_client():
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name='us-east-1'
    )

# MLflow Configuration
MLFLOW_TRACKING_URI = 'http://host.docker.internal:5000'
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment('Model Training Pipeline')

@task
def fetch_data_from_api() -> dict:
    """Fetch data from API and save to temporary files."""
    s3 = get_s3_client()
    tmp_dir = "./tmp"  # Replace with your desired path if needed
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=S3_MODEL_KEY)
    except:
        print("Model does not exist")
        s3.upload_file('./latest_model.h5', S3_BUCKET, S3_MODEL_KEY)
    current_date = datetime.now()
    start_date = current_date - timedelta(days=current_date.weekday())
    end_date = start_date + timedelta(days=6)
    start_date = start_date.strftime("%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")

    fetched_data = get_week_data(start_date, end_date)
    saved_files = {}
    for quantity_name, data in fetched_data.items():
        filepath = f"./tmp/aggregated_data_{quantity_name}.csv"
        data.to_csv(filepath)
        saved_files[quantity_name] = filepath
        print(f"Data for {quantity_name} saved to '{filepath}'")
    return saved_files

@task
def preprocess_and_feature_engineer(data_files: dict) -> str:
    """Process data and engineer features, save to S3."""
    features = process_data_and_feature_enginnering()
    s3 = get_s3_client()
    features_path = './tmp/features.csv'
    features.to_csv(features_path)
    s3.upload_file(features_path, S3_BUCKET, S3_FEATURES_KEY)
    print(f"Features saved to MinIO: s3://{S3_BUCKET}/{S3_FEATURES_KEY}")
    return features_path

@task
def train_model(features_path: str) -> str:
    """Train the model using the engineered features."""
    try:
        s3 = get_s3_client()
        s3.download_file(S3_BUCKET, S3_FEATURES_KEY, features_path)
        data = pd.read_csv(features_path, index_col=0)

        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        X_train, X_test = train_test_split(data_scaled, test_size=TEST_SIZE, random_state=42)

        model_path = r'./tmp/model.h5'
        try:
            s3.download_file(S3_BUCKET, S3_MODEL_KEY, model_path)
            model = load_model(model_path, compile=False)
            model.compile(optimizer='adam', loss="mse")
            print("Loaded previous model weights.")
        except Exception as e:
            print("error loading model", e)
            print("No previous model found. Training from scratch.")
            input_dim = X_train.shape[1]
            model = Sequential([
                Dense(64, activation='relu', input_dim=input_dim),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(64, activation='relu'),
                Dense(input_dim, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='mse')

        with mlflow.start_run():
            mlflow.keras.autolog()

            history = model.fit(
                X_train, X_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_split=VALIDATION_SPLIT,
                shuffle=True,
            )

            val_loss = history.history['val_loss'][-1]
            mlflow.log_metric("final_validation_loss", val_loss)

            reconstructions = model.predict(X_test)
            signature = infer_signature(X_test, reconstructions)
            reconstruction_errors = np.mean(np.square(X_test - reconstructions), axis=1)
            threshold = np.mean(reconstruction_errors) + THRESHOLD_FACTOR * np.std(reconstruction_errors)

            anomalies = reconstruction_errors > threshold
            mlflow.log_metric("anomaly_threshold", threshold)
            mlflow.log_metric("num_anomalies", np.sum(anomalies))

            if val_loss < VALIDATION_THRESHOLD:
                print("Validation passed. Saving model.")
                # model.save(r'D:/aurum/contrib/bridge-health/pipeline/prefect/tmp/model.h5')
                model.save(model_path)
                s3.upload_file(model_path, S3_BUCKET, S3_MODEL_KEY)
                mlflow.keras.log_model(
                    model,
                    artifact_path='model',
                    registered_model_name='DeployedModel',
                    signature=signature
                )
                print(f"New model weights saved to S3: s3://{S3_BUCKET}/{S3_MODEL_KEY}")
                print(f"Model deployed successfully")
            else:
                print(f"Validation failed. Model not saved. Loss: {val_loss}")

        return model_path

    except UnicodeEncodeError:
        pass
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise e


@flow(name="automated_model_pipeline")
def model_training_flow():
    """Main flow for the automated model pipeline."""
    data_files = fetch_data_from_api()
    features_path = preprocess_and_feature_engineer(data_files)
    train_model(features_path)

if __name__ == "__main__":
    # For local development and testing
    model_training_flow()

