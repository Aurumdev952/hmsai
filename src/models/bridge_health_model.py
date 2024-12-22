import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import pandas as pd

class FeatureExtractor:
    def __init__(self, config):
        self.config = config
        
    def extract_features(self, sensor_data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """Extract features from multiple sensor data sources"""
        features = []
        
        for sensor_id, data in sensor_data.items():
            sensor_features = self._extract_sensor_features(data)
            features.append(sensor_features)
            
        return np.concatenate(features, axis=1)
        
    def _extract_sensor_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features from a single sensor's data"""
        # Basic statistical features
        features = [
            data['value'].mean(),
            data['value'].std(),
            data['value'].min(),
            data['value'].max(),
            data['rate_of_change'].mean(),
            data['rolling_std'].mean()
        ]
        
        return np.array(features).reshape(1, -1)

class BridgeHealthModel:
    def __init__(self, config):
        self.config = config
        self.feature_extractor = FeatureExtractor(config)
        self.model = self._build_model()
        
    def _build_model(self) -> tf.keras.Model:
        """Build the neural network model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(18,)),  # 6 features * 3 sensors
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Output: Bridge Health Index (0-1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def prepare_dataset(
        self,
        sensor_data: Dict[str, pd.DataFrame],
        labels: Optional[np.ndarray] = None
    ) -> tf.data.Dataset:
        """Prepare data for training or prediction"""
        features = self.feature_extractor.extract_features(sensor_data)
        
        if labels is not None:
            return tf.data.Dataset.from_tensor_slices((features, labels))\
                   .batch(self.config.BATCH_SIZE)
        
        return tf.data.Dataset.from_tensor_slices(features)\
               .batch(self.config.BATCH_SIZE)
        
    def train(
        self,
        train_data: Dict[str, pd.DataFrame],
        train_labels: np.ndarray,
        validation_data: Optional[Tuple[Dict[str, pd.DataFrame], np.ndarray]] = None,
        epochs: int = 10
    ):
        """Train the model"""
        train_dataset = self.prepare_dataset(train_data, train_labels)
        
        validation_dataset = None
        if validation_data:
            validation_dataset = self.prepare_dataset(validation_data[0], validation_data[1])
        
        history = self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=validation_dataset,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=3),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=f"{self.config.MODEL_PATH}/best_model.h5",
                    save_best_only=True
                )
            ]
        )
        
        return history
        
    def predict(self, sensor_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Make predictions about bridge health"""
        dataset = self.prepare_dataset(sensor_data)
        predictions = self.model.predict(dataset)
        
        return {
            'health_index': float(predictions[0][0]),
            'timestamp': datetime.now().isoformat(),
            'confidence': 0.95  # This could be calculated based on model uncertainty
        }

class BridgeHealthPredictor:
    def __init__(self, config):
        self.config = config
        self.model = BridgeHealthModel(config)
        
    async def predict_bridge_health(
        self,
        bridge_id: str,
        sensor_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Predict bridge health and generate recommendations"""
        # Make health prediction
        prediction = self.model.predict(sensor_data)
        
        # Generate recommendations based on health index
        recommendations = self._generate_recommendations(prediction['health_index'])
        
        return {
            'bridge_id': bridge_id,
            'health_prediction': prediction,
            'recommendations': recommendations,
            'next_inspection_date': self._calculate_next_inspection(prediction['health_index'])
        }
        
    def _generate_recommendations(self, health_index: float) -> List[str]:
        """Generate maintenance recommendations based on health index"""
        recommendations = []
        
        if health_index < 0.3:
            recommendations.extend([
                "Immediate inspection required",
                "Consider bridge closure for safety",
                "Schedule emergency maintenance"
            ])
        elif health_index < 0.6:
            recommendations.extend([
                "Schedule inspection within 2 weeks",
                "Monitor sensor data daily",
                "Plan maintenance activities"
            ])
        else:
            recommendations.extend([
                "Continue regular monitoring",
                "Schedule next routine inspection",
                "Update maintenance log"
            ])
            
        return recommendations
        
    def _calculate_next_inspection(self, health_index: float) -> str:
        """Calculate next inspection date based on health index"""
        if health_index < 0.3:
            next_date = datetime.now() + timedelta(days=1)
        elif health_index < 0.6:
            next_date = datetime.now() + timedelta(days=14)
        else:
            next_date = datetime.now() + timedelta(days=90)
            
        return next_date.isoformat()

# Usage example:
def train_model():
    config = Config()  # Your config class
    predictor = BridgeHealthPredictor(config)
    
    # Load training data (placeholder)
    train_data = {
        'sensor1': pd.DataFrame(...),
        'sensor2': pd.DataFrame(...),
        'sensor3': pd.DataFrame(...)
    }
    train_labels = np.array([...])  # Historical health indices
    
    # Train the model
    predictor.model.train(train_data, train_labels, epochs=20)
    
    return predictor

async def predict_health(predictor: BridgeHealthPredictor, bridge_id: str, sensor_data: Dict[str, pd.DataFrame]):
    prediction = await predictor.predict_bridge_health(bridge_id, sensor_data)
    return prediction