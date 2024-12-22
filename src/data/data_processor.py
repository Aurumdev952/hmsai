import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SensorData:
    def __init__(self, sensor_id: str, sensor_type: str, location: Dict[str, float]):
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.location = location
        self.data = pd.DataFrame()
        
    def add_reading(self, timestamp: datetime, value: float, metadata: Optional[Dict] = None):
        """Add a new sensor reading with timestamp"""
        new_data = {
            'timestamp': timestamp,
            'value': value,
            **(metadata or {})
        }
        self.data = pd.concat([self.data, pd.DataFrame([new_data])])
        self.data = self.data.sort_values('timestamp')
        
class DataProcessor:
    def __init__(self, config):
        self.config = config
        
    def process_sensor_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process raw sensor data with comprehensive cleaning and feature extraction"""
        try:
            # Clean data
            data = self._clean_data(data)
            
            # Remove outliers
            data = self._remove_outliers(data)
            
            # Resample to regular intervals
            data = self._resample_data(data)
            
            # Calculate derived features
            data = self._calculate_features(data)
            
            return data
            
        except Exception as e:
            logger.error(f"Error processing sensor data: {str(e)}")
            raise
            
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean raw sensor data"""
        # Convert timestamp to datetime
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Remove duplicates
        data = data.drop_duplicates(subset=['timestamp'])
        
        # Sort by timestamp
        data = data.sort_values('timestamp')
        
        # Handle missing values
        data = data.interpolate(method='time')
        
        return data
        
    def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove statistical outliers using IQR method"""
        Q1 = data['value'].quantile(0.25)
        Q3 = data['value'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Mark outliers
        data['is_outlier'] = (data['value'] < lower_bound) | (data['value'] > upper_bound)
        
        # Interpolate outliers instead of removing them
        outlier_indices = data[data['is_outlier']].index
        data.loc[outlier_indices, 'value'] = np.nan
        data['value'] = data['value'].interpolate(method='time')
        
        return data.drop('is_outlier', axis=1)
        
    def _resample_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Resample data to regular intervals"""
        data = data.set_index('timestamp')
        resampled = data.resample(self.config.PROCESSING_INTERVAL).agg({
            'value': 'mean',
            **{col: 'first' for col in data.columns if col != 'value'}
        })
        return resampled.reset_index()
        
    def _calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived features from sensor data"""
        # Basic statistical features
        data['rolling_mean'] = data['value'].rolling(window=24, min_periods=1).mean()
        data['rolling_std'] = data['value'].rolling(window=24, min_periods=1).std()
        data['rolling_max'] = data['value'].rolling(window=24, min_periods=1).max()
        data['rolling_min'] = data['value'].rolling(window=24, min_periods=1).min()
        
        # Rate of change features
        data['rate_of_change'] = data['value'].diff() / data['value'].shift(1)
        data['acceleration'] = data['rate_of_change'].diff()
        
        # Trend features
        data['trend'] = data['value'].rolling(window=168, min_periods=1).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0]
        )
        
        return data

class BridgeDataManager:
    def __init__(self, config):
        self.config = config
        self.processor = DataProcessor(config)
        self.sensors: Dict[str, SensorData] = {}
        
    def register_sensor(self, sensor_id: str, sensor_type: str, location: Dict[str, float]):
        """Register a new sensor in the system"""
        if sensor_type not in self.config.SENSOR_TYPES:
            raise ValueError(f"Invalid sensor type: {sensor_type}")
        self.sensors[sensor_id] = SensorData(sensor_id, sensor_type, location)
        
    def add_sensor_reading(
        self,
        sensor_id: str,
        timestamp: datetime,
        value: float,
        metadata: Optional[Dict] = None
    ):
        """Add a new reading for a sensor"""
        if sensor_id not in self.sensors:
            raise KeyError(f"Sensor {sensor_id} not registered")
        self.sensors[sensor_id].add_reading(timestamp, value, metadata)
        
    def get_sensor_data(
        self,
        sensor_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """Get processed sensor data for a specific time range"""
        if sensor_id not in self.sensors:
            raise KeyError(f"Sensor {sensor_id} not registered")
            
        sensor = self.sensors[sensor_id]
        mask = (sensor.data['timestamp'] >= start_time) & (sensor.data['timestamp'] <= end_time)
        data = sensor.data[mask].copy()
        
        return self.processor.process_sensor_data(data)
        
    def get_bridge_data(
        self,
        bridge_id: str,
        sensor_ids: List[str],
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, pd.DataFrame]:
        """Get processed data for all sensors on a bridge"""
        bridge_data = {}
        
        for sensor_id in sensor_ids:
            try:
                sensor_data = self.get_sensor_data(sensor_id, start_time, end_time)
                bridge_data[sensor_id] = sensor_data
            except Exception as e:
                logger.error(f"Error getting data for sensor {sensor_id}: {str(e)}")
                continue
                
        return bridge_data