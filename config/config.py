class Config:
    # API configurations
    API_VERSION = "v1"
    API_PREFIX = f"/api/{API_VERSION}"
    API_BASE_URL = "http://localhost:8000"
    
    # Database configurations
    DB_HOST = "localhost"
    DB_PORT = 27017
    DB_NAME = "bridge_monitoring"
    
    # Model configurations
    MODEL_PATH = "models/"
    BATCH_SIZE = 32
    TRAIN_SPLIT = 0.8
    
    # Sensor configurations
    SENSOR_TYPES = [
        "inclinometer",
        "accelerometer",
        "displacement",
        "strain",
        "temperature",
        "moisture",
        "load_cell",
        "corrosion",
        "weather"
    ]
    
    # Processing configurations
    PROCESSING_INTERVAL = "1D"  # Daily processing
    DATA_RETENTION_DAYS = 30    # Keep processed data for 30 days
