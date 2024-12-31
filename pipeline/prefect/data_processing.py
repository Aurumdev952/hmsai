import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import skew, kurtosis
from rainflow import count_cycles
from statsmodels.tsa.seasonal import STL

RESAMPLE_INTERVAL = '5min'  # 5-minute intervals

# Function to load and process a single file
def process_file(file_path):
    df = pd.read_csv(file_path)
    value_column = [col for col in df.columns if col.lower() != 'time'][0]
    df['Time'] = pd.to_datetime(df['Time'])
    df.set_index('Time', inplace=True)
    df = df.resample(RESAMPLE_INTERVAL).mean()
    z_scores = (df[value_column] - df[value_column].mean()) / df[value_column].std()
    df = df[np.abs(z_scores) < 3]

    def butter_lowpass_filter(data, cutoff=0.1, fs=1.0, order=2):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data)

    df[value_column] = butter_lowpass_filter(df[value_column].fillna(0), cutoff=0.1, fs=1.0)
    return df

# Generate basic statistical features
def generate_statistical_features(df):
    features = pd.DataFrame(index=df.index)
    features['mean'] = df.mean(axis=1)
    features['std'] = df.std(axis=1)
    features['skew'] = df.apply(skew, axis=1)
    features['kurtosis'] = df.apply(kurtosis, axis=1)
    return features

# Generate dynamic rolling features
def generate_dynamic_features(df, window_size=12):
    features = pd.DataFrame(index=df.index)
    features['rolling_mean'] = df.rolling(window=window_size).mean().mean(axis=1)
    features['rolling_std'] = df.rolling(window=window_size).std().mean(axis=1)
    features['rolling_min'] = df.rolling(window=window_size).min().mean(axis=1)
    features['rolling_max'] = df.rolling(window=window_size).max().mean(axis=1)
    return features

# Compute correlation features
def generate_correlation_features(df):
    features = pd.DataFrame(index=df.index)
    corr_matrix = df.corr()
    for i, col1 in enumerate(df.columns):
        for j, col2 in enumerate(df.columns):
            if i < j:
                features[f'{col1}_vs_{col2}_corr'] = df[col1].rolling(12).corr(df[col2])
    return features

# Apply Rainflow Counting
def generate_rainflow_features(df):
    features = pd.DataFrame(index=df.index)
    for col in df.columns:
        cycles = count_cycles(df[col].dropna().values)
        features[f'{col}_rainflow_mean'] = np.mean([cycle[0] for cycle in cycles])
        features[f'{col}_rainflow_count'] = len(cycles)
    return features

# Apply STL Decomposition
def generate_stl_features(df):
    features = pd.DataFrame(index=df.index)
    # Calculate the period based on resampling interval
    period = int(24 * 60 / pd.Timedelta(RESAMPLE_INTERVAL).seconds / 60) # Assuming daily cycle

    # Ensure period is at least 2
    period = max(2, period)

    for col in df.columns:
        # Pass the calculated period to the STL function
        stl = STL(df[col].dropna(), period=period, seasonal=13)
        result = stl.fit()
        features[f'{col}_trend'] = result.trend
        features[f'{col}_seasonal'] = result.seasonal
        features[f'{col}_residual'] = result.resid
    return features
def process_data_and_feature_enginnering():
    # Configuration
    SENSOR_FILES = ['./tmp/aggregated_data_2DHRT_Pitch.csv', './tmp/aggregated_data_2DHRT_Roll.csv', './tmp/aggregated_data_Displacement.csv', './tmp/aggregated_data_Strain-xx-high_rate.csv']
    # Main processing loop
    processed_dfs = []
    for file in SENSOR_FILES:
        processed_df = process_file(file)
        processed_dfs.append(processed_df)

    merged_data = pd.concat(processed_dfs, axis=1)
    merged_data.columns = [f"Sensor_{i+1}" for i in range(len(processed_dfs))]
    merged_data.dropna(inplace=True)

    # Generate features
    statistical_features = generate_statistical_features(merged_data)
    dynamic_features = generate_dynamic_features(merged_data)
    correlation_features = generate_correlation_features(merged_data)
    rainflow_features = generate_rainflow_features(merged_data)
    stl_features = generate_stl_features(merged_data)

    # Combine all features
    all_features = pd.concat([statistical_features, dynamic_features, correlation_features, rainflow_features, stl_features], axis=1)
    all_features.dropna(inplace=True)
    return all_features
    print(f"Feature engineering complete")
