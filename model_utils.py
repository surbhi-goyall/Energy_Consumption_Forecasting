import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

# Load your Keras model
model = load_model("saved_models/transformer_base.h5")  # Update with your actual path if different

# You might need to fit a new scaler or load a saved one
scaler = MinMaxScaler()

def predict_energy(df):
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    
    # Fit scaler to the data
    usage_values = df[['energy_usage']].values
    scaler.fit(usage_values)
    scaled_usage = scaler.transform(usage_values)
    
    # Assume the model expects the last 24 hours of data
    if len(scaled_usage) < 24:
        raise ValueError("Need at least 24 data points for prediction")
    
    input_seq = scaled_usage[-24:].reshape(1, 24, 1)
    
    preds_scaled = model.predict(input_seq)
    preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

    # Generate timestamps for next 24 hours (assuming hourly data)
    last_time = df.index[-1]
    future_times = pd.date_range(start=last_time + pd.Timedelta(hours=1), periods=24, freq='H')
    
    return pd.DataFrame({'datetime': future_times, 'predicted_energy': preds})


def detect_peaks(df, threshold=0.9):
    # Flag top 10% as peaks
    threshold_value = df['predicted_energy'].quantile(threshold)
    df['is_peak'] = df['predicted_energy'] >= threshold_value
    return df


def detect_anomalies(df):
    # Simple method: points far from rolling mean
    df['rolling_mean'] = df['predicted_energy'].rolling(window=3, min_periods=1).mean()
    df['anomaly'] = abs(df['predicted_energy'] - df['rolling_mean']) > df['rolling_mean'] * 0.25
    return df


def generate_tips(df):
    avg_usage = df['predicted_energy'].mean()
    peak_usage = df[df['is_peak']]['predicted_energy'].mean() if df['is_peak'].any() else 0

    tips = []

    if peak_usage > avg_usage * 1.3:
        tips.append("Shift energy usage to non-peak hours to save on cost.")

    if avg_usage > 1.5:
        tips.append("Consider turning off unused devices or using energy-efficient appliances.")

    if not tips:
        tips.append("Great job! Your energy usage is within efficient range.")

    return tips
