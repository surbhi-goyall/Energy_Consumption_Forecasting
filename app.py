# import os
# import pickle
# import numpy as np
# import pandas as pd
# from flask import Flask, render_template, request, jsonify
# from tensorflow.keras.models import load_model
# from sklearn.preprocessing import MinMaxScaler

# app = Flask(__name__)

# # Load the pre-trained transformer model
# model = load_model('saved_models/transformer_base.h5')

# # Load the scaler used for normalizing the input data
# with open('saved_models/scaler.pkl', 'rb') as f:
#     scaler = pickle.load(f)

# # Utility function to handle data preprocessing
# def preprocess_data(file_path):
#     # Load the data
#     df = pd.read_csv(file_path)
    
#     # Scale the relevant columns using the previously saved scaler
#     # Assuming the relevant columns are 'column1', 'column2', etc. Adjust as needed.
#     scaled_data = scaler.transform(df[['Global_active_power']])  # Update with actual feature columns
    
#     # Reshape the data to match the expected input for the transformer model
#     # Assuming sequence length of 24 (you can adjust as necessary)
#     X = scaled_data[-24:].reshape(1, 24, scaled_data.shape[1])  # 1 sample, 24 time steps, number of features
#     return X, df

# # Route to upload CSV and predict
# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         # Check if the post request has the file part
#         if 'file' not in request.files:
#             return 'No file part', 400
#         file = request.files['file']
        
#         if file.filename == '':
#             return 'No selected file', 400
        
#         if file:
#             # Save the uploaded file temporarily
#             file_path = os.path.join('uploads', file.filename)
#             file.save(file_path)

#             # Preprocess the uploaded data
#             X, df = preprocess_data(file_path)

#             # Make predictions using the transformer model
#             predictions = model.predict(X)
            
#             # Invert the scaling on predictions (optional)
#             predictions_inversed = scaler.inverse_transform(predictions)

#             # Prepare output (you can modify this to suit your requirements)
#             result = {
#                 'predictions': predictions_inversed.tolist(),
#                 'original_data': df.to_dict(orient='records')  # Display original data
#             }

#             # Return results as a JSON response
#             return jsonify(result)
    
#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)


# import os
# import pandas as pd
# import numpy as np
# from flask import Flask, request, render_template
# import matplotlib.pyplot as plt
# from io import BytesIO
# import base64
# from model_utils import predict_energy, detect_peaks, detect_anomalies, generate_tips  # Assumed you have these

# app = Flask(__name__)

# UPLOAD_FOLDER = 'uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         file = request.files['file']
#         if file.filename == '':
#             return render_template('index.html', error="No file selected")

#         filepath = os.path.join(UPLOAD_FOLDER, file.filename)
#         file.save(filepath)

#         # Load and preprocess data
#         df = pd.read_csv(filepath, parse_dates=['datetime'])
#         df.sort_values('datetime', inplace=True)
#         df.set_index('datetime', inplace=True)

#         # Predict energy consumption
#         predictions = predict_energy(df)

#         # Detect peak hours
#         peak_hours = detect_peaks(df)

#         # Detect anomalies
#         anomalies = detect_anomalies(df)

#         # Generate energy-saving tips
#         tips = generate_tips(df)

#         # Create plot
#         img = BytesIO()
#         plt.figure(figsize=(10,5))
#         plt.plot(df.index, df['energy'], label='Actual')
#         plt.plot(df.index[-len(predictions):], predictions, label='Predicted')
#         plt.xlabel('Time')
#         plt.ylabel('Energy Consumption (kWh)')
#         plt.title('Energy Forecast')
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(img, format='png')
#         img.seek(0)
#         plot_url = base64.b64encode(img.getvalue()).decode()

#         return render_template(
#             'results.html', 
#             table=df.tail(10).to_html(classes='table table-bordered', border=0),
#             plot_url=plot_url,
#             peak_hours=peak_hours,
#             anomalies=anomalies,
#             tips=tips
#         )

#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the pre-trained model and scaler
model = load_model('saved_models/transformer_base.h5')
with open('saved_models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# --- Analysis and Prediction Functions ---

def predict_energy(df):
    df = df.copy()
    
    # Convert 'datetime' column to datetime and set it as index
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    
    # Scale the 'Global_active_power' column
    usage_values = df[['Global_active_power']].values
    scaler.fit(usage_values)
    scaled_usage = scaler.transform(usage_values)
    
    # Ensure there are enough data points (at least 24)
    if len(scaled_usage) < 24:
        raise ValueError("Need at least 24 data points for prediction")
    
    # Use the last 24 hours of data for initial prediction
    input_seq = scaled_usage[-24:].reshape(1, 24, 1)  # Transformer model expects (samples, timesteps, features)
    
    # Initialize an empty list to store predictions
    predictions = []
    
    # Predict the next 24 time steps iteratively
    for _ in range(24):
        # Make a prediction
        pred_scaled = model.predict(input_seq)
        
        # Inverse scale the prediction
        pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        
        # Append the prediction to the results
        predictions.append(pred[0])
        
        # Update the input sequence by appending the prediction
        # This ensures the next prediction is based on the new data
        input_seq = np.roll(input_seq, -1, axis=1)  # Shift data to remove the oldest timestep
        input_seq[0, -1, 0] = pred_scaled  # Append the predicted value
    
    # Generate future times (24 hours ahead)
    last_time = df.index[-1]
    future_times = pd.date_range(start=last_time + pd.Timedelta(hours=1), periods=24, freq='H')
    
    # Return predictions as a DataFrame
    return pd.DataFrame({'datetime': future_times, 'predicted_energy': predictions})



def detect_peaks(df, threshold=0.9):
    threshold_value = df['predicted_energy'].quantile(threshold)
    df['is_peak'] = df['predicted_energy'] >= threshold_value
    return df

def detect_anomalies(df):
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

# --- Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file uploaded', 400
        file = request.files['file']
        if file.filename == '':
            return 'No file selected', 400

        filepath = os.path.join('uploads', file.filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(filepath)

        df = pd.read_csv(filepath)
        try:
            pred_df = predict_energy(df)
            pred_df = detect_peaks(pred_df)
            pred_df = detect_anomalies(pred_df)
            tips = generate_tips(pred_df)

            return render_template('results.html', tables=pred_df.to_html(classes='table table-bordered', index=False), tips=tips)
        except Exception as e:
            return f"Error during processing: {str(e)}", 500

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
