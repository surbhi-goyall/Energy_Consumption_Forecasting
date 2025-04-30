import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load your dataset (replace 'your_dataset.csv' with the correct file path)
df = pd.read_csv('dataset.csv')  # Change this to your actual dataset

# Initialize the MinMaxScaler (you can choose another scaler if needed)
scaler = MinMaxScaler()

# Assuming 'column1', 'column2', 'column3' are the columns you want to scale.
# Adjust these column names to match the ones in your dataset.
scaled_data = scaler.fit_transform(df[['Global_active_power']])  # Update column names here

# Save the scaler to a file (saved_models/scaler.pkl)
with open('saved_models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Scaler saved successfully!")
