import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Ensure directories are created
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

# Load the dataset
data_path = 'data/weather_data.csv'
data = pd.read_csv(data_path)

# --- EDA Part ---

# 1. Summary Statistics
print("\n--- Summary Statistics ---\n")
print(data.describe())

# 2. Visualizations
# Scatter plot: Temperature vs. Humidity
plt.figure(figsize=(8, 6))
sns.scatterplot(x='humidity', y='temp', data=data)
plt.title('Temperature vs Humidity')
plt.xlabel('Humidity (%)')
plt.ylabel('Temperature (°C)')
plt.savefig('visualizations/temp_vs_humidity.png')
plt.show()

# Scatter plot: Temperature vs Wind Speed
plt.figure(figsize=(8, 6))
sns.scatterplot(x='wind_speed', y='temp', data=data)
plt.title('Temperature vs Wind Speed')
plt.xlabel('Wind Speed (km/h)')
plt.ylabel('Temperature (°C)')
plt.savefig('visualizations/temp_vs_wind_speed.png')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.savefig('visualizations/correlation_heatmap.png')
plt.show()

# --- Machine Learning Part ---

# Prepare the data
X = data[['day']].values  # Input (day)
y = data['temp'].values  # Output (temperature)

# Build a simple linear regression model
model = Sequential()
model.add(Dense(1, input_dim=1))  # Single input (day) and single output (temp)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=100, verbose=0)

# Save the trained model
model.save('models/weather_temp_predictor.h5')

# Make predictions
predictions = model.predict(X)

# Visualize the predictions
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, predictions, color='red', label='Predicted Temperatures')
plt.title('Weather Temperature Prediction')
plt.xlabel('Day')
plt.ylabel('Temperature (°C)')
plt.legend()

# Save the plot
plt.savefig('visualizations/temperature_predictions.png')
plt.show()

# Save predictions to CSV for Power BI
predictions_df = pd.DataFrame({
    'day': X.flatten(),
    'actual_temp': y,
    'predicted_temp': predictions.flatten()
})
predictions_df.to_csv('data/predicted_temperatures.csv', index=False)
