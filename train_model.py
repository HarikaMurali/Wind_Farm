import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load your dataset
df = pd.read_csv('data/wind_turbine_data.csv')

# Create a simple failure_status label
df['failure_status'] = (df['LV ActivePower (kW)'] < 300).astype(int)

# Use relevant features for prediction
X = df[['Wind Speed (m/s)', 'Theoretical_Power_Curve (KWh)', 'Wind Direction (°)']]
y = df['failure_status']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open('turbine_model.pkl', 'wb'))

print(f"Model Accuracy: {model.score(X_test, y_test)*100:.2f}%")
