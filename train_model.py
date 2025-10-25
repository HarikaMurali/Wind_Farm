import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np

# ========== Load All Datasets ==========

# 1. GE Turbine Power Curve.csv (Main dataset with all features)
ge_data = pd.read_csv('T1.csv')
print(f"GE Data shape: {ge_data.shape}")

# 2. Turbine_Data.csv (Has many features but mostly empty in your sample)
turbine_data = pd.read_csv('data/Turbine_Data.csv')
print(f"Turbine Data shape: {turbine_data.shape}")

# 3. Power Curve Reference (wind_turbine_data.csv - reference lookup table)
power_curve = pd.read_csv('data/GE Turbine Power Curve.csv')
print(f"Power Curve shape: {power_curve.shape}")

# ========== Data Preparation ==========

# Use GE data as primary source (most complete)
df = ge_data.copy()

# Create failure label: Low power output indicates potential failure
df['failure_status'] = (df['LV ActivePower (kW)'] < 300).astype(int)

# Select features available in GE dataset
feature_cols = [
    'Wind Speed (m/s)', 
    'Theoretical_Power_Curve (KWh)', 
    'Wind Direction (°)'
]

X = df[feature_cols].copy()
y = df['failure_status']

# Handle any missing values
X = X.fillna(X.median())

# Remove any infinite values
X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

print(f"\nFeatures used: {feature_cols}")
print(f"Total samples: {len(X)}")
print(f"Failure cases: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")

# ========== Train Model ==========

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10,
    random_state=42,
    class_weight='balanced'  # Handle imbalanced data
)

model.fit(X_train, y_train)

# Save model
pickle.dump(model, open('turbine_model.pkl', 'wb'))

# Evaluate
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(f"\n✅ Model Training Complete!")
print(f"Training Accuracy: {train_acc*100:.2f}%")
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Model saved as: turbine_model.pkl")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)
