import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("ev_battery_health.csv")

# Encode UsagePattern
le = LabelEncoder()
df['UsagePattern'] = le.fit_transform(df['UsagePattern'])  # Urban=2, Highway=0, Mixed=1

# Train-test split
X = df[['ChargeCycles', 'AvgTemp', 'AvgVoltage', 'UsagePattern']]
y = df['BatteryHealth']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45602)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("🔋 Battery Health Predictor Results")
print("-----------------------------------")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# 🚀 Take user input from terminal
try:
    charge_cycles = int(input("\nEnter charge cycles: "))
    avg_temp = float(input("Enter average battery temperature (°C): "))
    avg_voltage = float(input("Enter average voltage (V): "))
    usage_pattern = input("Enter usage pattern (Urban / Highway / Mixed): ").strip().capitalize()

    # Encode usage pattern
    if usage_pattern not in ['Urban', 'Highway', 'Mixed']:
        raise ValueError("Invalid usage pattern. Choose Urban, Highway, or Mixed.")
    encoded_usage = le.transform([usage_pattern])[0]

    # Predict
    input_data = [[charge_cycles, avg_temp, avg_voltage, encoded_usage]]
    predicted_health = model.predict(input_data)
    print(f"\n✅ Predicted Battery Health: {predicted_health[0]:.2f}%")

except Exception as e:
    print(f" Error: {e}")
