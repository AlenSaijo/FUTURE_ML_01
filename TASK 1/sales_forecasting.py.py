import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os  # Added missing import
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# --- STEP 2: DATA LOADING & CLEANING ---
file_path = r'C:\Users\alens\Desktop\TASK 1\train.csv\train.csv.csv'

# Check if file exists and isn't a folder before trying to open it
if not os.path.exists(file_path):
    print(f"❌ Error: The file was not found at {file_path}")
    sys.exit()
elif os.path.isdir(file_path):
    print(f"❌ Error: '{file_path}' is a folder, not a file!")
    sys.exit()

try:
    # Attempt to load the data
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    print("🚀 Data loaded successfully!")

    # Processing the data
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='mixed')
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    df['DayOfWeek'] = df['Order Date'].dt.dayofweek

    # Group by Date to get total daily sales
    daily_sales = df.groupby('Order Date').agg({'Sales': 'sum'}).reset_index()
    daily_sales['Month'] = daily_sales['Order Date'].dt.month
    daily_sales['Year'] = daily_sales['Order Date'].dt.year
    daily_sales['DayOfWeek'] = daily_sales['Order Date'].dt.dayofweek

except PermissionError:
    print("❌ PERMISSION DENIED: Close 'train.csv' in Excel and try again!")
    sys.exit()
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")
    sys.exit()

# --- STEP 3: FORECASTING MODEL ---
# Define features (X) and target (y)
X = daily_sales[['Year', 'Month', 'DayOfWeek']]
y = daily_sales['Sales']

# Split data: Use the last 20% of time for testing
split_index = int(len(daily_sales) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and show error
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"✅ Model Evaluation -> Mean Absolute Error: ${mae:.2f}")

# --- STEP 4: VISUALIZATION ---
plt.figure(figsize=(12, 6))
plt.plot(daily_sales['Order Date'][split_index:], y_test, label='Actual Sales', color='blue')
plt.plot(daily_sales['Order Date'][split_index:], predictions, label='Forecasted Sales', color='red', linestyle='--')
plt.title('Business Forecast: Actual vs Predicted Sales')
plt.xlabel('Date')
plt.ylabel('Sales ($)')
plt.legend()
plt.grid(True)
plt.show()