# 📦 Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# 📥 Load dataset
df = pd.read_csv("/Electric_Vehicle_Population_Size_History_By_County_.csv")

# 🧹 Data Preprocessing
df.fillna(method='ffill', inplace=True)

# Convert columns to numeric, coercing errors to NaN
for col in ['Battery Electric Vehicles (BEVs)',
            'Plug-In Hybrid Electric Vehicles (PHEVs)',
            'Electric Vehicle (EV) Total']:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False), errors='coerce').astype(int)


# Convert percentage to float
df['Percent Electric Vehicles'] = df['Percent Electric Vehicles'].astype(str).str.replace('%', '', regex=False).astype(float)

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# 📆 Feature Engineering
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter

# Encode categorical feature
df['Vehicle Primary Use'] = df['Vehicle Primary Use'].map({'Passenger': 0, 'Truck': 1})

# 🎯 Features and Target
X = df[['Year', 'Month', 'Quarter', 'Vehicle Primary Use']]
y = df['Electric Vehicle (EV) Total']

# 🔀 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🤖 Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 📈 Predictions and Evaluation
y_pred = model.predict(X_test)
print("\n📊 Model Evaluation:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

# 💾 Save the model
joblib.dump(model, 'ev_adoption_model.pkl')

# 🔮 Forecasting for Jan-Jun 2025 (Passenger Vehicles)
forecast_data = pd.DataFrame({
    'Year': [2025]*6,
    'Month': list(range(1, 7)),
    'Quarter': [1, 1, 1, 2, 2, 2],
    'Vehicle Primary Use': [0]*6  # Passenger
})

forecast = model.predict(forecast_data)
forecast_data['Predicted EV Total'] = forecast.astype(int)

print("\n🔮 Forecast for Jan-Jun 2025:")
print(forecast_data)

# 📊 Optional: Visualization
plt.figure(figsize=(8, 5))
sns.barplot(x=forecast_data['Month'], y=forecast_data['Predicted EV Total'], palette='viridis')
plt.title("Predicted EV Adoption (Jan-Jun 2025)")
plt.xlabel("Month")
plt.ylabel("Predicted EV Total")
plt.grid(True)
plt.show()