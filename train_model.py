import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import pickle

# Load data
df = pd.read_csv("city_day.csv")

# Drop rows where AQI (our target) is missing
df = df.dropna(subset=["AQI"])

# Select pollutant features
features = ["PM2.5", "PM10", "NO2", "NH3", "SO2", "CO", "O3"]
X = df[features]
y = df["AQI"]

# Fill missing pollutant values with column mean
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42
)

# Train Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"MAE:  {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R²:   {r2_score(y_test, y_pred):.4f}")

# Save model and imputer
with open("model.pkl", "wb") as f:
    pickle.dump({"model": model, "imputer": imputer, "features": features}, f)

print("Model saved to model.pkl")