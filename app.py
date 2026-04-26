# import streamlit as st
# import pandas as pd
# import pickle
# import os

# # Auto-train if model doesn't exist (for cloud deployment)
# if not os.path.exists("model.pkl"):
#     import subprocess
#     subprocess.run(["python", "train_model.py"])

# st.set_page_config(page_title="AQI Predictor", page_icon="🌿")
# st.title("🌿 Air Quality Index Predictor")
# st.markdown("Adjust the pollutant levels to predict the AQI and category.")

# # Load model
# if not os.path.exists("model.pkl"):
#     st.error("model.pkl not found! Run `python train_model.py` first.")
#     st.stop()

# try:
#     with open("model.pkl", "rb") as f:
#         saved = pickle.load(f)
#     model = saved["model"]
#     imputer = saved["imputer"]
#     features = saved["features"]
# except Exception as e:
#     st.error(f"Failed to load model: {e}")
#     st.stop()

# # AQI category helper
# def get_aqi_category(aqi):
#     if aqi <= 50:   return "Good", "🟢"
#     elif aqi <= 100: return "Satisfactory", "🟡"
#     elif aqi <= 200: return "Moderate", "🟠"
#     elif aqi <= 300: return "Poor", "🔴"
#     elif aqi <= 400: return "Very Poor", "🟣"
#     else:            return "Severe", "⚫"

# # --- Input sliders ---
# st.subheader("Pollutant levels (µg/m³)")

# col1, col2 = st.columns(2)

# with col1:
#     pm25 = st.slider("PM2.5", 0.0, 500.0, 60.0, step=1.0)
#     pm10 = st.slider("PM10",  0.0, 500.0, 90.0, step=1.0)
#     no2  = st.slider("NO2",   0.0, 200.0, 40.0, step=1.0)
#     nh3  = st.slider("NH3",   0.0, 100.0, 15.0, step=0.5)

# with col2:
#     so2  = st.slider("SO2",   0.0, 100.0, 20.0, step=0.5)
#     co   = st.slider("CO",    0.0, 50.0,   1.5, step=0.1)
#     o3   = st.slider("O3",    0.0, 200.0,  30.0, step=1.0)

# # --- Predict ---
# if st.button("Predict AQI"):
#     input_df = pd.DataFrame(
#         [[pm25, pm10, no2, nh3, so2, co, o3]],
#         columns=features
#     )
#     input_imputed = imputer.transform(input_df)
#     aqi_pred = model.predict(input_imputed)[0]
#     aqi_pred = round(aqi_pred)

#     category, emoji = get_aqi_category(aqi_pred)

#     st.divider()
#     st.subheader("Prediction")

#     col_a, col_b = st.columns(2)
#     with col_a:
#         st.metric("Predicted AQI", aqi_pred)
#     with col_b:
#         st.metric("Category", f"{emoji} {category}")

#     # Color-coded progress bar (AQI out of 500)
#     st.progress(
#         min(aqi_pred / 500, 1.0),
#         text=f"AQI {aqi_pred} / 500"
#     )

#     # Advice
#     advice = {
#         "Good":         "Air quality is great. Enjoy outdoor activities!",
#         "Satisfactory": "Air quality is acceptable. Sensitive individuals should be cautious.",
#         "Moderate":     "Sensitive groups may experience symptoms. Limit prolonged outdoor exposure.",
#         "Poor":         "Everyone may begin to feel health effects. Wear a mask outdoors.",
#         "Very Poor":    "Health alert! Avoid outdoor activities. Use air purifiers indoors.",
#         "Severe":       "Emergency conditions. Stay indoors, keep windows shut."
#     }
#     st.info(advice[category])

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import pickle
import urllib.request
import os

# Download dataset automatically if not present
DATA_URL = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/city_day.csv"
DATA_PATH = "city_day.csv"

if not os.path.exists(DATA_PATH):
    print("Downloading dataset...")
    try:
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
        print("Download complete.")
    except Exception as e:
        print(f"Download failed: {e}")
        raise

# Load data
df = pd.read_csv(DATA_PATH)

# Drop rows where AQI is missing
df = df.dropna(subset=["AQI"])

# Features and target
features = ["PM2.5", "PM10", "NO2", "NH3", "SO2", "CO", "O3"]
X = df[features]
y = df["AQI"]

# Fill missing values
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R²:  {r2_score(y_test, y_pred):.4f}")

# Save
with open("model.pkl", "wb") as f:
    pickle.dump({"model": model, "imputer": imputer, "features": features}, f)

print("model.pkl saved.")
