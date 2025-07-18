from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import requests

app = Flask(__name__)

# Load the trained AI weather prediction model
MODEL_PATH = "weather_lstm_model.keras"  # Ensure this path is correct
model = tf.keras.models.load_model(MODEL_PATH)

# OpenWeatherMap API Key
API_KEY = "54a57bc234ad752a4f59e59cd372201d"
CURRENT_WEATHER_URL = "http://api.openweathermap.org/data/2.5/weather"
FORECAST_URL = "http://api.openweathermap.org/data/2.5/forecast"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        city = request.form["city"]

        # Get current weather data
        params = {"q": city, "appid": API_KEY, "units": "metric"}
        response = requests.get(CURRENT_WEATHER_URL, params=params)

        if response.status_code == 200:
            data = response.json()

            # Extract current weather details
            temp = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            pressure = data["main"]["pressure"]
            wind_speed = data["wind"]["speed"]
            cloud_coverage = data["clouds"]["all"]
            visibility = data.get("visibility", 10000) / 1000  # Convert meters to km
            rain = data.get("rain", {}).get("1h", 0)

            # Prepare input for AI model (7 features)
            input_features = np.array([[temp, humidity, pressure, wind_speed, cloud_coverage, visibility, rain]])
            input_data = input_features.reshape(1, 1, 7)

            # Predict rainfall chance using AI model
            prediction = model.predict(input_data)[0][0] * 100

            # Get 5-day forecast data
            forecast_response = requests.get(FORECAST_URL, params=params)

            forecast_data = []
            if forecast_response.status_code == 200:
                forecast_json = forecast_response.json()
                forecast_list = forecast_json["list"]

                # Extract next 5 days' forecast (every 24 hours)
                for i in range(0, len(forecast_list), 8):
                    day_data = forecast_list[i]
                    forecast_temp = day_data["main"]["temp"]
                    forecast_humidity = day_data["main"]["humidity"]
                    forecast_wind_speed = day_data["wind"]["speed"]
                    forecast_clouds = day_data["clouds"]["all"]
                    forecast_rain = day_data.get("rain", {}).get("3h", 0)

                    # AI Prediction for future days
                    future_input = np.array([[forecast_temp, forecast_humidity, pressure, forecast_wind_speed, forecast_clouds, visibility, forecast_rain]])
                    future_input_data = future_input.reshape(1, 1, 7)
                    future_rain_chance = model.predict(future_input_data)[0][0] * 100

                    forecast_data.append({
                        "date": day_data["dt_txt"].split(" ")[0],  # Extract date only
                        "temp": round(forecast_temp, 1),
                        "humidity": forecast_humidity,
                        "wind_speed": round(forecast_wind_speed, 1),
                        "rain_chance": round(future_rain_chance, 2)
                    })

            return render_template(
                "index.html",
                city=city,
                temp=temp,
                humidity=humidity,
                pressure=pressure,
                wind_speed=wind_speed,
                cloud_coverage=cloud_coverage,
                visibility=visibility,
                rain=rain,
                prediction=round(prediction, 2),
                forecast=forecast_data
            )

        else:
            return render_template("index.html", error="City not found!")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
