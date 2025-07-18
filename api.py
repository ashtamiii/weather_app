import requests
import pandas as pd
import datetime

# OpenWeatherMap API Key
API_KEY = '54a57bc234ad752a4f59e59cd372201d'
CITY = "Chennai"  # Change this to your location
URL = f"http://api.openweathermap.org/data/2.5/forecast?q={CITY}&appid={API_KEY}&units=metric"

# Fetch data
response = requests.get(URL)
data = response.json()

# Extract relevant data
weather_data = []
for entry in data['list']:
    timestamp = datetime.datetime.utcfromtimestamp(entry['dt'])
    temp = entry['main']['temp']
    humidity = entry['main']['humidity']
    weather_data.append([timestamp, temp, humidity])

# Convert to DataFrame
df = pd.DataFrame(weather_data, columns=["Date", "Temperature", "Humidity"])
print(df.head())  # Check data
df.to_csv("weather_data.csv", index=False)  # Save for training
