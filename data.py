import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("weather.csv")  # Replace with your actual dataset filename

# Convert 'Date' column to datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Set 'Date' as the index for time-series modeling
df.set_index("Date", inplace=True)

# 🔍 **Step 1: Handle Duplicate Indices**
df = df[~df.index.duplicated(keep='first')]  # Keep only the first occurrence

# 🔍 **Step 2: Handle Missing Values**
df.replace("NA", np.nan, inplace=True)  # Replace "NA" strings with NaN
df.dropna(subset=["MinTemp", "MaxTemp", "Rainfall", "Humidity9am", "Humidity3pm"], inplace=True)

# Fill missing numerical values with mean
num_cols = ["Evaporation", "Sunshine", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
            "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm",
            "Temp9am", "Temp3pm", "RISK_MM"]
df[num_cols] = df[num_cols].apply(lambda col: col.fillna(col.mean()))

# Fill missing categorical values with mode
cat_cols = ["WindGustDir", "WindDir9am", "WindDir3pm"]
df[cat_cols] = df[cat_cols].apply(lambda col: col.fillna(col.mode()[0]))

# 🔍 **Step 3: Encode Categorical Variables**
label_encoders = {}
for col in cat_cols + ["RainToday", "RainTomorrow"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoders for later use

# 🔍 **Step 4: Normalize Numerical Features**
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# 🔍 **Step 5: Feature Selection for LSTM**
selected_features = ["MinTemp", "MaxTemp", "Rainfall", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm"]
target = "RainTomorrow"  # Predicting if it will rain tomorrow

X = df[selected_features]
y = df[target]

# 🔍 **Step 6: Reshape Data for LSTM**
X = np.array(X).reshape(X.shape[0], 1, X.shape[1])  # Reshape to 3D (samples, timesteps, features)

# Split into Train & Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# 🔍 **Step 7: Plot Data to Check Distribution**
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df["Humidity9am"], y=df["Temp9am"], alpha=0.5)
plt.xlabel("Humidity at 9 AM")
plt.ylabel("Temperature at 9 AM")
plt.title("Humidity vs. Temperature")
plt.show()

# Print final dataset shape
print("Training data shape:", X_train.shape, y_train.shape)
print("Testing data shape:", X_test.shape, y_test.shape)

# Save processed data for training
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("✅ Preprocessed data saved successfully: X_train.npy, X_test.npy, y_train.npy, y_test.npy")

