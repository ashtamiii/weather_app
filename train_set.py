import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 🔍 **Step 1: Load Processed Data**
X_train = np.load("X_train.npy")  # Ensure the preprocessing script saves these files
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

# 🔍 **Step 2: Define LSTM Model**
model = Sequential([
    LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25, activation='relu'),
    Dense(units=1, activation='sigmoid')  # Binary classification (Rain/No Rain)
])

# 🔍 **Step 3: Compile Model**
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 🔍 **Step 4: Train Model**
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])  # Increased epochs to 50

# 🔍 **Step 5: Save Trained Model**
model.save("lstm_weather_model.h5")

# 🔍 **Step 6: Evaluate Model**
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# 🔍 **Step 7: Plot Training History**
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.show()

print("LSTM Model Training Completed and Saved Successfully ✅")
model.save("weather_lstm_model.keras")
