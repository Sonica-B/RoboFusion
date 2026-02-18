import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from RoboFusion.DataCollection.scripts.utils.configs import *

# Load and preprocess the dataset
data_file = "D:/RoboFusion2/RoboFusion/DataCollection/raw_data/gestures/gesture_data.csv"

# Read valid rows with 43 values
with open(data_file, 'r') as file:
    lines = file.readlines()

valid_lines = [line for line in lines if len(line.strip().split(',')) == 43]
data = pd.DataFrame(
    [line.strip().split(',') for line in valid_lines],
    columns=['label'] + [f'feature_{i+1}' for i in range(42)]
)

data.iloc[:, 1:] = data.iloc[:, 1:].astype(float)
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# Shuffle data
X, y = shuffle(X, y, random_state=42)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler for use in app.py
scaler_path = "D:/RoboFusion2/RoboFusion/MLModel/models/scaler.pkl"
np.save(scaler_path, [scaler.mean_, scaler.scale_])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(42,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y_encoded)), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32)

# Save model
model.save(MODEL_PATH, save_format="keras")
print(f"Model saved to {MODEL_PATH}")

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
