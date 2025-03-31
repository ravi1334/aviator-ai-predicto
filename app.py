import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import os
from flask import Flask, jsonify

app = Flask(__name__)

def load_real_data(csv_path):
    """Load real crash multipliers from a CSV file, or create an empty one if missing."""
    if not os.path.exists(csv_path):
        pd.DataFrame(columns=["Crash Multiplier"]).to_csv(csv_path, index=False)
    df = pd.read_csv(csv_path)
    return df["Crash Multiplier"].values if not df.empty else np.array([])

def create_model():
    """Build a neural network model for predicting crash multipliers."""
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(10,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='linear')  # Predicts the next crash multiplier
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

csv_path = "crash_multipliers.csv"
data = load_real_data(csv_path)

def get_prediction():
    if len(data) > 10:
        X_train = np.array([data[i:i+10] for i in range(len(data)-10)])
        y_train = np.array([data[i+10] for i in range(len(data)-10)])
        
        model = create_model()
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
        
        last_10_crashes = np.array([data[-10:]])
        predicted_next = model.predict(last_10_crashes)[0][0]
        return round(predicted_next, 2)
    return None

@app.route('/predict', methods=['GET'])
def predict():
    prediction = get_prediction()
    if prediction:
        return jsonify({"Predicted Crash Multiplier": prediction})
    else:
        return jsonify({"Error": "Not enough data for prediction"})

if __name__ == '__main__':
    app.run(debug=True)
    
