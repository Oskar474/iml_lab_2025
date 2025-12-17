import json
import requests
import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


WINDOW_SIZE = 200
EPOCHS = 20
BATCH_SIZE = 32

def pull_data():
    url = "https://query1.finance.yahoo.com/v8/finance/chart/AAPL?interval=1d&range=5y"

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    data = response.json()

    with open("AAPL.json", "w") as f:
        json.dump(data, f, indent=2)



def load_data(filename):
    with open(filename, "r") as f:
        data = json.load(f)

    close_prices = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
    close_prices = [c for c in close_prices if c is not None]

    return np.array(close_prices, dtype=np.float32)

def prep_data(prices):
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(prices.reshape(-1, 1))
    return scaled_prices, scaler

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

def build_model():
    model = keras.Sequential([
        LSTM(50, activation="tanh", input_shape=(WINDOW_SIZE, 1)),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )
    return model

def train_and_evaluate(X, y, window_size, epochs=20, batch_size=32):
    split = int(0.8 * len(X))

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(50, activation="tanh", input_shape=(window_size, 1)),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1
    )

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print(f"\nŚredni błąd kwadratowy (MSE): {mse:.6f}")

    return model, mse


def get_predictions(model, data, scaler, window_size):
    last_sequence = data[-window_size:]
    last_sequence = last_sequence.reshape(1, window_size, 1)

    next_price_scaled = model.predict(last_sequence)
    next_price = scaler.inverse_transform(next_price_scaled)


    return next_price[0][0]


def main():

    pull_data()

    prices = load_data("AAPL.json")
    scaled_prices, scaler = prep_data(prices)

    X, y = create_sequences(scaled_prices, WINDOW_SIZE)


    model = build_model()
    model, mse = train_and_evaluate(
        X, y,
        window_size=WINDOW_SIZE,
        epochs=20
    )

    prediction = get_predictions(
        model, scaled_prices, scaler, WINDOW_SIZE
    )

    print(f"Przewidywana cena: {prediction:.2f}")


if __name__ == "__main__":
    main()