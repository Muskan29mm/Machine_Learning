import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Reproducibility
# To generate same results/sequence every run
tf.random.set_seed(42)
np.random.seed(42)


def split_sequence(sequence, n_steps):
    """Return input/output arrays using a simple sliding window."""
    n = len(sequence) - n_steps
    X = [sequence[i : i + n_steps] for i in range(n)]
    y = [sequence[i + n_steps] for i in range(n)]
    return np.array(X), np.array(y)


def build_model(n_steps: int, n_features: int, units: int = 50):
    model = Sequential([
        SimpleRNN(units, activation="relu", input_shape=(n_steps, n_features)),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def main():
    data = np.arange(100)
    n_steps = 3

    X, y = split_sequence(data, n_steps)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # (samples, timesteps, features)

    model = build_model(n_steps, 1)
    model.fit(X, y, epochs=50, verbose=0)

    x_input = np.array([70, 71, 72]).reshape((1, n_steps, 1))
    yhat = model.predict(x_input, verbose=0)[0, 0]

    print("Input:", [70, 71, 72])
    print(f"Predicted next: {yhat:.2f} (expected: 73)")


if __name__ == "__main__":
    main()