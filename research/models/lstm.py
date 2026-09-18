import numpy as np

def make_sequences(X, y, sequence_length):
    xs, ys = [], []
    for i in range(sequence_length, len(X)):
        xs.append(X[i-sequence_length:i])
        ys.append(y[i])
    return np.asarray(xs), np.asarray(ys)

def build_model(sequence_length, n_features):
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input((sequence_length, n_features)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def train(X_train, y_train, X_val, y_val, epochs=30, batch_size=32):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    model = build_model(X_train.shape[1], X_train.shape[2])
    stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=epochs, batch_size=batch_size,
              callbacks=[stop], verbose=0)
    return model
