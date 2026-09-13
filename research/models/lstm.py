import numpy as np


def make_sequences(X, y, sequence_length):

    X = np.asarray(X)
    y = np.asarray(y)

    X_sequences = []
    y_sequences = []

    for i in range(sequence_length, len(X)):

        X_sequences.append(
            X[i - sequence_length:i]
        )

        y_sequences.append(
            y[i]
        )

    return (
        np.asarray(X_sequences),
        np.asarray(y_sequences)
    )


def build_model(
    sequence_length,
    n_features
):

    import tensorflow as tf

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(
                sequence_length,
                n_features
            )
        ),

        tf.keras.layers.LSTM(
            64,
            return_sequences=True
        ),

        tf.keras.layers.Dropout(
            0.20
        ),

        tf.keras.layers.LSTM(
            32
        ),

        tf.keras.layers.Dropout(
            0.20
        ),

        tf.keras.layers.Dense(
            16,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            1
        )
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model


def train(
    X_train,
    y_train,
    X_validation,
    y_validation,
    epochs=30,
    batch_size=32
):

    import tensorflow as tf

    tf.keras.utils.set_random_seed(
        42
    )

    model = build_model(
        X_train.shape[1],
        X_train.shape[2]
    )

    early_stopping = (
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )
    )

    model.fit(
        X_train,
        y_train,

        validation_data=(
            X_validation,
            y_validation
        ),

        epochs=epochs,
        batch_size=batch_size,

        callbacks=[
            early_stopping
        ],

        verbose=0
    )

    return model