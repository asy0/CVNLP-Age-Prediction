import tensorflow as tf
from tensorflow.keras import layers, models
from config import INPUT_SHAPE, LEARNING_RATE

def build_model() -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=INPUT_SHAPE),

        layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # --- Classification head ---
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),

        # --- Output layer ---
        layers.Dense(1, activation='linear', name="age_prediction")
    ])

    # Compile for regression (predict continuous age)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mae',
        metrics=['mae']
    )

    return model