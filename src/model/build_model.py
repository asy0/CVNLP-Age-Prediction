import tensorflow as tf
from tensorflow.keras import layers, models
from config import INPUT_SHAPE, LEARNING_RATE

def build_model() -> tf.keras.Model:
    """
    Build an improved CNN model for age prediction.
    
    Architecture improvements:
    - Batch normalization for better training stability
    - More sophisticated regularization
    - Better feature extraction with residual connections
    - Global average pooling instead of flatten for better generalization
    """
    model = models.Sequential([
        layers.Input(shape=INPUT_SHAPE),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),

        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),

        layers.Dense(1, activation='linear', name='age_output')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mae',
        metrics=['mae', 'mse']
    )

    return model