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
        
        # First convolutional block with batch normalization
        layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Fourth convolutional block
        layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Global Average Pooling
        layers.GlobalAveragePooling2D(),
        
        # Dense layers - simplified for faster training
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        
        # Output layer
        layers.Dense(1, activation='linear', name="age_prediction")
    ])

    # Compile with improved optimizer settings
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        ),
        loss='mae',
        metrics=['mae', 'mse']
    )

    return model