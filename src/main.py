import os

# Silence Tensorflow INFO and WARNING, leave ERROR
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from pathlib import Path
from data_loading.get_image_paths import get_image_paths
from model.model import build_model
from preprocessing.build_datasets import build_splits, make_tf_dataset
from preprocessing.parse_age import parse_age
from validation.filenames_valid import filenames_valid
from visualization.visualize_samples import show_samples

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR  = PROJECT_ROOT / "in_the_wild_faces"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

def main():
    """
    # run once after every change in "in_the_wild_faces"
    # searches for wrong file names
    filenames_valid(DATASET_DIR)
    return
    """

    """ get 'SYSTEM:/PROJECT_ROOT/in_the_wild_faces/*.jpg' for all images """
    image_paths = get_image_paths(DATASET_DIR)

    """ shows some sample images """
    show_samples(image_paths, 5)

    """ get [age_1, age_2, ..., age_n] for all images"""
    age_labels = parse_age(image_paths)

    """ 
    store 
    (['SYSTEM:/PROJECT_ROOT/in_the_wild_faces/first.jpg', ..., 'SYSTEM:/PROJECT_ROOT/in_the_wild_faces/last.jpg'] 
    [age_1, age_2, ..., age_n]) 
    in 3 splits: Training, Validation, Testing.
    amount of values stored:
    train : (19286, 19286)... 80% of data
    val   : (2411, 2411)...  10% of data
    test  : (2411, 2411)...  10% of data
    """
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = build_splits(image_paths, age_labels)

    """
    Convert splits into TensorFlow Datasets (tf.data.Dataset)
    More efficient for training by creating batches of data
                                                      images         age
    train_ds : (_PrefetchDataset) 603 batches of (32, 128, 128, 3), (32,)
    val_ds   : (_PrefetchDataset) 76  batches of (32, 128, 128, 3), (32,)
    test_ds  : (_PrefetchDataset) 76  batches of (32, 128, 128, 3), (32,)
    
    images → RGB images normalized to [0, 1]
    ages → age labels corresponding to each image
    
    32 batchsize, images
    128x pixels
    128y pixels
    3 channels (RGB)
    32 batchsize, age labels
    """
    train_ds = make_tf_dataset(x_train, y_train, training=True)
    val_ds = make_tf_dataset(x_val, y_val)
    test_ds = make_tf_dataset(x_test, y_test)

    """
    Build the model
    - A 3×3 filter (kernel) scans the image and computes pattern activations at each layer
    - After each Conv Layer, a 2×2 MaxPool halves width/height (keeps strongest responses).
    
    Input Layer: (128 Pixels, 128 Pixels, 3 channels - RGB) -   the image
    
    Conv Layer 1: (126 Features, 126 Features, 32 Filters)  -   Filters are small scanning grids that detect patterns
    Conv Layer 2: (61 Features, 61 Features, 64 Filters)    -   Features are detected patterns from previous layers
    Conv Layer 3: (28 Features, 28 Features, 128 Filters)
    
    Flatten Layer: (14×14×128 = 25,088 features)    -   Just a 1D Array version of the same information
    
    2 Dense Layer: Combine all learned features to find complex relationships
    2 Dropouts: Random Deactivation of Neurons during training to reduce overfitting
    
    Output Layer: 1 Neuron, Age Prediction
    .
    .
    .
    model.summary() ->
    Model: "sequential"
    ┌─────────────────────────────────┬────────────────────────┬───────────────┐
    │ Layer (type)                    │ Output Shape           │       Param # │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ conv2d (Conv2D)                 │ (None, 126, 126, 32)   │           896 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ max_pooling2d (MaxPooling2D)    │ (None, 63, 63, 32)     │             0 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ conv2d_1 (Conv2D)               │ (None, 61, 61, 64)     │        18,496 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ max_pooling2d_1 (MaxPooling2D)  │ (None, 30, 30, 64)     │             0 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ conv2d_2 (Conv2D)               │ (None, 28, 28, 128)    │        73,856 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ max_pooling2d_2 (MaxPooling2D)  │ (None, 14, 14, 128)    │             0 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ flatten (Flatten)               │ (None, 25088)          │             0 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ dense (Dense)                   │ (None, 128)            │     3,211,392 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ dropout (Dropout)               │ (None, 128)            │             0 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ dense_1 (Dense)                 │ (None, 64)             │         8,256 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ dropout_1 (Dropout)             │ (None, 64)             │             0 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ age_output (Dense)              │ (None, 1)              │            65 │
    └─────────────────────────────────┴────────────────────────┴───────────────┘
     Total params: 3,312,961 (12.64 MB)
     Trainable params: 3,312,961 (12.64 MB)
     Non-trainable params: 0 (0.00 B)
    """
    model = build_model()

    """
    EarlyStopping: If no improvement after 5 epochs, the model is stopped. Best Version is saved.
    ReduceLROnPlateau: Reduce learning rate if validation error is not improving for 2 epochs.
    ModelCheckpoint: Save Copy of model each time validation MAE improves.
    """
    ckpt_path = ARTIFACTS_DIR / "best_model_version.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_mae", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_mae", factor=0.5, patience=2, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(ckpt_path), monitor="val_mae", save_best_only=True)
    ]

    """
    model training with 30 runs (epochs)
    
    """
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        callbacks=callbacks,
    )

    """
    Saving final model
    """
    final_path = ARTIFACTS_DIR / "final_model_version.keras"
    model.save(final_path)

if __name__ == "__main__":
    main()