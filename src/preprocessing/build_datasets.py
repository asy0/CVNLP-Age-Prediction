from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

def build_splits(image_paths: List[Path], ages: List[int], test_size=0.1, val_size=0.1, seed=42):
    image_paths = np.array([str(path) for path in image_paths])
    ages = np.array(ages, dtype=np.float32)

    bins = [0, 10, 20, 30, 40, 50, 60, 70, 120]
    age_bins = np.digitize(ages, bins, right=False)

    X_train_val, X_test, y_train_val, y_test, bins_train_val, bins_test = train_test_split(
        image_paths, ages, age_bins, test_size=test_size, random_state=seed, stratify=age_bins
    )

    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=seed, stratify=bins_train_val
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# Image preprocessing for TensorFlow
def preprocess_image(path: tf.Tensor, target_size=(128, 128)) -> tf.Tensor:
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = tf.cast(img, tf.float32) / 255.0
    return img

def augment_image(img: tf.Tensor) -> tf.Tensor:
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    return img

def make_tf_dataset(paths: np.ndarray, labels: np.ndarray, batch_size=32, training=False, target_size=(128, 128)) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load(path, label):
        img = preprocess_image(path, target_size)
        if training:
            img = augment_image(img)
        return img, tf.cast(label, tf.float32)

    if training:
        ds = ds.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds