from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

IMG_SIZE: Tuple[int, int] = (128, 128)
BATCH_SIZE: int = 32
SEED: int = 42
AGE_BINS = [0, 10, 20, 30, 40, 50, 60, 70, 120]

def build_splits(image_paths: List[Path], ages: List[int], test_size=0.1, val_size=0.1,):
    image_paths = np.array([str(path) for path in image_paths])
    ages = np.array(ages, dtype=np.float32)

    age_bins = np.digitize(ages, AGE_BINS, right=False)

    x_train_val, x_test, y_train_val, y_test, bins_train_val, bins_test = train_test_split(
        image_paths, ages, age_bins, test_size=test_size, random_state=None, stratify=age_bins
    )

    val_ratio = val_size / (1 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=val_ratio, random_state=None, stratify=bins_train_val
    )
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def preprocess_image(path: tf.Tensor) -> tf.Tensor:
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img

def augment_image(img: tf.Tensor) -> tf.Tensor:
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    return img

def make_tf_dataset(paths: np.ndarray, labels: np.ndarray, training=False) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load(path, label):
        img = preprocess_image(path)
        if training:
            img = augment_image(img)
        return img, tf.cast(label, tf.float32)

    if training:
        ds = ds.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds