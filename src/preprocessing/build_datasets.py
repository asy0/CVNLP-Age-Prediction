from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from config import IMG_SIZE, BATCH_SIZE, SEED, AGE_BINS, AUGMENTATION_CONFIG

def build_splits(image_paths: List[Path], ages: List[int], 
                test_size: float = 0.1, val_size: float = 0.1) -> Tuple:
    """
    Build stratified train/validation/test splits.
    
    Args:
        image_paths: List of image file paths
        ages: List of age labels
        test_size: Fraction of data for testing
        val_size: Fraction of data for validation
        
    Returns:
        Tuple of (x_train, y_train), (x_val, y_val), (x_test, y_test)
    """
    image_paths = np.array([str(path) for path in image_paths])
    ages = np.array(ages, dtype=np.float32)

    # Create age bins for stratified splitting
    age_bins = np.digitize(ages, AGE_BINS, right=False)

    # First split: separate test set
    x_train_val, x_test, y_train_val, y_test, bins_train_val, bins_test = train_test_split(
        image_paths, ages, age_bins, 
        test_size=test_size, 
        random_state=SEED, 
        stratify=age_bins
    )

    # Second split: separate validation from training
    val_ratio = val_size / (1 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, 
        test_size=val_ratio, 
        random_state=SEED, 
        stratify=bins_train_val
    )
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def preprocess_image(path: tf.Tensor) -> tf.Tensor:
    """
    Preprocess a single image: load, decode, resize, and normalize.
    
    Args:
        path: Tensor containing the file path
        
    Returns:
        Preprocessed image tensor
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img

def augment_image(img: tf.Tensor) -> tf.Tensor:
    """
    Apply data augmentation to an image - minimal for faster convergence.
    
    Args:
        img: Input image tensor
        
    Returns:
        Augmented image tensor
    """
    # Only very basic augmentation for speed
    if AUGMENTATION_CONFIG['horizontal_flip']:
        img = tf.image.random_flip_left_right(img)
    
    # Ensure pixel values stay in valid range
    img = tf.clip_by_value(img, 0.0, 1.0)
    
    return img

def make_tf_dataset(paths: np.ndarray, labels: np.ndarray, training: bool = False) -> tf.data.Dataset:
    """
    Create a TensorFlow dataset with optimized data loading.
    
    Args:
        paths: Array of image file paths
        labels: Array of age labels
        training: Whether this is for training (enables augmentation and shuffling)
        
    Returns:
        Optimized TensorFlow dataset
    """
    # Create dataset from tensors
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load_and_preprocess(path, label):
        """Load and preprocess a single image-label pair."""
        img = preprocess_image(path)
        if training:
            img = augment_image(img)
        return img, tf.cast(label, tf.float32)

    # Apply preprocessing with parallel processing
    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle for training
    if training:
        ds = ds.shuffle(buffer_size=min(len(paths), 1000), reshuffle_each_iteration=True)

    # Batch and prefetch for optimal performance
    ds = ds.batch(BATCH_SIZE, drop_remainder=training)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds