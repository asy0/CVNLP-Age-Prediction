import tensorflow as tf
import matplotlib.pyplot as plt
import random

from pathlib import Path

def show_samples(image_paths, n):
    sample_paths = random.sample(image_paths, n)
    images = []
    for path in sample_paths:
        img_raw = tf.io.read_file(str(path))
        img = tf.image.decode_jpeg(img_raw, channels=3)
        images.append((path, img))

    plt.figure(figsize=(12, 6))
    for i, (path, img) in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img.numpy().astype("uint8"))
        plt.axis("off")
        plt.title(path.name[:15] + "...")
    plt.show()

def show_results(x_test, y_test, y_pred, n):
    plt.figure(figsize=(15, 6))

    for i in range(n):
        img_raw = tf.io.read_file(x_test[i])
        img = tf.image.decode_jpeg(img_raw, channels=3)
        img = tf.image.resize(img, (128, 128))

        plt.subplot(2, 5, i + 1)
        plt.imshow(img.numpy().astype("uint8"))
        plt.axis("off")

        filename = Path(x_test[i]).name
        plt.title(f"{filename}\nPredicted Age: {y_pred[i]:.0f}\nTrue Age: {y_test[i]:.0f}",
                  fontsize=9, pad=5)
    plt.show()