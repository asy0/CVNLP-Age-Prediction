import tensorflow as tf
import matplotlib.pyplot as plt
import random

def show_samples(image_paths, n):
    sample_paths = random.sample(image_paths, n)
    images = []
    for path in sample_paths:
        img_raw = tf.io.read_file(str(path))
        img = tf.image.decode_jpeg(img_raw, channels=3)
        images.append((path, img))
#asdf
    plt.figure(figsize=(12, 6))
    for i, (path, img) in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img.numpy().astype("uint8"))
        plt.axis("off")
        plt.title(path.name[:15] + "...")
    plt.show()