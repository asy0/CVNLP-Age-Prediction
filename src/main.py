import os
from pathlib import Path

# Silence Tensorflow INFO and WARNING, leave ERROR
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from data_loading.load_images import get_image_paths
from preprocessing.build_datasets import build_splits#, make_dataset
from preprocessing.labels import parse_age
from validation.filenames_valid import filenames_valid
from visualization.visualize_samples import show_samples

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR  = PROJECT_ROOT / "in_the_wild_faces"

def main():
    """
    # run once after every change in "in_the_wild_faces"
    # searches for wrong file names
    filenames_valid(DATASET_DIR)
    return
    """

    image_paths = get_image_paths(DATASET_DIR)
    # show_samples(image_paths, 5)
    age_labels = parse_age(image_paths)

    # Split dataset
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = build_splits(image_paths, age_labels)

    print(X_train, y_train)
    """ UNDER CONSTRUCTION
    # Create tf.data.Dataset objects
    train_ds = make_dataset(X_train, y_train, training=True)
    val_ds = make_dataset(X_val, y_val)
    test_ds = make_dataset(X_test, y_test)
    """

if __name__ == "__main__":
    main()