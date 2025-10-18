import os
from pathlib import Path

# Silence Tensorflow INFO and WARNING, leave ERROR
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from data_loading.load_images import get_image_paths
from preprocessing.labels import parse_labels
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
    show_samples(image_paths, 5)
    image_labels = parse_labels(image_paths)

if __name__ == "__main__":
    main()