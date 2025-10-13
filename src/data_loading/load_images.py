from pathlib import Path

def get_image_paths(dataset_dir):
    """Return a list of all .jpg image paths."""
    dataset_path = (Path(__file__).resolve().parent.parent / dataset_dir).resolve()
    image_paths = list(dataset_path.glob("*.jpg"))
    return image_paths