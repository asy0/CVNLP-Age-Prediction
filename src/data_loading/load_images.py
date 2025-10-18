from pathlib import Path

def get_image_paths(dataset_dir):
    """Return a list of all .jpg image paths."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = (Path(__file__).resolve().parent.parent / dataset_dir).resolve()

    image_paths = list(dataset_dir.rglob("*.jpg"))
    return image_paths