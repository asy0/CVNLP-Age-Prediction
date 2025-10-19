import os
from pathlib import Path

# Silence Tensorflow INFO and WARNING, leave ERROR
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from data_loading.load_images import get_image_paths
from preprocessing.build_datasets import build_splits, make_tf_dataset
from preprocessing.parse_age import parse_age
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
    in 3 splits. Training, Validation, Testing.
    amount of values stored:
    train : (19286, 19286)... 80% of data
    val   : (2411, 2411)...  10% of data
    test  : (2411, 2411)...  10% of data
    """
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = build_splits(image_paths, age_labels)

    """
    Convert splits into TensorFlow Datasets (tf.data.Dataset)
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
    train_ds = make_tf_dataset(X_train, y_train, training=True)
    val_ds = make_tf_dataset(X_val, y_val)
    test_ds = make_tf_dataset(X_test, y_test)

if __name__ == "__main__":
    main()