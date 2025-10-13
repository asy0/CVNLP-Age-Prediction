from data_loading.load_images import get_image_paths
from validation.filenames_valid import filenames_valid
from visualization.visualize_samples import show_samples

def main():
    dataset_dir = "../in_the_wild_faces"
    if not filenames_valid(dataset_dir):
        return

    image_paths = get_image_paths(dataset_dir)
    show_samples(image_paths, 5)

if __name__ == "__main__":
    main()