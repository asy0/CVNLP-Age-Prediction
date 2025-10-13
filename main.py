from validation.filenames_valid import filenames_valid

def main():
    if not filenames_valid(dataset_dir = "in_the_wild_faces"):
        return

if __name__ == "__main__":
    main()