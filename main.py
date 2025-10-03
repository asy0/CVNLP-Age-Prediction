from validation.filename_check import filename_check

def main():
    print("🔍 Checking dataset filenames...")
    ok = filename_check("in_the_wild_faces") # pass dataset dir here

    if not ok:
        print("❌ Dataset filename check failed. Stopping program.")
        return

    print("✅ Dataset filename check passed. Continuing with training pipeline...")
    # === Your ML code continues here ===
    # e.g., train_test_split, DataLoader, training loop, etc.

if __name__ == "__main__":
    main()