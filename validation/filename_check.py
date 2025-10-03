from pathlib import Path
import re

"""
Validate dataset filenames against UTKFace format.
Return True if all files are valid, otherwise False.

UTKFace format:
* The labels of each face image is embedded in the file name, formated like [age]_[gender]_[race]_[date&time].jpg
* [age] is an integer from 0 to 116, indicating the age
* [gender] is either 0 (male) or 1 (female)
* [race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).
* [date&time] is in the format of yyyymmddHHMMSSFFF, showing the date and time an image was collected to UTKFace
"""
# Initially 10 false entries were detected. Those were manually repaired.
# During repairs discovered a mistake in labeling (white person given false race label).
# --> Could be a wider spread problem.
def filename_check(dataset_dir):
    dataset_dir = Path(__file__).resolve().parent.parent / dataset_dir
    pattern = re.compile(r"^(\d{1,3})_(0|1)_[0-4]_\d{17}\.jpg$")

    invalid_files = []
    valid_count = 0

    for p in dataset_dir.rglob("*"):
        if p.is_file():
            if pattern.match(p.name):
                valid_count += 1
            else:
                invalid_files.append(p)

    print(f"✅ {valid_count} valid filenames found.")

    if invalid_files:
        print(f"❌ {len(invalid_files)} invalid filenames detected:")
        for f in invalid_files:
            print(f"  {f}")
        return False
    else:
        print("🎉 All files follow the UTKFace naming format.")
        return True
