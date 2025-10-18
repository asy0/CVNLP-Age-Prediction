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

pattern = re.compile(r"^(\d{1,3})_(0|1)_[0-4]_\d{17}\.jpg$")

def filenames_valid(dataset_dir):
    print("🔍 Checking dataset filenames...")
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = (Path(__file__).resolve().parent.parent / dataset_dir).resolve()

    invalid, valid = [], 0

    for p in dataset_dir.rglob("*"):
        if p.is_file():
            if pattern.match(p.name):
                valid += 1
            else:
                invalid.append(p)

    print(f"✅ {valid} valid filenames found.")

    if invalid:
        count = len(invalid)
        noun = "filename" if count == 1 else "filenames"
        print(f"❌ {count} invalid {noun} detected:")
        for f in invalid:
            print(f"  {f.name}")
        return
    else:
        print("🎉 All files follow the UTKFace naming format.")
        return