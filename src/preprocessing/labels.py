from pathlib import Path
import re
from typing import Tuple, List

# age_gender_race_yyyymmddHHMMSSFFF.jpg
_RX = re.compile(r"^(\d{1,3})_(0|1)_([0-4])_\d{17}\.jpg$")

def parse_labels(image_paths: List[Path]) -> List[Tuple[int, int, int]]:
    image_labels = []
    for image_path in image_paths:
        match = _RX.match(image_path.name)
        age = int(match.group(1))
        gender = int(match.group(2))
        race = int(match.group(3))
        image_labels.append((age, gender, race))
    return image_labels
