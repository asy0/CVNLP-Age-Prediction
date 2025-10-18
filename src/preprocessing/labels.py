from pathlib import Path
import re
from typing import Tuple, List

# age_gender_race_yyyymmddHHMMSSFFF.jpg
_RX = re.compile(r"^(\d{1,3})_(0|1)_([0-4])_\d{17}\.jpg$")

def parse_age(image_paths: List[Path]) -> List[int]:
    ages = []
    for path in image_paths:
        match = _RX.match(path.name)
        ages.append(int(match.group(1)))

    return ages