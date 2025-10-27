from pathlib import Path
import re
from typing import List
import logging

# UTKFace filename pattern: age_gender_race_yyyymmddHHMMSSFFF.jpg
_RX = re.compile(r"^(\d{1,3})_(0|1)_([0-4])_\d{17}\.jpg$")

def parse_age(image_paths: List[Path]) -> List[int]:
    """
    Parse age from UTKFace filename format.
    
    Args:
        image_paths: List of image file paths
        
    Returns:
        List of age values extracted from filenames
        
    Raises:
        ValueError: If any filename doesn't match the expected format
    """
    ages = []
    invalid_files = []
    
    for path in image_paths:
        match = _RX.match(path.name)
        if match:
            age = int(match.group(1))
            # Validate age range (0-116 as per UTKFace specification)
            if 0 <= age <= 116:
                ages.append(age)
            else:
                invalid_files.append(f"{path.name}: age {age} out of range")
        else:
            invalid_files.append(f"{path.name}: invalid format")
    
    if invalid_files:
        logger = logging.getLogger(__name__)
        logger.warning(f"Found {len(invalid_files)} invalid files:")
        for invalid_file in invalid_files[:10]:  # Show first 10
            logger.warning(f"  {invalid_file}")
        if len(invalid_files) > 10:
            logger.warning(f"  ... and {len(invalid_files) - 10} more")
        
        if len(invalid_files) > len(image_paths) * 0.1:  # More than 10% invalid
            raise ValueError(f"Too many invalid filenames: {len(invalid_files)}/{len(image_paths)}")
    
    if not ages:
        raise ValueError("No valid age labels found in any filenames")
    
    logging.getLogger(__name__).info(f"Successfully parsed {len(ages)} age labels")
    return ages