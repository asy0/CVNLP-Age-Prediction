"""
Configuration settings for the age prediction project.
"""
from pathlib import Path
from typing import Tuple

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "in_the_wild_faces"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
BEST_MODEL_PATH = ARTIFACTS_DIR / "best_model_version.keras"
FINAL_MODEL_PATH = ARTIFACTS_DIR / "final_model_version.keras"

# Model configuration
INPUT_SHAPE: Tuple[int, int, int] = (128, 128, 3)
BATCH_SIZE: int = 32
EPOCHS: int = 20
LEARNING_RATE: float = 0.001
MIN_LEARNING_RATE: float = 1e-6

# Data configuration
IMG_SIZE: Tuple[int, int] = (128, 128)
TEST_SIZE: float = 0.1
VAL_SIZE: float = 0.1
SEED: int = 42
AGE_BINS = [0, 10, 20, 30, 40, 50, 60, 70, 120]

# Training configuration
EARLY_STOPPING_PATIENCE: int = 6
LR_REDUCTION_PATIENCE: int = 2
LR_REDUCTION_FACTOR: float = 0.5

# Data augmentation
AUGMENTATION_CONFIG = {
    'brightness_delta': 0.1,
    'contrast_lower': 0.9,
    'contrast_upper': 1.1,
    'horizontal_flip': True
}

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
