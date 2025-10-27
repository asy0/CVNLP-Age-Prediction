import argparse
import os
import sys
import logging
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional

# Silence Tensorflow INFO and WARNING, leave ERROR
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

from data_loading.get_image_paths import get_image_paths
from model.build_model import build_model
from preprocessing.build_datasets import build_splits, make_tf_dataset
from preprocessing.parse_age import parse_age
from validation.filenames_valid import filenames_valid
from visualization.visualize_samples import show_samples, show_results
from config import (
    PROJECT_ROOT, DATASET_DIR, ARTIFACTS_DIR, BEST_MODEL_PATH, FINAL_MODEL_PATH,
    EPOCHS, EARLY_STOPPING_PATIENCE, LR_REDUCTION_PATIENCE, LR_REDUCTION_FACTOR,
    MIN_LEARNING_RATE, LOG_LEVEL, LOG_FORMAT
)

def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(PROJECT_ROOT / "age_prediction.log")
        ]
    )
    return logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Age Prediction Model Training and Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py                    # Use saved model for evaluation
  python src/main.py --train           # Train a new model
  python src/main.py --validate-only   # Only validate dataset filenames
        """
    )
    parser.add_argument(
        "--train", 
        action="store_true", 
        help="Train a new model for this run"
    )
    parser.add_argument(
        "--validate-only", 
        action="store_true", 
        help="Only validate dataset filenames and exit"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS})"
    )
    return parser.parse_args()

def validate_environment() -> None:
    """Validate that required directories and files exist."""
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")
    
    if not ARTIFACTS_DIR.exists():
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created artifacts directory: {ARTIFACTS_DIR}")

def train_model(model: tf.keras.Model, train_ds: tf.data.Dataset, 
                val_ds: tf.data.Dataset, epochs: int, logger: logging.Logger) -> None:
    """Train the model with callbacks and save the best version."""
    # Setup callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_mae", 
            patience=EARLY_STOPPING_PATIENCE, 
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_mae", 
            factor=LR_REDUCTION_FACTOR, 
            patience=LR_REDUCTION_PATIENCE, 
            min_lr=MIN_LEARNING_RATE,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH), 
            monitor="val_mae", 
            save_best_only=True,
            verbose=1
        )
    ]
    
    logger.info(f"Training for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(FINAL_MODEL_PATH)
    logger.info(f"Final model saved to: {FINAL_MODEL_PATH}")

def load_model(logger: logging.Logger) -> tf.keras.Model:
    """Load the best available model."""
    if BEST_MODEL_PATH.exists():
        logger.info(f"Loading best model from: {BEST_MODEL_PATH}")
        return tf.keras.models.load_model(BEST_MODEL_PATH, compile=True)
    elif FINAL_MODEL_PATH.exists():
        logger.info(f"Loading final model from: {FINAL_MODEL_PATH}")
        return tf.keras.models.load_model(FINAL_MODEL_PATH, compile=True)
    else:
        raise FileNotFoundError(
            "No trained model found. Please train a model first with --train flag."
        )

def main():
    """Main function for age prediction model training and evaluation."""
    # Setup logging and parse arguments
    logger = setup_logging()
    args = parse_arguments()
    
    try:
        # Validate environment
        validate_environment()
        
        # Validate dataset filenames if requested
        if args.validate_only:
            logger.info("Validating dataset filenames...")
            filenames_valid(DATASET_DIR)
            return
        
        logger.info("Starting age prediction pipeline...")
        
        """ get 'SYSTEM:/PROJECT_ROOT/in_the_wild_faces/*.jpg' for all images """
        logger.info("Loading image paths...")
        image_paths = get_image_paths(DATASET_DIR)
        logger.info(f"Found {len(image_paths)} images")
        
        if not image_paths:
            raise ValueError("No images found in dataset directory")
        
        """ shows some sample images """
        logger.info("Displaying sample images...")
        show_samples(image_paths, 6)
        
        """ get [age_1, age_2, ..., age_n] for all images"""
        logger.info("Parsing age labels from filenames...")
        age_labels = parse_age(image_paths)
        logger.info(f"Parsed {len(age_labels)} age labels")

        """ 
        store 
        (['SYSTEM:/PROJECT_ROOT/in_the_wild_faces/first.jpg', ..., 'SYSTEM:/PROJECT_ROOT/in_the_wild_faces/last.jpg'] 
        [age_1, age_2, ..., age_n]) 
        in 3 splits: Training, Validation, Testing.
        amount of values stored:
        train : (19286, 19286)... 80% of data
        val   : (2411, 2411)...  10% of data
        test  : (2411, 2411)...  10% of data
        """
        logger.info("Building data splits...")
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = build_splits(image_paths, age_labels)
        logger.info(f"Data splits - Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")
        
        """
        Convert splits into TensorFlow Datasets (tf.data.Dataset)
        More efficient for training by creating batches of data
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
        logger.info("Creating TensorFlow datasets...")
        train_ds = make_tf_dataset(x_train, y_train, training=True)
        val_ds = make_tf_dataset(x_val, y_val)
        test_ds = make_tf_dataset(x_test, y_test)
        logger.info("TensorFlow datasets created successfully")

        """
        Build the model
        - A 3×3 filter (kernel) scans the image and computes pattern activations at each layer
        - After each Conv Layer, a 2×2 MaxPool halves width/height (keeps strongest responses).
        
        Input Layer: (128 Pixels, 128 Pixels, 3 channels - RGB) -   the image
        
        Conv Layer 1: (126 Features, 126 Features, 32 Filters)  -   Filters are small scanning grids that detect patterns
        Conv Layer 2: (61 Features, 61 Features, 64 Filters)    -   Features are detected patterns from previous layers
        Conv Layer 3: (28 Features, 28 Features, 128 Filters)
        
        Flatten Layer: (14×14×128 = 25,088 features)    -   Just a 1D Array version of the same information
        
        2 Dense Layer: Combine all learned features to find complex relationships
        2 Dropouts: Random Deactivation of Neurons during training to reduce overfitting
        
        Output Layer: 1 Neuron, Age Prediction
        .
        .
        .
        model.summary() ->
        Model: "sequential"
        ┌─────────────────────────────────┬────────────────────────┬───────────────┐
        │ Layer (type)                    │ Output Shape           │       Param # │
        ├─────────────────────────────────┼────────────────────────┼───────────────┤
        │ conv2d (Conv2D)                 │ (None, 126, 126, 32)   │           896 │
        ├─────────────────────────────────┼────────────────────────┼───────────────┤
        │ max_pooling2d (MaxPooling2D)    │ (None, 63, 63, 32)     │             0 │
        ├─────────────────────────────────┼────────────────────────┼───────────────┤
        │ conv2d_1 (Conv2D)               │ (None, 61, 61, 64)     │        18,496 │
        ├─────────────────────────────────┼────────────────────────┼───────────────┤
        │ max_pooling2d_1 (MaxPooling2D)  │ (None, 30, 30, 64)     │             0 │
        ├─────────────────────────────────┼────────────────────────┼───────────────┤
        │ conv2d_2 (Conv2D)               │ (None, 28, 28, 128)    │        73,856 │
        ├─────────────────────────────────┼────────────────────────┼───────────────┤
        │ max_pooling2d_2 (MaxPooling2D)  │ (None, 14, 14, 128)    │             0 │
        ├─────────────────────────────────┼────────────────────────┼───────────────┤
        │ flatten (Flatten)               │ (None, 25088)          │             0 │
        ├─────────────────────────────────┼────────────────────────┼───────────────┤
        │ dense (Dense)                   │ (None, 128)            │     3,211,392 │
        ├─────────────────────────────────┼────────────────────────┼───────────────┤
        │ dropout (Dropout)               │ (None, 128)            │             0 │
        ├─────────────────────────────────┼────────────────────────┼───────────────┤
        │ dense_1 (Dense)                 │ (None, 64)             │         8,256 │
        ├─────────────────────────────────┼────────────────────────┼───────────────┤
        │ dropout_1 (Dropout)             │ (None, 64)             │             0 │
        ├─────────────────────────────────┼────────────────────────┼───────────────┤
        │ age_output (Dense)              │ (None, 1)              │            65 │
        └─────────────────────────────────┴────────────────────────┴───────────────┘
         Total params: 3,312,961 (12.64 MB)
         Trainable params: 3,312,961 (12.64 MB)
         Non-trainable params: 0 (0.00 B)
        """
        logger.info("Building model...")
        model = build_model()
        logger.info("Model built successfully")
        
        if args.train:
            """
            EarlyStopping: If no improvement after 5 epochs, the model is stopped. Best Version is saved.
            ReduceLROnPlateau: Reduce learning rate if validation error is not improving for 2 epochs.
            ModelCheckpoint: Save Copy of model each time validation MAE improves.
            
            Model training with 50 runs (epochs) or until it gets interrupted by the callbacks.
            It takes a while, get some coffe! :)
            
            Each epoch it seems to find Corrupt JPEG data up to two times.
            We could not think of a way of fixing this issue.
            The model still fits without a problem.
            """
            logger.info("Starting model training...")
            train_model(model, train_ds, val_ds, args.epochs, logger)
            
            """
            Saving final model
            """
        else:
            """
            If a best model was saved use it, 
            otherwise check if a new model was trained, 
            else exit program
            """
            model = load_model(logger)
        
        """
        evaluate the model on test data and show average age prediction error in Years
        """
        evaluate_model(model, test_ds, x_test, y_test, logger)
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

def print_section(title: str, width: int = 80) -> None:
    """Print a nicely formatted section header."""
    print("\n" + Colors.CYAN + "="*width + Colors.ENDC)
    print(Colors.CYAN + Colors.BOLD + f" {title}".center(width) + Colors.ENDC)
    print(Colors.CYAN + "="*width + Colors.ENDC)

def print_metric(label: str, value: str, unit: str = "", width: int = 50) -> None:
    """Print a metric in a formatted way."""
    print(f"  {Colors.BOLD}{label:<30}{Colors.ENDC} {Colors.GREEN}{value:>15}{Colors.ENDC} {unit}")

def evaluate_model(model: tf.keras.Model, test_ds: tf.data.Dataset, 
                  x_test: np.ndarray, y_test: np.ndarray, logger: logging.Logger) -> None:
    """Evaluate the model and display results."""
    logger.info("Evaluating model on test data...")
    
    # Evaluate model
    print("\n" + "🔍 " + "Evaluating model on test data..." + "\n")
    results = model.evaluate(test_ds, verbose=1)
    
    # results contains [loss, metric1, metric2, ...]
    # Since model is compiled with loss='mae', metrics=['mae', 'mse']
    # results[0] = loss (which is also mae)
    # results[1] = mae metric
    # results[2] = mse metric
    test_loss = float(results[0])
    test_mae = float(results[1]) if len(results) > 1 else test_loss

    # Log the results
    logger.info(f"Test Loss: {test_loss}, Test MAE: {test_mae}, Results length: {len(results)}")
    # Get predictions
    logger.info("Generating predictions...")
    age_pred = model.predict(test_ds, verbose=1).squeeze().astype(np.float32)
    
    # Calculate additional metrics
    mse = np.mean((age_pred - y_test) ** 2)
    rmse = float(np.sqrt(mse))
    bias = float(np.mean(age_pred - y_test))
    
    # Calculate error distribution
    errors = np.abs(age_pred - y_test)
    error_mean = np.mean(errors)
    error_std = np.std(errors)
    error_median = np.median(errors)
    
    # Calculate accuracy within thresholds
    within_3_years = np.mean(errors <= 3.0) * 100
    within_5_years = np.mean(errors <= 5.0) * 100
    within_10_years = np.mean(errors <= 10.0) * 100
    
    # Display results with better formatting
    print_section("📊 MODEL EVALUATION RESULTS")
    
    print(f"\n{Colors.BOLD}📈 Overall Performance Metrics:{Colors.ENDC}")
    print(Colors.CYAN + "-" * 80 + Colors.ENDC)
    print_metric("Mean Absolute Error (MAE)", f"{test_mae:.2f}", "years")
    print_metric("Mean Squared Error (MSE)", f"{mse:.2f}", "years²")
    print_metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}", "years")
    print_metric("Bias (Avg Prediction - True Age)", f"{bias:+.2f}", "years")
    
    print(f"\n{Colors.BOLD}📉 Error Distribution:{Colors.ENDC}")
    print(Colors.CYAN + "-" * 80 + Colors.ENDC)
    print_metric("Mean Error", f"{error_mean:.2f}", "years")
    print_metric("Median Error", f"{error_median:.2f}", "years")
    print_metric("Standard Deviation", f"{error_std:.2f}", "years")
    print_metric("Max Error", f"{np.max(errors):.2f}", "years")
    print_metric("Min Error", f"{np.min(errors):.2f}", "years")
    
    print(f"\n{Colors.BOLD}🎯 Accuracy Thresholds:{Colors.ENDC}")
    print(Colors.CYAN + "-" * 80 + Colors.ENDC)
    print_metric("Within ±3 years", f"{within_3_years:.1f}", "%")
    print_metric("Within ±5 years", f"{within_5_years:.1f}", "%")
    print_metric("Within ±10 years", f"{within_10_years:.1f}", "%")
    
    print(Colors.CYAN + "="*80 + Colors.ENDC)
    
    # Show sample predictions
    logger.info("Displaying sample predictions...")
    print("\n" + "🖼️  Generating visualizations..." + "\n")
    show_results(x_test, y_test, age_pred, 6)
    
    # Show detailed predictions for first 20 samples
    print_section("🔍 DETAILED PREDICTIONS (First 20 Samples)")
    print(f"\n{Colors.BOLD}{'Filename':<40} {'True Age':<12} {'Predicted':<12} {'Error':<12} {'Status':<15}{Colors.ENDC}")
    print(Colors.CYAN + "-" * 91 + Colors.ENDC)
    
    for i in range(min(20, len(x_test))):
        filename = os.path.basename(x_test[i])
        true_age = y_test[i]
        pred_age = age_pred[i]
        error = abs(pred_age - true_age)
        
        # Status based on error with colors
        if error <= 3.0:
            status = Colors.GREEN + "✓ Excellent" + Colors.ENDC
        elif error <= 5.0:
            status = Colors.BLUE + "○ Good" + Colors.ENDC
        elif error <= 10.0:
            status = Colors.YELLOW + "△ Fair" + Colors.ENDC
        else:
            status = Colors.RED + "✗ Poor" + Colors.ENDC
        
        print(f"{filename:<40} {true_age:>10.1f}  {Colors.CYAN}{pred_age:>10.1f}{Colors.ENDC}  {Colors.YELLOW}{error:>10.1f}{Colors.ENDC}  {status}")
    
    print(Colors.CYAN + "="*91 + Colors.ENDC)
    print(f"\n{Colors.BOLD}Status Legend:{Colors.ENDC}")
    print(f"  {Colors.GREEN}✓ Excellent{Colors.ENDC} (≤3 yrs)  {Colors.BLUE}○ Good{Colors.ENDC} (≤5 yrs)  {Colors.YELLOW}△ Fair{Colors.ENDC} (≤10 yrs)  {Colors.RED}✗ Poor{Colors.ENDC} (>10 yrs)")
    print(f"\n{Colors.GREEN}✨ Evaluation complete!{Colors.ENDC}\n")

if __name__ == "__main__":
    main()