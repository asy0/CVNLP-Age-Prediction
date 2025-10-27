# Age Prediction from Facial Images

A deep learning project for predicting age from facial images using Convolutional Neural Networks (CNN) and the UTKFace dataset.

## 🚀 Features

- **Advanced CNN Architecture**: Improved model with batch normalization, better regularization, and global average pooling
- **Data Augmentation**: Comprehensive augmentation pipeline for better generalization
- **Stratified Splitting**: Ensures balanced age distribution across train/validation/test sets
- **Professional Logging**: Comprehensive logging system with file and console output
- **Error Handling**: Robust error handling and validation throughout the pipeline
- **Performance Optimized**: Efficient data loading with TensorFlow's tf.data API

## 📁 Project Structure

```
age-prediction/
├── src/
│   ├── config.py              # Configuration settings
│   ├── main.py                # Main execution script
│   ├── data_loading/
│   │   └── get_image_paths.py # Image path loading utilities
│   ├── model/
│   │   └── build_model.py     # CNN model architecture
│   ├── preprocessing/
│   │   ├── build_datasets.py  # Dataset creation and augmentation
│   │   └── parse_age.py       # Age label parsing from filenames
│   ├── validation/
│   │   └── filenames_valid.py # Dataset validation utilities
│   └── visualization/
│       └── visualize_samples.py # Visualization functions
├── artifacts/                 # Saved models and checkpoints
├── in_the_wild_faces/         # UTKFace dataset
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd age-prediction
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv age-prediction_env
   source age-prediction_env/bin/activate  # On Windows: age-prediction_env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

### Basic Usage

**Evaluate with existing model**:
```bash
python src/main.py
```

**Train a new model**:
```bash
python src/main.py --train
```

**Train with custom epochs**:
```bash
python src/main.py --train --epochs 50
```

**Validate dataset only**:
```bash
python src/main.py --validate-only
```

### Command Line Options

- `--train`: Train a new model (default: use existing model)
- `--epochs N`: Number of training epochs (default: 50)
- `--validate-only`: Only validate dataset filenames and exit

## 🏗️ Model Architecture

The improved CNN architecture includes:

- **4 Convolutional Blocks**: Each with batch normalization and dropout
- **Global Average Pooling**: Reduces overfitting compared to flattening
- **Progressive Feature Maps**: 32 → 64 → 128 → 256 filters
- **Dense Layers**: 512 → 256 → 128 → 1 neurons with batch normalization
- **Regularization**: Dropout layers and L2 regularization

## 📊 Performance Improvements

### Data Loading Optimizations
- **Parallel Processing**: `tf.data.AUTOTUNE` for optimal CPU utilization
- **Prefetching**: Reduces training time by overlapping data loading and model execution
- **Memory Efficiency**: Optimized batch processing and data pipeline

### Model Improvements
- **Batch Normalization**: Stabilizes training and improves convergence
- **Better Regularization**: Strategic dropout placement and global average pooling
- **Enhanced Augmentation**: Additional color space augmentations (saturation, hue)

### Code Quality
- **Type Hints**: Full type annotation for better code maintainability
- **Error Handling**: Comprehensive error handling and validation
- **Logging**: Professional logging system with configurable levels
- **Documentation**: Detailed docstrings and inline comments

## 📈 Expected Results

With the improved architecture and training pipeline, you can expect:
- **Better Generalization**: Reduced overfitting through improved regularization
- **Faster Training**: Optimized data loading and model architecture
- **More Stable Training**: Batch normalization and better optimizer settings
- **Professional Output**: Comprehensive logging and error reporting

## 🔧 Configuration

All configuration parameters are centralized in `src/config.py`:

- Model hyperparameters (learning rate, batch size, etc.)
- Data augmentation settings
- Training configuration (epochs, callbacks, etc.)
- Logging configuration

## 📝 Logging

The application creates detailed logs in:
- **Console**: Real-time progress updates
- **File**: `age_prediction.log` in the project root

## 📊 Dataset: UTKFace

- **Size**: Over 20,000 face images in the wild (different backgrounds)
- **Age Range**: 0-116 years
- **Attributes**: Age, Gender, Ethnicity, Date & Time of Photograph
- **Varied Settings**: poses, facial expressions, illumination, occlusion, resolution, etc.


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Acknowledgments

- UTKFace dataset for providing the facial images
- TensorFlow team for the excellent deep learning framework
