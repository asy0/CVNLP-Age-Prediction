import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random

from pathlib import Path

def show_samples(image_paths, n):
    """
    Display random sample images from the dataset.
    
    Args:
        image_paths: List of image file paths
        n: Number of samples to display
    """
    # Limit to max 6 samples for better layout
    num_samples = min(n, 6)
    sample_paths = random.sample(image_paths, num_samples)
    images = []
    for path in sample_paths:
        img_raw = tf.io.read_file(str(path))
        img = tf.image.decode_jpeg(img_raw, channels=3)
        # Resize images for better fit
        img = tf.image.resize(img, (128, 128))
        images.append((path, img))

    # Create compact 2x3 grid layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i in range(num_samples):
        axes[i].imshow(images[i][1].numpy().astype("uint8"))
        axes[i].axis("off")
        # Simplified title with just the sample number
        axes[i].set_title(f"Sample {i+1}", fontsize=12, fontweight='bold')
    
    # Hide unused subplots
    for i in range(num_samples, 6):
        axes[i].axis('off')
    
    # layout with space for title
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Main title
    fig.suptitle("Random Sample Images from Dataset", 
                 fontsize=18, fontweight='bold', y=0.97,
                 color='#540846', family='Arial')
    
    plt.show()

def show_results(x_test, y_test, y_pred, n):
    """
    Display prediction results with images, true ages, predicted ages, and errors.
    
    Args:
        x_test: Test image paths
        y_test: True age labels
        y_pred: Predicted age labels
        n: Number of samples to display (max 6 for 2x3 grid)
    """
    # Randomly select n samples from test set (max 6)
    num_samples = min(n, len(x_test), 6)
    random_indices = random.sample(range(len(x_test)), num_samples)
    
    # Extract selected samples
    selected_x = [x_test[i] for i in random_indices]
    selected_y_test = [y_test[i] for i in random_indices]
    selected_y_pred = [y_pred[i] for i in random_indices]
    
    # Calculate errors for color coding
    errors = np.abs(np.array(selected_y_test) - np.array(selected_y_pred))
    
    # Create figure with 2x3 grid for 6 samples
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    #  background
    fig.patch.set_facecolor('#FFFFFF')
    
    #  color scheme based on error
    def get_color(error):
        if error <= 3.0:
            return '#10B981'  # Modern Green
        elif error <= 5.0:
            return '#3B82F6'  # Modern Blue
        elif error <= 10.0:
            return '#F59E0B'  # Modern Orange
        else:
            return '#EF4444'  # Modern Red
    
    for i in range(num_samples):
        # Load and display image with better quality
        img_raw = tf.io.read_file(selected_x[i])
        img = tf.image.decode_jpeg(img_raw, channels=3)
        img = tf.image.resize(img, (110, 110))
        
        # Add subtle border to images
        axes[i].imshow(img.numpy().astype("uint8"))
        for spine in axes[i].spines.values():
            spine.set_visible(True)
            spine.set_color('#E5E7EB')
            spine.set_linewidth(2)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        
        # Calculate error and determine status
        error = errors[i]
        status_color = get_color(error)
        
        if error <= 3.0:
            status = "✓ Excellent"
        elif error <= 5.0:
            status = "○ Good"
        elif error <= 10.0:
            status = "△ Fair"
        else:
            status = "✗ Poor"
        
        # label without filename clutter
        title = (f"Sample {i+1}\n"
                f"True: {selected_y_test[i]:.0f}y | Pred: {selected_y_pred[i]:.0f}y\n"
                f"Error: {error:.1f}y → {status}")
        
        axes[i].set_title(title, fontsize=11, pad=10, color=status_color, 
                         fontweight='bold', family='Arial', y=1.08)
    
    # Hide unused subplots
    for i in range(num_samples, 6):
        axes[i].axis('off')
    
    # layout with proper spacing
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    

    
    # Main title with  styling
    fig.suptitle("Age Prediction Results - Test Samples", 
                 fontsize=20, fontweight='bold', y=0.98,
                 color='#111827', family='Arial')



    # legend alternative styling
    legend_text = "✓ Excellent (≤3yrs)  |  ○ Good (≤5yrs)  |  △ Fair (≤10yrs)  |  ✗ Poor (>10yrs)"
    # Add subtle performance summary
    avg_error = np.mean(errors)
    summary_text = f"Average Error: {avg_error:.2f} years | Total Samples: {len(errors)}"
    fig.text(0.5, 0.97, summary_text, ha='center', fontsize=11,
             style='italic', color='#6B7280', family='Arial')
    
    fig.text(0.5, 0.03, legend_text, ha='center', fontsize=12, 
             bbox=dict(boxstyle='round,pad=1.5', facecolor='#F9FAFB', 
                      edgecolor='#701667', linewidth=2, alpha=0.95),
             family='Arial', weight='bold')
    
    plt.show()