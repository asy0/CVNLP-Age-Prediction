import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import Button
from matplotlib.lines import Line2D
import numpy as np
import random

from pathlib import Path


def use_modern_theme():
    """Apply a clean, modern Matplotlib theme and return common colors ."""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        # Fallback for older Matplotlib versions
        plt.style.use('seaborn-whitegrid')

    modern_colors = {
        'figure_bg': '#F5F7FB',  # soft slate
        'axes_bg': '#FFFFFF',  # pure white panels
        'accent': '#701667',  # plum accent
        'muted_text': '#6B7280',
        'strong_text': '#111827',
        'light_edge': '#E5E7EB',
    }

    mpl.rcParams.update({
        'figure.facecolor': modern_colors['figure_bg'],
        'savefig.facecolor': modern_colors['figure_bg'],
        'axes.facecolor': modern_colors['axes_bg'],
        'axes.edgecolor': modern_colors['light_edge'],
        'axes.titleweight': 'bold',
        'axes.titlesize': 12,
        'font.family': 'DejaVu Sans',
        'grid.color': '#D1D5DB',
        'grid.alpha': 0.4,
        'axes.grid': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'toolbar': 'None',  # hide default toolbar for a cleaner look
    })

    return modern_colors


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

    # Theme and compact 2x3 grid layout
    colors = use_modern_theme()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor=colors['figure_bg'])
    axes = axes.flatten()
    if hasattr(fig, 'canvas') and hasattr(fig.canvas, 'manager') and hasattr(fig.canvas.manager, 'set_window_title'):
        fig.canvas.manager.set_window_title('Age Prediction • Samples')

    for i in range(num_samples):
        axes[i].imshow(images[i][1].numpy().astype("uint8"))
        axes[i].set_facecolor(colors['axes_bg'])
        for side in ['left', 'bottom']:
            axes[i].spines[side].set_visible(True)
            axes[i].spines[side].set_color(colors['light_edge'])
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].grid(False)
        # Simplified title with just the sample number
        axes[i].set_title(f"Sample {i + 1}", fontsize=12, fontweight='bold')

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

    # Create figure with 2x3 grid for 6 samples and modern theme
    colors = use_modern_theme()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor=colors['figure_bg'])
    axes = axes.flatten()
    if hasattr(fig, 'canvas') and hasattr(fig.canvas, 'manager') and hasattr(fig.canvas.manager, 'set_window_title'):
        fig.canvas.manager.set_window_title('Age Prediction • Results')

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
        axes[i].set_facecolor(colors['axes_bg'])
        for spine in axes[i].spines.values():
            spine.set_visible(True)
            spine.set_color(colors['light_edge'])
            spine.set_linewidth(1.8)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].grid(False)

        # Calculate error and determine status
        error = errors[i]
        status_color = get_color(error)

        if error <= 3.0:
            status = "Excellent"
        elif error <= 5.0:
            status = "Good"
        elif error <= 10.0:
            status = "Fair"
        else:
            status = "Poor"

        # label without filename clutter
        title = (f"Sample {i + 1}\n"
                 f"True: {selected_y_test[i]:.0f}y | Pred: {selected_y_pred[i]:.0f}y\n"
                 f"Error: {error:.1f}y → {status}")

        axes[i].set_title(title, fontsize=11, pad=8, color=status_color,
                          fontweight='bold', family='Arial', y=1.04)

    # Hide unused subplots
    for i in range(num_samples, 6):
        axes[i].axis('off')

    # layout with proper spacing
    plt.tight_layout(rect=[0, 0.08, 1, 0.90])

    # Main title with modern styling (accent color + underline)
    title_text = fig.suptitle("Age Prediction Results - Test Samples",
                              fontsize=20, fontweight='bold', y=0.99,
                              color=colors['accent'], family='Arial')
    try:
        title_text.set_underline(True)
    except Exception:
        pass

    # legend alternative styling
    legend_text = "✓ Excellent (≤3yrs)  |  ○ Good (≤5yrs)  |  △ Fair (≤10yrs)  |  ✗ Poor (>10yrs)"
    # Add subtle performance summary
    avg_error = np.mean(errors)
    summary_text = f"Average Error: {avg_error:.2f} years | Total Samples: {len(errors)}"
    fig.text(0.5, 0.905, summary_text, ha='center', fontsize=11,
             style='italic', color=colors['muted_text'], family='Arial')

    # Colored legend at the same bottom position (no overlay panel)
    base_y = 0.034
    separator_color = colors['muted_text']
    parts = [
        ("Excellent (≤3yrs)", '#10B981'),
        ("Good (≤5yrs)", '#3B82F6'),
        ("Fair (≤10yrs)", '#F59E0B'),
        ("Poor (>10yrs)", '#EF4444'),
    ]
    xs = [0.36, 0.52, 0.68, 0.84]
    legend_artists = []
    # marker positions slightly to the left of labels
    marker_xs = [x - 0.065 for x in xs]
    marker_styles = ['s', 'o', '^', 'x']
    for (label, color), mx, mstyle in zip(parts, marker_xs, marker_styles):
        marker = Line2D([mx], [base_y], marker=mstyle, linestyle='None',
                        markersize=10, markeredgewidth=2.2,
                        markerfacecolor='none' if mstyle != 's' else color,
                        markeredgecolor=color, transform=fig.transFigure)
        fig.add_artist(marker)
        legend_artists.append(marker)
    for (label, color), x in zip(parts, xs):
        artist = fig.text(x, base_y, label, ha='center', va='center', fontsize=12,
                          color=color, family='Arial', weight='bold')
        legend_artists.append(artist)
    # separators between parts
    for x in [0.42, 0.58, 0.74]:
        sep = fig.text(x, base_y, '|', ha='center', va='center', fontsize=12,
                       color=separator_color, family='Arial', weight='bold')
        legend_artists.append(sep)

    # --- Interactive toggle (button + keyboard 'L') with fade animation ---
    is_visible = {'value': True}
    toolbar_state = {'obj': None, 'shown': False}

    def set_legend_alpha(alpha: float) -> None:
        for a in legend_artists:
            a.set_alpha(alpha)

    def animate_to(target_alpha: float) -> None:
        current_alpha = legend_artists[0].get_alpha()
        if current_alpha is None:
            current_alpha = 1.0 if is_visible['value'] else 0.0
        steps = np.linspace(current_alpha, target_alpha, 8)
        idx = {'i': 0}

        def step(**_):
            # Guard against timer race conditions
            if idx['i'] >= len(steps):
                return
            set_legend_alpha(float(steps[idx['i']]))
            fig.canvas.draw_idle()
            idx['i'] += 1
            if idx['i'] < len(steps):
                t = fig.canvas.new_timer(interval=18)
                t.add_callback(step)
                t.start()

        step()

    def toggle(_event=None):
        is_visible['value'] = not is_visible['value']
        animate_to(1.0 if is_visible['value'] else 0.0)
        # Show/hide native toolbar at the bottom (standard Matplotlib toolbar)
        try:
            from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
            import tkinter as tk  # type: ignore
            manager = plt.get_current_fig_manager()
            canvas = manager.canvas
            root = manager.window
            if toolbar_state['obj'] is None:
                # Create but do not auto-pack
                tb = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
                tb.update()
                toolbar_state['obj'] = tb
            tb = toolbar_state['obj']
            if tb is not None:
                if is_visible['value']:
                    tb.pack(side=tk.BOTTOM, fill=tk.X)
                    toolbar_state['shown'] = True
                else:
                    if toolbar_state['shown']:
                        tb.pack_forget()
                        toolbar_state['shown'] = False
        except Exception:
            # Non-Tk backends: ignore toolbar toggle
            pass

    # Button in bottom-left
    btn_ax = fig.add_axes([0.015, 0.005, 0.06, 0.04])
    toggle_btn = Button(btn_ax, 'Legend', color='#EDE9F7', hovercolor='#DDD6FE')
    toggle_btn.label.set_color(colors['accent'])
    toggle_btn.on_clicked(toggle)

    # Keyboard shortcut 'L' to toggle
    def on_key(event):
        if event.key and event.key.lower() == 'l':
            toggle()

    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()