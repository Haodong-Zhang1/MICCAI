# kits23_segmentation/utils/visualization.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import torch
import torch.nn.functional as F
from skimage import measure
import nibabel as nib
from mpl_toolkits.axes_grid1 import make_axes_locatable


def create_kidney_tumor_colormap():
    """Create custom colormap for kidney tumor visualization."""
    # Colors: Background (black), Kidney (red), Tumor (yellow)
    colors = [(0, 0, 0), (0.7, 0.3, 0.3), (0.9, 0.9, 0.3)]
    return LinearSegmentedColormap.from_list('kidney_tumor_cmap', colors, N=3)


def visualize_slice(image, mask=None, pred=None, axis=0, slice_idx=None,
                    fig=None, ax=None, show_colorbar=True, title=None):
    """
    Visualize a 2D slice from a 3D volume with optional segmentation overlay.

    Args:
        image: 3D image volume
        mask: Ground truth segmentation (optional)
        pred: Predicted segmentation (optional)
        axis: Axis to slice (0=sagittal, 1=coronal, 2=axial)
        slice_idx: Index of slice to visualize (None for middle slice)
        fig: Matplotlib figure (optional)
        ax: Matplotlib axis (optional)
        show_colorbar: Whether to show colorbar
        title: Plot title

    Returns:
        Matplotlib figure and axis
    """
    # Create figure if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Transpose image based on slice axis
    if axis == 0:  # Sagittal
        if slice_idx is None:
            slice_idx = image.shape[0] // 2
        image_slice = image[slice_idx, :, :]
        mask_slice = None if mask is None else mask[slice_idx, :, :]
        pred_slice = None if pred is None else pred[slice_idx, :, :]
    elif axis == 1:  # Coronal
        if slice_idx is None:
            slice_idx = image.shape[1] // 2
        image_slice = image[:, slice_idx, :]
        mask_slice = None if mask is None else mask[:, slice_idx, :]
        pred_slice = None if pred is None else pred[:, slice_idx, :]
    else:  # Axial
        if slice_idx is None:
            slice_idx = image.shape[2] // 2
        image_slice = image[:, :, slice_idx]
        mask_slice = None if mask is None else mask[:, :, slice_idx]
        pred_slice = None if pred is None else pred[:, :, slice_idx]

    # Plot grayscale image
    img_plot = ax.imshow(image_slice, cmap='gray')

    # Add colorbar for the image
    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(img_plot, cax=cax)

    # Create segmentation colormap
    cmap = create_kidney_tumor_colormap()

    # Overlay ground truth mask if provided
    if mask_slice is not None:
        # Create overlay with transparency
        mask_rgba = cmap(mask_slice.astype(int))
        mask_rgba[..., 3] = np.where(mask_slice > 0, 0.5, 0)
        ax.imshow(mask_rgba)

    # Overlay predicted mask if provided
    if pred_slice is not None:
        # Create contours for prediction
        contours = measure.find_contours(pred_slice == 1, 0.5)  # Kidney contours
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], 'b-', linewidth=1.5)

        contours = measure.find_contours(pred_slice == 2, 0.5)  # Tumor contours
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=1.5)

    # Set title
    slice_types = ['Sagittal', 'Coronal', 'Axial']
    default_title = f"{slice_types[axis]} Slice (Index: {slice_idx})"
    ax.set_title(title or default_title)

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax


def visualize_multi_slice(image, mask=None, pred=None, axis=2, num_slices=5,
                          spacing=None, figsize=(20, 16)):
    """
    Visualize multiple slices from a 3D volume with optional overlays.

    Args:
        image: 3D image volume
        mask: Ground truth segmentation (optional)
        pred: Predicted segmentation (optional)
        axis: Axis to slice (0=sagittal, 1=coronal, 2=axial)
        num_slices: Number of slices to visualize
        spacing: Voxel spacing for physical dimensions (optional)
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    max_idx = image.shape[axis]
    step = max(1, max_idx // (num_slices + 1))
    slice_indices = list(range(step, max_idx - step + 1, step))[:num_slices]

    fig, axes = plt.subplots(1, num_slices, figsize=figsize)
    if num_slices == 1:
        axes = [axes]

    for i, slice_idx in enumerate(slice_indices):
        visualize_slice(image, mask, pred, axis, slice_idx, fig, axes[i],
                        show_colorbar=(i == num_slices - 1))

        # Add physical dimensions if spacing is provided
        if spacing is not None:
            if axis == 0:
                dim_text = f"Pos: {slice_idx * spacing[0]:.1f} mm"
            elif axis == 1:
                dim_text = f"Pos: {slice_idx * spacing[1]:.1f} mm"
            else:
                dim_text = f"Pos: {slice_idx * spacing[2]:.1f} mm"
            axes[i].set_xlabel(dim_text)

    # Overall title
    slice_types = ['Sagittal', 'Coronal', 'Axial']
    title = f"{slice_types[axis]} Slices"

    if mask is not None and pred is not None:
        title += " (Blue: Kidney Prediction, Red: Tumor Prediction)"
    elif mask is not None:
        title += " with Ground Truth"
    elif pred is not None:
        title += " with Predictions"

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    return fig


def visualize_3d_segmentation(segmentation, threshold=0.5, class_idx=None):
    """
    Create a 3D visualization of segmentation surface using marching cubes.

    Args:
        segmentation: 3D segmentation volume
        threshold: Threshold for surface extraction
        class_idx: Class index to visualize (None for all)

    Returns:
        Matplotlib figure
    """
    try:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        print("3D plotting requires mplot3d. Please install with: pip install matplotlib")
        return None

    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Define colors for different classes
    colors = ['red', 'gold']

    # Extract and visualize surfaces for selected classes
    classes_to_plot = [1, 2] if class_idx is None else [class_idx]

    for i, cls in enumerate(classes_to_plot):
        # Extract binary mask for the current class
        binary_segmentation = (segmentation == cls)

        # Skip if no voxels for this class
        if not np.any(binary_segmentation):
            continue

        # Downsampling for performance if needed
        spacing = np.array([3, 3, 3])  # Adjust based on volume size
        if np.prod(binary_segmentation.shape) > 64 ** 3:
            from scipy.ndimage import zoom
            zoom_factor = 64 / np.max(binary_segmentation.shape)
            binary_segmentation = zoom(binary_segmentation.astype(float), zoom_factor, order=0)

        # Extract surface mesh using marching cubes
        try:
            verts, faces, _, _ = measure.marching_cubes(binary_segmentation, threshold)

            # Create mesh
            mesh = Poly3DCollection(verts[faces], alpha=0.7)

            # Set colors and properties
            color = colors[i % len(colors)]
            mesh.set_facecolor(color)
            mesh.set_edgecolor('k')
            mesh.set_linewidth(0.1)

            # Add to plot
            ax.add_collection3d(mesh)

            # Set plot limits
            ax.set_xlim(0, binary_segmentation.shape[0])
            ax.set_ylim(0, binary_segmentation.shape[1])
            ax.set_zlim(0, binary_segmentation.shape[2])

        except Exception as e:
            print(f"Error creating 3D visualization: {e}")

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if class_idx is None:
        ax.set_title('3D Visualization of Kidney (Red) and Tumor (Yellow)')
    else:
        class_names = {1: 'Kidney', 2: 'Tumor'}
        ax.set_title(f'3D Visualization of {class_names.get(class_idx, f"Class {class_idx}")}')

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    return fig


def visualize_prediction_comparison(image, gt_mask, pred_mask, case_id, axis=2,
                                    num_slices=3, output_dir=None):
    """
    Create visualization comparing ground truth and prediction for a case.

    Args:
        image: 3D image volume
        gt_mask: Ground truth segmentation
        pred_mask: Predicted segmentation
        case_id: Case identifier
        axis: Axis to slice (0=sagittal, 1=coronal, 2=axial)
        num_slices: Number of slices to visualize
        output_dir: Directory to save visualization (optional)

    Returns:
        Matplotlib figure
    """
    # Find slices with most segmentation content
    if gt_mask is not None:
        content_per_slice = []
        for i in range(gt_mask.shape[axis]):
            if axis == 0:
                slice_content = np.sum(gt_mask[i, :, :] > 0)
            elif axis == 1:
                slice_content = np.sum(gt_mask[:, i, :] > 0)
            else:
                slice_content = np.sum(gt_mask[:, :, i] > 0)
            content_per_slice.append(slice_content)

        # Get indices of slices with most content
        slice_indices = np.argsort(content_per_slice)[-num_slices:]

        # Sort indices
        slice_indices = sorted(slice_indices)
    else:
        # Default to evenly spaced slices
        max_idx = image.shape[axis]
        step = max(1, max_idx // (num_slices + 1))
        slice_indices = list(range(step, max_idx - step + 1, step))[:num_slices]

    # Create figure with three rows: image, ground truth, prediction
    fig, axes = plt.subplots(3, num_slices, figsize=(5 * num_slices, 15))

    # Plot images
    for i, slice_idx in enumerate(slice_indices):
        # Original image
        visualize_slice(image, None, None, axis, slice_idx, fig, axes[0, i],
                        show_colorbar=(i == num_slices - 1), title=f"Slice {slice_idx}")

        # Ground truth overlay
        if gt_mask is not None:
            visualize_slice(image, gt_mask, None, axis, slice_idx, fig, axes[1, i],
                            show_colorbar=False, title="Ground Truth")

        # Prediction overlay
        if pred_mask is not None:
            visualize_slice(image, None, pred_mask, axis, slice_idx, fig, axes[2, i],
                            show_colorbar=False, title="Prediction")

        # Row titles
    axes[0, 0].set_ylabel("Original Image", fontsize=14)
    axes[1, 0].set_ylabel("Ground Truth", fontsize=14)
    axes[2, 0].set_ylabel("Prediction", fontsize=14)

    # Main title
    slice_types = ['Sagittal', 'Coronal', 'Axial']
    fig.suptitle(f"Case {case_id} - {slice_types[axis]} View Comparison", fontsize=16)

    plt.tight_layout()

    # Save figure if output directory provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{case_id}_{slice_types[axis].lower()}_comparison.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")

    return fig


def plot_training_history(history, output_dir=None):
    """
    Plot training and validation metrics history.

    Args:
        history: Dictionary containing training history
        output_dir: Directory to save plots (optional)

    Returns:
        Dictionary of matplotlib figures
    """
    # Create output directory if provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # Create figures
    figures = {}

    # Plot training and validation loss
    if 'train_loss' in history and 'val_loss' in history:
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(history['train_loss']) + 1)

        ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')

        ax.set_title('Training and Validation Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

        figures['loss'] = fig

        if output_dir is not None:
            fig.savefig(os.path.join(output_dir, 'loss_history.png'), dpi=200, bbox_inches='tight')

    # Plot Dice scores
    dice_metrics = ['train_dice_kidney', 'val_dice_kidney', 'train_dice_tumor', 'val_dice_tumor']
    if all(metric in history for metric in dice_metrics):
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(history['train_dice_kidney']) + 1)

        ax.plot(epochs, history['train_dice_kidney'], 'b-', label='Train Kidney Dice')
        ax.plot(epochs, history['val_dice_kidney'], 'b--', label='Val Kidney Dice')
        ax.plot(epochs, history['train_dice_tumor'], 'r-', label='Train Tumor Dice')
        ax.plot(epochs, history['val_dice_tumor'], 'r--', label='Val Tumor Dice')

        ax.set_title('Training and Validation Dice Coefficients')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Dice Coefficient')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

        figures['dice'] = fig

        if output_dir is not None:
            fig.savefig(os.path.join(output_dir, 'dice_history.png'), dpi=200, bbox_inches='tight')

    # Plot learning rate if available
    if 'learning_rate' in history:
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(history['learning_rate']) + 1)

        ax.plot(epochs, history['learning_rate'], 'g-')
        ax.set_title('Learning Rate Schedule')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Learning Rate')
        ax.set_yscale('log')
        ax.grid(True, linestyle='--', alpha=0.7)

        figures['lr'] = fig

        if output_dir is not None:
            fig.savefig(os.path.join(output_dir, 'lr_history.png'), dpi=200, bbox_inches='tight')

    return figures


def visualize_data_distribution(metadata_df, output_dir=None):
    """
    Create visualizations of dataset statistics and distributions.

    Args:
        metadata_df: DataFrame containing dataset metadata
        output_dir: Directory to save plots (optional)

    Returns:
        Dictionary of matplotlib figures
    """
    # Create output directory if provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    figures = {}

    # Ensure required columns are present
    required_cols = ['case_id', 'spacing', 'image_shape', 'kidney_volume', 'tumor_volume']
    if not all(col in metadata_df.columns for col in required_cols):
        print("Metadata DataFrame is missing required columns.")
        return figures

    # Plot tumor size distribution
    fig, ax = plt.subplots(figsize=(10, 6))

    metadata_df['tumor_volume_cc'] = metadata_df['tumor_volume'] * \
                                     metadata_df['spacing'].apply(lambda x: np.prod(x)) / 1000

    ax.hist(metadata_df['tumor_volume_cc'], bins=30, alpha=0.7, color='royalblue')
    ax.set_title('Tumor Volume Distribution')
    ax.set_xlabel('Tumor Volume (cc)')
    ax.set_ylabel('Number of Cases')
    ax.grid(True, linestyle='--', alpha=0.7)

    # Log scale for better visualization of small tumors
    ax.set_xscale('log')

    figures['tumor_volume'] = fig

    if output_dir is not None:
        fig.savefig(os.path.join(output_dir, 'tumor_volume_distribution.png'),
                    dpi=200, bbox_inches='tight')

    # Plot tumor-to-kidney ratio
    fig, ax = plt.subplots(figsize=(10, 6))

    metadata_df['tumor_kidney_ratio'] = metadata_df['tumor_volume'] / metadata_df['kidney_volume']

    ax.hist(metadata_df['tumor_kidney_ratio'], bins=30, alpha=0.7, color='darkorange')
    ax.set_title('Tumor to Kidney Volume Ratio Distribution')
    ax.set_xlabel('Tumor/Kidney Volume Ratio')
    ax.set_ylabel('Number of Cases')
    ax.grid(True, linestyle='--', alpha=0.7)

    figures['tumor_kidney_ratio'] = fig

    if output_dir is not None:
        fig.savefig(os.path.join(output_dir, 'tumor_kidney_ratio.png'),
                    dpi=200, bbox_inches='tight')

    # Plot image shape distribution
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract image shapes as tuples
    shapes = metadata_df['image_shape'].apply(lambda x: tuple(x) if isinstance(x, list) else x)

    # Count occurrences of each shape
    shape_counts = shapes.value_counts()

    # Bar plot of shape counts
    ax.bar(range(len(shape_counts)), shape_counts.values, alpha=0.7, color='seagreen')
    ax.set_xticks(range(len(shape_counts)))
    ax.set_xticklabels([str(s) for s in shape_counts.index], rotation=45)
    ax.set_title('Image Shape Distribution')
    ax.set_xlabel('Image Shape (D, H, W)')
    ax.set_ylabel('Number of Cases')
    ax.grid(True, linestyle='--', alpha=0.7)

    figures['image_shapes'] = fig

    if output_dir is not None:
        fig.savefig(os.path.join(output_dir, 'image_shape_distribution.png'),
                    dpi=200, bbox_inches='tight')

    # Plot voxel spacing distribution
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract spacings for visualization
    spacings = np.array([s for s in metadata_df['spacing'].values])

    # Boxplot of spacings in each dimension
    ax.boxplot([spacings[:, 0], spacings[:, 1], spacings[:, 2]])
    ax.set_xticklabels(['X', 'Y', 'Z'])
    ax.set_title('Voxel Spacing Distribution')
    ax.set_ylabel('Spacing (mm)')
    ax.grid(True, linestyle='--', alpha=0.7)

    figures['voxel_spacing'] = fig

    if output_dir is not None:
        fig.savefig(os.path.join(output_dir, 'voxel_spacing_distribution.png'),
                    dpi=200, bbox_inches='tight')

    return figures


def save_nifti_segmentation(segmentation, affine, output_path):
    """
    Save segmentation as NIfTI file.

    Args:
        segmentation: 3D segmentation array
        affine: Affine transformation matrix
        output_path: Output file path
    """
    # Create NIfTI image
    nifti_img = nib.Nifti1Image(segmentation.astype(np.int16), affine)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save file
    nib.save(nifti_img, output_path)


def visualize_performance_metrics(metrics_dict, output_dir):
    """
    可视化性能指标
    
    Args:
        metrics_dict: 包含各种性能指标的字典
        output_dir: 输出目录路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 推理时间分布
    plt.figure(figsize=(10, 6))
    plt.hist(metrics_dict['inference_times'], bins=30, alpha=0.7)
    plt.title('推理时间分布')
    plt.xlabel('时间 (秒)')
    plt.ylabel('频次')
    plt.savefig(os.path.join(output_dir, 'inference_time_distribution.png'))
    plt.close()
    
    # 2. 内存使用趋势
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_dict['memory_usage'])
    plt.title('训练过程中内存使用趋势')
    plt.xlabel('训练步数')
    plt.ylabel('内存使用 (GB)')
    plt.savefig(os.path.join(output_dir, 'memory_usage_trend.png'))
    plt.close()
    
    # 3. 准确性和效率的权衡
    plt.figure(figsize=(10, 6))
    plt.scatter(metrics_dict['inference_times'], 
                metrics_dict['dice_scores'],
                alpha=0.5)
    plt.title('准确性和效率权衡')
    plt.xlabel('推理时间 (秒)')
    plt.ylabel('Dice分数')
    plt.savefig(os.path.join(output_dir, 'accuracy_efficiency_tradeoff.png'))
    plt.close()
    
    # 4. 模型复杂度分析
    plt.figure(figsize=(10, 6))
    plt.bar(['总参数量', '可训练参数量'], 
            [metrics_dict['total_params'], 
             metrics_dict['trainable_params']])
    plt.title('模型参数量分析')
    plt.ylabel('参数量')
    plt.savefig(os.path.join(output_dir, 'model_complexity.png'))
    plt.close()


def create_performance_report(metrics_dict, output_dir):
    """
    生成性能报告
    
    Args:
        metrics_dict: 包含各种性能指标的字典
        output_dir: 输出目录路径
    """
    report_path = os.path.join(output_dir, 'performance_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# 模型性能报告\n\n")
        
        # 准确性指标
        f.write("## 准确性指标\n")
        f.write(f"- 肾脏Dice分数: {metrics_dict['dice_kidney']:.4f}\n")
        f.write(f"- 肿瘤Dice分数: {metrics_dict['dice_tumor']:.4f}\n")
        f.write(f"- 平均Dice分数: {metrics_dict['mean_dice']:.4f}\n\n")
        
        # 效率指标
        f.write("## 效率指标\n")
        f.write(f"- 平均推理时间: {metrics_dict['mean_inference_time']:.4f}秒\n")
        f.write(f"- 最大内存使用: {metrics_dict['max_memory_usage']:.2f}GB\n")
        f.write(f"- 模型大小: {metrics_dict['model_size_mb']:.2f}MB\n\n")
        
        # 复杂度指标
        f.write("## 模型复杂度\n")
        f.write(f"- 总参数量: {metrics_dict['total_params']:,}\n")
        f.write(f"- 可训练参数量: {metrics_dict['trainable_params']:,}\n")