"""
Utility functions for preprocessing pipeline
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def setup_logging(log_dir: str = "logs", level: str = "INFO"):
    """Setup logging configuration"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"preprocessing_{timestamp}.log"
    
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Logging initialized. Log file: {log_file}")


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load image from file (supports Korean/Unicode paths)
    한글 경로를 지원하는 이미지 로딩
    """
    try:
        # Use numpy to read file as bytes, then decode with OpenCV
        # This workaround supports Korean and other Unicode characters in paths
        with open(image_path, 'rb') as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


def save_image(image: np.ndarray, output_path: str, compression: int = 9):
    """
    Save image to file (supports Korean/Unicode paths)
    한글 경로를 지원하는 이미지 저장
    """
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Encode image to bytes, then write to file
        # This workaround supports Korean and other Unicode characters in paths
        if output_path.endswith('.png'):
            success, encoded_image = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, compression])
        else:
            ext = Path(output_path).suffix
            success, encoded_image = cv2.imencode(ext, image)

        if success:
            with open(output_path, 'wb') as f:
                f.write(encoded_image.tobytes())
            logger.debug(f"Saved image: {output_path}")
        else:
            logger.error(f"Failed to encode image: {output_path}")

    except Exception as e:
        logger.error(f"Error saving image {output_path}: {e}")


def visualize_preprocessing_steps(intermediate_images: Dict[str, np.ndarray], 
                                  output_path: str):
    """Visualize all preprocessing steps"""
    num_steps = len(intermediate_images)
    fig, axes = plt.subplots(2, (num_steps + 1) // 2, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, (step_name, image) in enumerate(intermediate_images.items()):
        axes[idx].imshow(image, cmap='gray')
        axes[idx].set_title(step_name.replace('_', ' ').title())
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(len(intermediate_images), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved preprocessing visualization: {output_path}")


def visualize_annotations(image: np.ndarray, 
                         annotations: List[Dict],
                         output_path: str,
                         format_type: str = 'yolo'):
    """Visualize image with annotations"""
    # Convert to BGR for colored annotations
    vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = image.shape
    
    colors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]
    
    for ann in annotations:
        class_id = ann['class_id']
        color = colors[class_id % len(colors)]
        
        if ann['type'] == 'bbox':
            bbox = ann['bbox']
            
            if format_type == 'yolo':
                # Convert from normalized YOLO format
                x_center, y_center, width, height = bbox
                x1 = int((x_center - width / 2) * w)
                y1 = int((y_center - height / 2) * h)
                x2 = int((x_center + width / 2) * w)
                y2 = int((y_center + height / 2) * h)
            else:
                x1, y1, x2, y2 = map(int, bbox)
            
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_image, f"Class {class_id}", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        elif ann['type'] == 'segmentation':
            polygon = ann['polygon']
            
            if format_type == 'yolo':
                # Convert from normalized coordinates
                points = np.array([[int(p[0] * w), int(p[1] * h)] for p in polygon])
            else:
                points = np.array([[int(p[0]), int(p[1])] for p in polygon])
            
            cv2.polylines(vis_image, [points], True, color, 2)
            
            # Fill with transparency
            overlay = vis_image.copy()
            cv2.fillPoly(overlay, [points], color)
            cv2.addWeighted(overlay, 0.3, vis_image, 0.7, 0, vis_image)
    
    save_image(vis_image, output_path)
    logger.info(f"Saved annotation visualization: {output_path}")


def compute_dataset_statistics(image_paths: List[str]) -> Dict:
    """Compute statistics across entire dataset"""
    logger.info(f"Computing statistics for {len(image_paths)} images...")
    
    resolutions = []
    aspect_ratios = []
    mean_intensities = []
    std_intensities = []
    
    for img_path in image_paths:
        image = load_image(img_path)
        if image is None:
            continue
        
        h, w = image.shape
        resolutions.append((w, h))
        aspect_ratios.append(w / h)
        mean_intensities.append(np.mean(image))
        std_intensities.append(np.std(image))
    
    stats = {
        'num_images': len(resolutions),
        'resolutions': {
            'min': np.min(resolutions, axis=0).tolist(),
            'max': np.max(resolutions, axis=0).tolist(),
            'mean': np.mean(resolutions, axis=0).tolist(),
            'median': np.median(resolutions, axis=0).tolist()
        },
        'aspect_ratios': {
            'min': float(np.min(aspect_ratios)),
            'max': float(np.max(aspect_ratios)),
            'mean': float(np.mean(aspect_ratios)),
            'median': float(np.median(aspect_ratios))
        },
        'intensity': {
            'mean': float(np.mean(mean_intensities)),
            'std': float(np.mean(std_intensities)),
            'global_mean': float(np.mean(mean_intensities)),
            'global_std': float(np.std(mean_intensities))
        }
    }
    
    logger.info(f"Dataset statistics computed: {stats}")
    return stats


def create_reference_histogram(image_paths: List[str], 
                               output_path: str,
                               num_samples: int = 100) -> np.ndarray:
    """Create reference histogram from high-quality samples"""
    logger.info(f"Creating reference histogram from {num_samples} images...")
    
    # Select subset of images
    selected_paths = image_paths[:num_samples]
    
    histograms = []
    for img_path in selected_paths:
        image = load_image(img_path)
        if image is None:
            continue
        
        # Compute histogram
        hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
        histograms.append(hist)
    
    # Average histograms
    reference_histogram = np.mean(histograms, axis=0).astype(np.float32)
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, reference_histogram)
    
    logger.info(f"Reference histogram saved to {output_path}")
    return reference_histogram


def validate_image_quality(image: np.ndarray, config: Dict) -> Tuple[bool, List[str]]:
    """Validate image quality based on criteria"""
    issues = []
    h, w = image.shape
    
    # Check resolution
    min_res = config['min_resolution']
    max_res = config['max_resolution']
    
    if w < min_res[0] or h < min_res[1]:
        issues.append(f"Resolution too low: {w}x{h} < {min_res[0]}x{min_res[1]}")
    
    if w > max_res[0] or h > max_res[1]:
        issues.append(f"Resolution too high: {w}x{h} > {max_res[0]}x{max_res[1]}")
    
    # Check brightness
    mean_intensity = np.mean(image)
    if mean_intensity < config['min_brightness']:
        issues.append(f"Image too dark: mean={mean_intensity:.1f}")
    if mean_intensity > config['max_brightness']:
        issues.append(f"Image too bright: mean={mean_intensity:.1f}")
    
    # Check blur (Laplacian variance)
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    if laplacian_var < config['blur_threshold']:
        issues.append(f"Image too blurry: variance={laplacian_var:.1f}")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def create_dataset_split(image_paths: List[str],
                        train_ratio: float = 0.7,
                        val_ratio: float = 0.15,
                        test_ratio: float = 0.15,
                        random_seed: int = 42) -> Dict[str, List[str]]:
    """Split dataset into train/val/test sets"""
    np.random.seed(random_seed)
    
    # Shuffle paths
    shuffled_paths = np.array(image_paths)
    np.random.shuffle(shuffled_paths)
    
    # Calculate split indices
    n = len(shuffled_paths)
    train_idx = int(n * train_ratio)
    val_idx = train_idx + int(n * val_ratio)
    
    splits = {
        'train': shuffled_paths[:train_idx].tolist(),
        'val': shuffled_paths[train_idx:val_idx].tolist(),
        'test': shuffled_paths[val_idx:].tolist()
    }
    
    logger.info(f"Dataset split: train={len(splits['train'])}, "
               f"val={len(splits['val'])}, test={len(splits['test'])}")
    
    return splits


def compare_before_after(original_image: np.ndarray,
                        processed_image: np.ndarray,
                        output_path: str):
    """Create side-by-side comparison visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title(f'Original\n{original_image.shape[1]}x{original_image.shape[0]}')
    axes[0].axis('off')
    
    axes[1].imshow(processed_image, cmap='gray')
    axes[1].set_title(f'Preprocessed\n{processed_image.shape[1]}x{processed_image.shape[0]}')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved before/after comparison: {output_path}")