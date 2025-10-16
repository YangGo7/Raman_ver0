"""
Dental Panoramic X-ray Preprocessing Pipeline
Main preprocessor class implementing all preprocessing steps
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class DentalPreprocessor:
    """
    Universal preprocessor for dental panoramic X-ray images
    Standardizes images from various sources to uniform format
    """
    
    def __init__(self, config_path: str = "config/preprocessing_config.yaml"):
        """
        Initialize preprocessor with configuration
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)

        # Extract target size from config
        res_config = self.config.get('resolution', {})
        target_w = res_config.get('target_width', 2048)
        target_h = res_config.get('target_height', 1024)
        self.target_size = (target_w, target_h)

        # Extract aspect ratio
        aspect_config = self.config.get('aspect_ratio', {})
        self.target_ratio = aspect_config.get('target_ratio', 2.0)

        # Load reference histogram for matching
        hist_config = self.config.get('histogram_matching', {})
        if hist_config.get('enabled', False):
            ref_path = hist_config.get('reference_path', '')
            self.reference_histogram = self._load_reference_histogram(ref_path) if ref_path else None
        else:
            self.reference_histogram = None

        # Initialize CLAHE
        clahe_config = self.config.get('clahe', {})
        if clahe_config.get('enabled', False):
            clip_limit = clahe_config.get('clip_limit', 2.0)
            tile_size = tuple(clahe_config.get('tile_size', [8, 8]))
            self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        else:
            self.clahe = None
        
        logger.info(f"Preprocessor initialized with target size: {self.target_size}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load YAML configuration file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def _load_reference_histogram(self, ref_path: str) -> np.ndarray:
        """Load pre-computed reference histogram"""
        if Path(ref_path).exists():
            return np.load(ref_path)
        else:
            logger.warning(f"Reference histogram not found at {ref_path}")
            return None
    
    def preprocess(self, image: np.ndarray, 
                   visualize: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        Main preprocessing pipeline
        
        Args:
            image: Input grayscale image (H, W) or (H, W, 3)
            visualize: Whether to save intermediate steps
            
        Returns:
            preprocessed_image: Standardized image
            metadata: Processing metadata including transforms
        """
        metadata = {
            'original_shape': image.shape,
            'transforms': {}
        }
        
        # Ensure grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        intermediate_images = {'original': image.copy()}
        
        # Step 1: ROI Detection & Crop
        roi_config = self.config.get('roi', {})
        if roi_config.get('method') == 'auto':
            image, roi_info = self.detect_and_crop_roi(image)
            metadata['transforms']['roi'] = roi_info
            intermediate_images['roi_cropped'] = image.copy()
        
        # Step 2: Aspect Ratio Normalization
        image, aspect_info = self.normalize_aspect_ratio(image)
        metadata['transforms']['aspect_ratio'] = aspect_info
        intermediate_images['aspect_normalized'] = image.copy()
        
        # Step 3: Intensity Standardization
        image, intensity_info = self.standardize_intensity(image)
        metadata['transforms']['intensity'] = intensity_info
        intermediate_images['intensity_standardized'] = image.copy()
        
        # Step 4: Resolution Standardization
        image, resolution_info = self.standardize_resolution(image)
        metadata['transforms']['resolution'] = resolution_info
        intermediate_images['final'] = image.copy()
        
        metadata['final_shape'] = image.shape
        
        if visualize:
            metadata['intermediate_images'] = intermediate_images
        
        return image, metadata
    
    def detect_and_crop_roi(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Step 1: Detect teeth region and crop
        
        Args:
            image: Input grayscale image
            
        Returns:
            cropped_image: Image cropped to ROI
            roi_info: ROI coordinates and metadata
        """
        cfg = self.config.get('roi', {})
        
        # Apply Otsu's thresholding
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel_size = cfg.get('kernel_size', 5)
        iterations = cfg.get('iterations', 2)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations)
        
        # Find largest connected component using OpenCV
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        # Skip background (label 0), find largest component
        if num_labels <= 1:
            # Fallback: use default crop
            logger.warning("No ROI detected, using fallback crop")
            return self._fallback_crop(image), {'method': 'fallback'}

        # Get largest region (excluding background)
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = np.argmax(areas) + 1  # +1 because we skipped background

        # Check if area is reasonable
        min_area = cfg.get('min_area_ratio', 0.1) * image.shape[0] * image.shape[1]
        largest_area = stats[largest_idx, cv2.CC_STAT_AREA]

        if largest_area < min_area:
            logger.warning(f"ROI area too small ({largest_area} < {min_area}), using fallback")
            return self._fallback_crop(image), {'method': 'fallback'}

        # Get bounding box with margin
        minc = stats[largest_idx, cv2.CC_STAT_LEFT]
        minr = stats[largest_idx, cv2.CC_STAT_TOP]
        width = stats[largest_idx, cv2.CC_STAT_WIDTH]
        height = stats[largest_idx, cv2.CC_STAT_HEIGHT]
        maxc = minc + width
        maxr = minr + height
        height, width = image.shape
        
        margin = cfg.get('margin_ratio', 0.05)
        margin_h = int((maxr - minr) * margin)
        margin_w = int((maxc - minc) * margin)
        
        minr = max(0, minr - margin_h)
        maxr = min(height, maxr + margin_h)
        minc = max(0, minc - margin_w)
        maxc = min(width, maxc + margin_w)
        
        # Crop
        cropped = image[minr:maxr, minc:maxc]
        
        roi_info = {
            'method': 'otsu',
            'bbox': [minc, minr, maxc, maxr],  # x1, y1, x2, y2
            'area_ratio': largest_area / (height * width)
        }
        
        return cropped, roi_info
    
    def _fallback_crop(self, image: np.ndarray) -> np.ndarray:
        """Fallback crop using predefined margins"""
        h, w = image.shape

        # Use default margins if not specified
        top = int(h * 0.1)
        bottom = h - int(h * 0.1)
        left = int(w * 0.05)
        right = w - int(w * 0.05)

        return image[top:bottom, left:right]
    
    def normalize_aspect_ratio(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Step 2: Normalize aspect ratio to target (2:1)
        
        Args:
            image: Input image
            
        Returns:
            normalized_image: Image with target aspect ratio
            aspect_info: Transformation metadata
        """
        cfg = self.config.get('aspect_ratio', {})
        h, w = image.shape
        current_ratio = w / h
        target_ratio = cfg.get('target_ratio', 2.0)

        if cfg.get('method') == 'pad':
            # Calculate target dimensions
            pad_value = cfg.get('pad_value', 0)

            if current_ratio < target_ratio:
                # Image is too narrow, pad width
                target_w = int(h * target_ratio)
                pad_w = target_w - w
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left

                padded = cv2.copyMakeBorder(
                    image, 0, 0, pad_left, pad_right,
                    cv2.BORDER_CONSTANT, value=pad_value
                )
                aspect_info = {
                    'method': 'pad_width',
                    'pad': [0, 0, pad_left, pad_right]  # top, bottom, left, right
                }
            else:
                # Image is too wide, pad height
                target_h = int(w / target_ratio)
                pad_h = target_h - h
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top

                padded = cv2.copyMakeBorder(
                    image, pad_top, pad_bottom, 0, 0,
                    cv2.BORDER_CONSTANT, value=pad_value
                )
                aspect_info = {
                    'method': 'pad_height',
                    'pad': [pad_top, pad_bottom, 0, 0]
                }

            aspect_info['original_ratio'] = current_ratio
            aspect_info['target_ratio'] = target_ratio

            return padded, aspect_info

        else:
            # No padding, just record the ratio
            return image, {'method': 'none', 'ratio': current_ratio}
    
    def standardize_intensity(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Step 3: Standardize intensity using brightness normalization, histogram matching and CLAHE

        Args:
            image: Input image

        Returns:
            standardized_image: Intensity-standardized image
            intensity_info: Processing metadata
        """
        intensity_info = {}
        result = image.copy()

        # Brightness Normalization (적용 우선순위 1)
        brightness_config = self.config.get('brightness_normalization', {})
        if brightness_config.get('enabled', False):
            method = brightness_config.get('method', 'zscore')

            if method == 'zscore':
                target_mean = brightness_config.get('target_mean', 127.5)
                target_std = brightness_config.get('target_std', 50.0)

                # Z-score normalization
                current_mean = np.mean(result)
                current_std = np.std(result)

                if current_std > 0:
                    result = (result - current_mean) / current_std
                    result = result * target_std + target_mean
                    result = np.clip(result, 0, 255).astype(np.uint8)

                    intensity_info['brightness_normalized'] = {
                        'method': 'zscore',
                        'original_mean': float(current_mean),
                        'original_std': float(current_std),
                        'target_mean': target_mean,
                        'target_std': target_std
                    }
                else:
                    logger.warning("Image has zero std, skipping brightness normalization")
                    intensity_info['brightness_normalized'] = False

            elif method == 'minmax':
                # Min-Max normalization to [0, 255]
                min_val = np.min(result)
                max_val = np.max(result)

                if max_val > min_val:
                    result = ((result - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                    intensity_info['brightness_normalized'] = {
                        'method': 'minmax',
                        'original_range': [float(min_val), float(max_val)]
                    }
                else:
                    logger.warning("Image has constant intensity, skipping brightness normalization")
                    intensity_info['brightness_normalized'] = False

            elif method == 'histogram':
                # Histogram equalization
                result = cv2.equalizeHist(result)
                intensity_info['brightness_normalized'] = {'method': 'histogram_equalization'}

            else:
                logger.warning(f"Unknown brightness normalization method: {method}")
                intensity_info['brightness_normalized'] = False
        else:
            intensity_info['brightness_normalized'] = False

        # Histogram Matching
        hist_config = self.config.get('histogram_matching', {})
        if hist_config.get('enabled', False):
            if self.reference_histogram is not None:
                # Simple histogram equalization instead of matching
                logger.warning("Histogram matching not supported without skimage, using histogram equalization")
                result = cv2.equalizeHist(result)
                intensity_info['histogram_matched'] = 'equalized'
            else:
                logger.warning("Reference histogram not available, skipping histogram matching")
                intensity_info['histogram_matched'] = False

        # CLAHE
        if self.clahe is not None:
            result = self.clahe.apply(result)
            intensity_info['clahe_applied'] = True

        return result, intensity_info
    
    def standardize_resolution(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Step 4: Resize to target resolution
        
        Args:
            image: Input image
            
        Returns:
            resized_image: Image at target resolution
            resolution_info: Resize metadata
        """
        cfg = self.config.get('resolution', {})
        target_w = cfg.get('target_width', 2048)
        target_h = cfg.get('target_height', 1024)
        h, w = image.shape

        # Choose interpolation
        interp_methods = {
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4
        }
        interp_method = cfg.get('interpolation', 'lanczos')
        interp = interp_methods.get(interp_method, cv2.INTER_LINEAR)

        # Resize
        resized = cv2.resize(image, (target_w, target_h), interpolation=interp)

        resolution_info = {
            'original_size': [w, h],
            'target_size': [target_w, target_h],
            'interpolation': interp_method,
            'scale_x': target_w / w,
            'scale_y': target_h / h
        }

        return resized, resolution_info