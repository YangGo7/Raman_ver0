"""
Dental Panoramic X-ray Preprocessing Pipeline
Main preprocessor class implementing all preprocessing steps
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from skimage import exposure, morphology
from skimage.measure import label, regionprops
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
        self.target_size = tuple(self.config['preprocessing']['resolution']['target_size'])
        self.target_ratio = self.config['preprocessing']['aspect_ratio']['target_ratio']
        
        # Load reference histogram for matching
        if self.config['preprocessing']['intensity']['histogram_matching']['enabled']:
            ref_path = self.config['preprocessing']['intensity']['histogram_matching']['reference_path']
            self.reference_histogram = self._load_reference_histogram(ref_path)
        else:
            self.reference_histogram = None
        
        # Initialize CLAHE
        if self.config['preprocessing']['intensity']['clahe']['enabled']:
            clip_limit = self.config['preprocessing']['intensity']['clahe']['clip_limit']
            tile_size = tuple(self.config['preprocessing']['intensity']['clahe']['tile_grid_size'])
            self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        else:
            self.clahe = None
        
        logger.info(f"Preprocessor initialized with target size: {self.target_size}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load YAML configuration file"""
        with open(config_path, 'r') as f:
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
        if self.config['preprocessing']['roi_detection']['enabled']:
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
        cfg = self.config['preprocessing']['roi_detection']
        
        # Apply Otsu's thresholding
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = np.ones((cfg['morphology']['kernel_size'], 
                         cfg['morphology']['kernel_size']), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, 
                                  iterations=cfg['morphology']['iterations'])
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, 
                                  iterations=cfg['morphology']['iterations'])
        
        # Find largest connected component
        labeled = label(binary)
        regions = regionprops(labeled)
        
        if not regions:
            # Fallback: use default crop
            logger.warning("No ROI detected, using fallback crop")
            return self._fallback_crop(image), {'method': 'fallback'}
        
        # Get largest region
        largest_region = max(regions, key=lambda r: r.area)
        
        # Check if area is reasonable
        min_area = cfg['min_area_ratio'] * image.shape[0] * image.shape[1]
        if largest_region.area < min_area:
            logger.warning(f"ROI area too small ({largest_region.area} < {min_area}), using fallback")
            return self._fallback_crop(image), {'method': 'fallback'}
        
        # Get bounding box with margin
        minr, minc, maxr, maxc = largest_region.bbox
        height, width = image.shape
        
        margin = cfg['margin_ratio']
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
            'area_ratio': largest_region.area / (height * width)
        }
        
        return cropped, roi_info
    
    def _fallback_crop(self, image: np.ndarray) -> np.ndarray:
        """Fallback crop using predefined margins"""
        cfg = self.config['preprocessing']['roi_detection']['fallback_crop']
        h, w = image.shape
        
        top = int(h * cfg['top'])
        bottom = h - int(h * cfg['bottom'])
        left = int(w * cfg['left'])
        right = w - int(w * cfg['right'])
        
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
        cfg = self.config['preprocessing']['aspect_ratio']
        h, w = image.shape
        current_ratio = w / h
        target_ratio = cfg['target_ratio']
        
        if cfg['method'] == 'pad':
            # Calculate target dimensions
            if current_ratio < target_ratio:
                # Image is too narrow, pad width
                target_w = int(h * target_ratio)
                pad_w = target_w - w
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                
                padded = cv2.copyMakeBorder(
                    image, 0, 0, pad_left, pad_right,
                    cv2.BORDER_CONSTANT, value=cfg['padding_value']
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
                    cv2.BORDER_CONSTANT, value=cfg['padding_value']
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
        Step 3: Standardize intensity using histogram matching and CLAHE
        
        Args:
            image: Input image
            
        Returns:
            standardized_image: Intensity-standardized image
            intensity_info: Processing metadata
        """
        intensity_info = {}
        result = image.copy()
        
        # Histogram Matching
        if self.config['preprocessing']['intensity']['histogram_matching']['enabled']:
            if self.reference_histogram is not None:
                result = exposure.match_histograms(result, self.reference_histogram)
                intensity_info['histogram_matched'] = True
            else:
                logger.warning("Reference histogram not available, skipping histogram matching")
                intensity_info['histogram_matched'] = False
        
        # CLAHE
        if self.config['preprocessing']['intensity']['clahe']['enabled'] and self.clahe is not None:
            result = self.clahe.apply(result)
            intensity_info['clahe_applied'] = True
        
        # Normalization
        norm_cfg = self.config['preprocessing']['intensity']['normalization']
        if norm_cfg['method'] == 'zscore':
            mean = np.mean(result)
            std = np.std(result)
            result = ((result - mean) / (std + 1e-8) * 50 + 128).clip(0, 255).astype(np.uint8)
            intensity_info['normalization'] = 'zscore'
        elif norm_cfg['method'] == 'minmax':
            result = ((result - result.min()) / (result.max() - result.min() + 1e-8) * 255).astype(np.uint8)
            intensity_info['normalization'] = 'minmax'
        
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
        cfg = self.config['preprocessing']['resolution']
        target_w, target_h = cfg['target_size']
        h, w = image.shape
        
        # Choose interpolation
        interp_methods = {
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4
        }
        interp = interp_methods.get(cfg['interpolation'], cv2.INTER_LINEAR)
        
        # Resize
        if cfg['preserve_aspect']:
            # Already handled in aspect ratio normalization
            resized = cv2.resize(image, (target_w, target_h), interpolation=interp)
        else:
            resized = cv2.resize(image, (target_w, target_h), interpolation=interp)
        
        resolution_info = {
            'original_size': [w, h],
            'target_size': [target_w, target_h],
            'interpolation': cfg['interpolation'],
            'scale_x': target_w / w,
            'scale_y': target_h / h
        }
        
        return resized, resolution_info