"""
Brightness Normalizer
밝기 정규화 모듈

Normalize brightness across different datasets to ensure consistency
여러 데이터셋 간 밝기를 정규화하여 일관성 확보
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class BrightnessNormalizer:
    """
    Normalize image brightness to match a target distribution
    이미지 밝기를 목표 분포에 맞춰 정규화
    """

    def __init__(self,
                 target_mean: float = 128.0,
                 target_std: float = 50.0,
                 method: str = 'zscore'):
        """
        Initialize brightness normalizer

        Args:
            target_mean: Target mean brightness (0-255)
            target_std: Target standard deviation
            method: Normalization method ('zscore', 'minmax', 'histogram')
        """
        self.target_mean = target_mean
        self.target_std = target_std
        self.method = method

        logger.info(f"BrightnessNormalizer initialized: "
                   f"target_mean={target_mean:.1f}, target_std={target_std:.1f}, method={method}")

    def normalize(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Normalize image brightness

        Args:
            image: Input grayscale image (H, W)

        Returns:
            normalized_image: Brightness-normalized image
            stats: Statistics dictionary
        """
        if len(image.shape) != 2:
            raise ValueError("Input must be grayscale image")

        # Original statistics
        original_mean = np.mean(image)
        original_std = np.std(image)

        # Select normalization method
        if self.method == 'zscore':
            normalized = self._zscore_normalize(image, original_mean, original_std)
        elif self.method == 'minmax':
            normalized = self._minmax_normalize(image)
        elif self.method == 'histogram':
            normalized = self._histogram_normalize(image)
        else:
            logger.warning(f"Unknown method: {self.method}, using zscore")
            normalized = self._zscore_normalize(image, original_mean, original_std)

        # Final statistics
        final_mean = np.mean(normalized)
        final_std = np.std(normalized)

        stats = {
            'method': self.method,
            'original_mean': float(original_mean),
            'original_std': float(original_std),
            'target_mean': float(self.target_mean),
            'target_std': float(self.target_std),
            'final_mean': float(final_mean),
            'final_std': float(final_std),
        }

        return normalized, stats

    def _zscore_normalize(self,
                          image: np.ndarray,
                          current_mean: float,
                          current_std: float) -> np.ndarray:
        """
        Z-score normalization (recommended)
        Z-점수 정규화 (권장)

        Transforms image to have target mean and std
        이미지를 목표 평균과 표준편차를 갖도록 변환
        """
        # Avoid division by zero
        if current_std < 1e-6:
            logger.warning("Image has near-zero std, skipping normalization")
            return image

        # Standardize (mean=0, std=1)
        standardized = (image.astype(np.float32) - current_mean) / current_std

        # Scale to target distribution
        normalized = standardized * self.target_std + self.target_mean

        # Clip to valid range
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)

        return normalized

    def _minmax_normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Min-max normalization
        최소-최대 정규화
        """
        img_min = image.min()
        img_max = image.max()

        if img_max - img_min < 1e-6:
            logger.warning("Image has near-zero range")
            return image

        # Normalize to 0-1
        normalized = (image.astype(np.float32) - img_min) / (img_max - img_min)

        # Scale to target range around target_mean
        range_half = self.target_std * 2  # Approximate range
        target_min = max(0, self.target_mean - range_half)
        target_max = min(255, self.target_mean + range_half)

        normalized = normalized * (target_max - target_min) + target_min
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)

        return normalized

    def _histogram_normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Histogram equalization followed by adjustment
        히스토그램 평활화 후 조정
        """
        # Apply CLAHE for better local contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(image)

        # Adjust to target mean
        current_mean = np.mean(equalized)
        shift = self.target_mean - current_mean

        adjusted = equalized.astype(np.float32) + shift
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)

        return adjusted

    def batch_analyze(self, images: list) -> dict:
        """
        Analyze brightness statistics for a batch of images
        이미지 배치의 밝기 통계 분석

        Args:
            images: List of grayscale images

        Returns:
            Statistics dictionary
        """
        means = []
        stds = []

        for img in images:
            if len(img.shape) == 2:
                means.append(np.mean(img))
                stds.append(np.std(img))

        return {
            'num_images': len(images),
            'mean_brightness': float(np.mean(means)),
            'std_brightness': float(np.std(means)),
            'mean_contrast': float(np.mean(stds)),
            'std_contrast': float(np.std(stds)),
        }


class AdaptiveBrightnessNormalizer:
    """
    Adaptive brightness normalizer that learns from reference dataset
    참조 데이터셋에서 학습하는 적응형 밝기 정규화
    """

    def __init__(self):
        self.target_mean = None
        self.target_std = None
        self.is_fitted = False

    def fit(self, images: list):
        """
        Learn target distribution from reference images
        참조 이미지들로부터 목표 분포 학습

        Args:
            images: List of reference grayscale images
        """
        all_means = []
        all_stds = []

        for img in images:
            if len(img.shape) == 2:
                all_means.append(np.mean(img))
                all_stds.append(np.std(img))

        self.target_mean = np.mean(all_means)
        self.target_std = np.mean(all_stds)
        self.is_fitted = True

        logger.info(f"Fitted to reference: mean={self.target_mean:.2f}, std={self.target_std:.2f}")

    def transform(self, image: np.ndarray) -> np.ndarray:
        """
        Transform image to match learned distribution
        학습된 분포에 맞춰 이미지 변환
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before transform")

        normalizer = BrightnessNormalizer(
            target_mean=self.target_mean,
            target_std=self.target_std,
            method='zscore'
        )

        normalized, _ = normalizer.normalize(image)
        return normalized

    def fit_transform(self, reference_images: list, target_images: list) -> list:
        """
        Fit on reference and transform target images
        참조로 학습하고 타겟 이미지 변환

        Args:
            reference_images: Reference dataset images
            target_images: Images to transform

        Returns:
            List of transformed images
        """
        self.fit(reference_images)

        transformed = []
        for img in target_images:
            transformed.append(self.transform(img))

        return transformed


def auto_brightness_correction(image: np.ndarray,
                               target_percentile: float = 50.0,
                               target_value: float = 128.0) -> np.ndarray:
    """
    Automatic brightness correction based on percentile
    백분위수 기반 자동 밝기 보정

    Args:
        image: Input grayscale image
        target_percentile: Percentile to match (default: median = 50%)
        target_value: Target value for the percentile

    Returns:
        Corrected image
    """
    current_value = np.percentile(image, target_percentile)
    shift = target_value - current_value

    corrected = image.astype(np.float32) + shift
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    return corrected


if __name__ == "__main__":
    # Test
    import matplotlib.pyplot as plt

    # Create test images with different brightness
    img1 = np.random.randint(100, 200, (100, 100), dtype=np.uint8)  # Bright
    img2 = np.random.randint(50, 150, (100, 100), dtype=np.uint8)   # Dark

    # Normalize
    normalizer = BrightnessNormalizer(target_mean=128, target_std=50)

    norm1, stats1 = normalizer.normalize(img1)
    norm2, stats2 = normalizer.normalize(img2)

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    axes[0, 0].imshow(img1, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title(f'Original 1\nMean: {stats1["original_mean"]:.1f}')

    axes[0, 1].imshow(norm1, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title(f'Normalized 1\nMean: {stats1["final_mean"]:.1f}')

    axes[0, 2].hist([img1.ravel(), norm1.ravel()], bins=50, label=['Original', 'Normalized'], alpha=0.7)
    axes[0, 2].axvline(128, color='red', linestyle='--', label='Target')
    axes[0, 2].legend()
    axes[0, 2].set_title('Histogram 1')

    axes[1, 0].imshow(img2, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title(f'Original 2\nMean: {stats2["original_mean"]:.1f}')

    axes[1, 1].imshow(norm2, cmap='gray', vmin=0, vmax=255)
    axes[1, 1].set_title(f'Normalized 2\nMean: {stats2["final_mean"]:.1f}')

    axes[1, 2].hist([img2.ravel(), norm2.ravel()], bins=50, label=['Original', 'Normalized'], alpha=0.7)
    axes[1, 2].axvline(128, color='red', linestyle='--', label='Target')
    axes[1, 2].legend()
    axes[1, 2].set_title('Histogram 2')

    plt.tight_layout()
    plt.savefig('brightness_normalization_test.png')
    print("Test completed. Check brightness_normalization_test.png")
