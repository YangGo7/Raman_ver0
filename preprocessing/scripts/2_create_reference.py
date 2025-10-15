"""
Script 2: Create Reference Histogram
스크립트 2: 참조 히스토그램 생성

Select high-quality images and create reference histogram
고품질 이미지를 선택하여 참조 히스토그램을 생성합니다
"""

import sys
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils import setup_logging, load_image, create_reference_histogram, validate_image_quality


def create_reference(data_dir: str, 
                    output_path: str = "config/reference_histogram.npy",
                    num_samples: int = 100,
                    quality_filter: bool = True):
    """
    Create reference histogram from high-quality samples
    
    Args:
        data_dir: Directory containing images
        output_path: Path to save reference histogram
        num_samples: Number of images to use
        quality_filter: Whether to filter by quality
    """
    setup_logging()
    
    # Find all images
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_paths = []
    
    data_path = Path(data_dir)
    for ext in image_extensions:
        image_paths.extend(list(data_path.rglob(f'*{ext}')))
    
    print(f"Found {len(image_paths)} images")
    
    if len(image_paths) < num_samples:
        print(f"Warning: Only {len(image_paths)} images available, less than requested {num_samples}")
        num_samples = len(image_paths)
    
    # Quality filtering
    if quality_filter:
        print("\nFiltering images by quality...")
        
        quality_config = {
            'min_resolution': [1500, 700],
            'max_resolution': [5000, 2500],
            'min_brightness': 30,
            'max_brightness': 225,
            'blur_threshold': 100
        }
        
        quality_images = []
        for img_path in image_paths:
            image = load_image(str(img_path))
            if image is None:
                continue
            
            is_valid, issues = validate_image_quality(image, quality_config)
            if is_valid:
                quality_images.append(img_path)
            
            if len(quality_images) >= num_samples * 2:  # Get more than needed
                break
        
        print(f"Found {len(quality_images)} high-quality images")
        selected_paths = quality_images[:num_samples]
    else:
        # Random selection
        import random
        random.seed(42)
        selected_paths = random.sample(image_paths, num_samples)
    
    print(f"\nUsing {len(selected_paths)} images for reference histogram")
    
    # Load all selected images and compute histograms
    print("\nComputing histograms...")
    histograms = []
    intensities = []
    
    for i, img_path in enumerate(selected_paths):
        if (i + 1) % 10 == 0:
            print(f"Processing {i+1}/{len(selected_paths)}...")
        
        image = load_image(str(img_path))
        if image is None:
            continue
        
        # Compute histogram
        hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
        histograms.append(hist)
        intensities.append(np.mean(image))
    
    # Average histograms
    reference_histogram = np.mean(histograms, axis=0).astype(np.float32)
    
    # Normalize histogram
    reference_histogram = reference_histogram / np.sum(reference_histogram)
    
    # Save reference histogram
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_file, reference_histogram)
    
    print(f"\n✓ Reference histogram saved to: {output_file}")
    
    # Visualize reference histogram
    print("\nCreating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Reference histogram
    axes[0, 0].plot(reference_histogram, color='blue', linewidth=2)
    axes[0, 0].set_xlabel('Intensity Value')
    axes[0, 0].set_ylabel('Normalized Frequency')
    axes[0, 0].set_title('Reference Histogram')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. All individual histograms (overlaid)
    for hist in histograms[:50]:  # Show first 50
        axes[0, 1].plot(hist / np.sum(hist), alpha=0.1, color='gray')
    axes[0, 1].plot(reference_histogram, color='red', linewidth=2, label='Reference')
    axes[0, 1].set_xlabel('Intensity Value')
    axes[0, 1].set_ylabel('Normalized Frequency')
    axes[0, 1].set_title(f'Individual Histograms (n={len(histograms)})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Intensity distribution of selected images
    axes[1, 0].hist(intensities, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Mean Intensity')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Intensity Distribution of Reference Images')
    axes[1, 0].axvline(np.mean(intensities), color='r', linestyle='--', 
                      label=f'Mean: {np.mean(intensities):.1f}')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Sample images used for reference
    axes[1, 1].axis('off')
    axes[1, 1].text(0.5, 0.5, 
                   f'Reference Created\n\n'
                   f'Images used: {len(histograms)}\n'
                   f'Mean intensity: {np.mean(intensities):.1f}\n'
                   f'Std intensity: {np.std(intensities):.1f}\n\n'
                   f'Histogram saved to:\n{output_file}',
                   ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    viz_path = output_file.parent / 'reference_histogram_visualization.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization saved to: {viz_path}")
    
    # Save metadata
    metadata = {
        'num_images': len(histograms),
        'mean_intensity': float(np.mean(intensities)),
        'std_intensity': float(np.std(intensities)),
        'quality_filter': quality_filter,
        'histogram_sum': float(np.sum(reference_histogram))
    }
    
    metadata_path = output_file.parent / 'reference_metadata.yaml'
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    
    print(f"✓ Metadata saved to: {metadata_path}")
    
    # Compute and save global statistics
    print("\nComputing global statistics...")
    all_intensities = []
    for img_path in image_paths[:1000]:  # Sample 1000 images
        image = load_image(str(img_path))
        if image is not None:
            all_intensities.append(np.mean(image))
    
    global_stats = {
        'global_mean': float(np.mean(all_intensities)),
        'global_std': float(np.std(all_intensities)),
        'reference_mean': float(np.mean(intensities)),
        'reference_std': float(np.std(intensities))
    }
    
    stats_path = output_file.parent / 'global_stats.yaml'
    with open(stats_path, 'w') as f:
        yaml.dump(global_stats, f, default_flow_style=False)
    
    print(f"✓ Global statistics saved to: {stats_path}")
    
    print("\n" + "="*60)
    print("REFERENCE CREATION COMPLETE")
    print("="*60)
    print(f"Reference histogram: {output_file}")
    print(f"Visualization: {viz_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Global stats: {stats_path}")
    print("\nNext steps:")
    print("1. Review the reference histogram visualization")
    print("2. Adjust config/preprocessing_config.yaml if needed")
    print("3. Run: python scripts/3_preprocess_all.py")
    print("="*60)


if __name__ == "__main__":
    # 여기에 참조 히스토그램을 생성할 이미지 폴더 절대경로를 지정하세요
    # Specify absolute paths to image folders here
    data_dirs = [
        r"C:\dental_pano\rootinfection\YOLODataset\images\train",
        r"C:\dental_pano\rootinfection\YOLODataset\images\val",
        r"C:\dental_pano\instance_seg_data\images\train",
        r"C:\dental_pano\instance_seg_data\images\val",
        r"C:\dental_pano\bone_level_seg\test-1\train\images",
        r"C:\dental_pano\bone_level_seg\test-1\valid\images",
        r"C:\dental_pano\caries\YOLODataset\images\train",
        r"C:\dental_pano\caries\YOLODataset\images\val",
    ]

    # 설정
    # Settings
    num_samples = 100  # 각 폴더당 사용할 이미지 수
    quality_filter = True  # 품질 필터링 사용 여부
    base_output_dir = "config"  # 기본 저장 디렉토리

    # 각 폴더별로 참조 히스토그램 생성
    # Create reference histogram for each folder
    print("=" * 60)
    print("각 폴더별로 참조 히스토그램을 생성합니다")
    print("Creating reference histogram for each folder")
    print("=" * 60)

    for data_dir in data_dirs:
        data_path = Path(data_dir)

        # 폴더명 추출 및 정리
        # Extract and clean folder name
        # 예: C:\...\bone_level_seg\test-1\train\images -> bone_train
        path_parts = data_path.parts

        # 상위 폴더명 찾기 (데이터셋 이름)
        dataset_name = None
        folder_type = None

        for i, part in enumerate(path_parts):
            if part in ['train', 'val', 'valid', 'test']:
                folder_type = part
                # 데이터셋 이름은 상위 폴더들에서 추출
                if i >= 2:
                    dataset_name = path_parts[i-2]
                break

        if dataset_name is None:
            dataset_name = data_path.parent.name
        if folder_type is None:
            folder_type = data_path.name

        # 폴더명 정리 (bone_level_seg -> bone)
        dataset_name = dataset_name.replace('_level_seg', '').replace('_seg', '').replace('_data', '')
        folder_type = folder_type.replace('valid', 'val')

        # 최종 폴더명: dataset_type (예: bone_train, caries_val)
        folder_name = f"{dataset_name}_{folder_type}"

        # 출력 경로 설정
        output_path = str(Path(base_output_dir) / folder_name / "reference_histogram.npy")

        print(f"\n\n{'='*60}")
        print(f"생성 중: {folder_name}")
        print(f"Creating: {folder_name}")
        print(f"소스: {data_dir}")
        print(f"출력: {output_path}")
        print(f"{'='*60}\n")

        try:
            create_reference(
                data_dir,
                output_path,
                num_samples,
                quality_filter
            )
        except Exception as e:
            print(f"❌ Error processing {folder_name}: {e}")
            continue

    print("\n\n" + "=" * 60)
    print("모든 참조 히스토그램 생성 완료!")
    print("All reference histograms created!")
    print("=" * 60)