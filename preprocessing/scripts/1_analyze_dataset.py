"""
Script 1: Analyze Dataset
스크립트 1: 데이터셋 분석

Analyze the entire dataset and generate statistics report
전체 데이터셋을 분석하고 통계 보고서를 생성합니다
"""

import sys
from pathlib import Path
import yaml
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add src to path / src 경로 추가
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils import setup_logging, load_image, compute_dataset_statistics

def analyze_dataset(data_dirs, output_dir: str = "analysis_results"):
    """
    Analyze dataset and generate comprehensive report
    데이터셋을 분석하고 종합 보고서를 생성합니다

    Args:
        data_dirs: Root directory or list of directories containing images
                  이미지가 포함된 루트 디렉토리 또는 디렉토리 리스트
        output_dir: Directory to save analysis results
                   분석 결과를 저장할 디렉토리
    """
    setup_logging()

    # Convert to list if single directory / 단일 디렉토리면 리스트로 변환
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]

    # Create output directory / 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all images from all directories / 모든 디렉토리에서 이미지 찾기
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_paths = []

    for data_dir in data_dirs:
        data_path = Path(data_dir)
        print(f"Scanning directory: {data_dir}")
        for ext in image_extensions:
            found = list(data_path.rglob(f'*{ext}'))
            image_paths.extend(found)
            if found:
                print(f"  Found {len(found)} {ext} files")

    print(f"\nTotal: Found {len(image_paths)} images from {len(data_dirs)} directories")
    
    if len(image_paths) == 0:
        print("No images found! Please check the data directory.")
        return
    
    # Compute statistics / 통계 계산
    print("\nComputing dataset statistics...")
    stats = compute_dataset_statistics([str(p) for p in image_paths])

    # Save statistics / 통계 저장
    stats_file = output_path / "dataset_statistics.yaml"
    with open(stats_file, 'w') as f:
        yaml.dump(stats, f, default_flow_style=False)
    print(f"\nStatistics saved to {stats_file}")

    # Print summary / 요약 출력
    print("\n" + "="*60)
    print("DATASET STATISTICS SUMMARY")
    print("="*60)
    print(f"Total images: {stats['num_images']}")
    print(f"\nResolution range:")
    print(f"  Min: {stats['resolutions']['min']}")
    print(f"  Max: {stats['resolutions']['max']}")
    print(f"  Mean: {[int(x) for x in stats['resolutions']['mean']]}")
    print(f"  Median: {[int(x) for x in stats['resolutions']['median']]}")
    print(f"\nAspect ratio:")
    print(f"  Min: {stats['aspect_ratios']['min']:.2f}")
    print(f"  Max: {stats['aspect_ratios']['max']:.2f}")
    print(f"  Mean: {stats['aspect_ratios']['mean']:.2f}")
    print(f"  Median: {stats['aspect_ratios']['median']:.2f}")
    print(f"\nIntensity:")
    print(f"  Mean: {stats['intensity']['mean']:.2f}")
    print(f"  Std: {stats['intensity']['std']:.2f}")
    print("="*60)
    
    # Detailed analysis - collect per-image data / 세부 분석 - 이미지별 데이터 수집
    print("\nPerforming detailed analysis...")
    resolutions = []
    aspect_ratios = []
    intensities = []
    file_sizes = []

    for img_path in image_paths[:1000]:  # Limit to 1000 for speed / 속도를 위해 1000개로 제한
        try:
            image = load_image(str(img_path))
            if image is None:
                continue
            
            h, w = image.shape
            resolutions.append((w, h))
            aspect_ratios.append(w / h)
            intensities.append(np.mean(image))
            file_sizes.append(img_path.stat().st_size / 1024)  # KB / 킬로바이트
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    # Create visualizations / 시각화 생성
    print("\nCreating visualizations...")

    # 1. Resolution distribution / 해상도 분포
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Width distribution / 너비 분포
    widths = [r[0] for r in resolutions]
    axes[0, 0].hist(widths, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Width (pixels)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Width Distribution')
    axes[0, 0].axvline(np.mean(widths), color='r', linestyle='--', label='Mean')
    axes[0, 0].legend()

    # Height distribution / 높이 분포
    heights = [r[1] for r in resolutions]
    axes[0, 1].hist(heights, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Height (pixels)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Height Distribution')
    axes[0, 1].axvline(np.mean(heights), color='r', linestyle='--', label='Mean')
    axes[0, 1].legend()

    # Aspect ratio distribution / 종횡비 분포
    axes[1, 0].hist(aspect_ratios, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Aspect Ratio (W/H)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Aspect Ratio Distribution')
    axes[1, 0].axvline(np.mean(aspect_ratios), color='r', linestyle='--', label='Mean')
    axes[1, 0].axvline(2.0, color='g', linestyle='--', label='Target (2:1)')
    axes[1, 0].legend()

    # Intensity distribution / 밝기 분포
    axes[1, 1].hist(intensities, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Mean Intensity')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Intensity Distribution')
    axes[1, 1].axvline(np.mean(intensities), color='r', linestyle='--', label='Mean')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'distribution_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'distribution_analysis.png'}")

    # 2. Scatter plot: Resolution vs Aspect Ratio / 산점도: 해상도 vs 종횡비
    fig, ax = plt.subplots(figsize=(10, 6))
    areas = [w * h for w, h in resolutions]
    scatter = ax.scatter(aspect_ratios, areas, c=intensities, cmap='viridis', alpha=0.6)
    ax.set_xlabel('Aspect Ratio (W/H)')
    ax.set_ylabel('Image Area (pixels²)')
    ax.set_title('Resolution vs Aspect Ratio (colored by intensity)')
    ax.axvline(2.0, color='r', linestyle='--', label='Target Aspect Ratio (2:1)')
    plt.colorbar(scatter, label='Mean Intensity')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path / 'resolution_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'resolution_scatter.png'}")

    # 3. Sample images / 샘플 이미지
    print("\nSaving sample images...")
    sample_dir = output_path / 'samples'
    sample_dir.mkdir(exist_ok=True)

    # Select diverse samples (low, medium, high resolution) / 다양한 샘플 선택 (낮음, 중간, 높은 해상도)
    sorted_by_area = sorted(zip(image_paths, areas), key=lambda x: x[1])
    sample_indices = [
        0,  # Smallest / 가장 작은
        len(sorted_by_area) // 4,  # 25th percentile / 25 백분위수
        len(sorted_by_area) // 2,  # Median / 중앙값
        3 * len(sorted_by_area) // 4,  # 75th percentile / 75 백분위수
        len(sorted_by_area) - 1  # Largest / 가장 큰
    ]
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i, idx in enumerate(sample_indices):
        img_path, area = sorted_by_area[idx]
        image = load_image(str(img_path))
        if image is not None:
            axes[i].imshow(image, cmap='gray')
            axes[i].set_title(f'{image.shape[1]}x{image.shape[0]}\nArea: {area:.0f}px²')
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(sample_dir / 'sample_images.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {sample_dir / 'sample_images.png'}")

    # Generate recommendations / 권장사항 생성
    print("\n" + "="*60)
    print("PREPROCESSING RECOMMENDATIONS")
    print("="*60)
    
    # 종횡비 검사
    mean_aspect = stats['aspect_ratios']['mean']
    if abs(mean_aspect - 2.0) > 0.3:
        print(f"⚠️  Mean aspect ratio ({mean_aspect:.2f}) differs from target (2.0)")
        print("   → Aspect ratio normalization is CRITICAL / 종횡비 정규화 필수")
    else:
        print(f"✓  Mean aspect ratio ({mean_aspect:.2f}) is close to target")

    # 밝기 변동성 검사
    intensity_std = stats['intensity']['global_std']
    if intensity_std > 30:
        print(f"\n⚠️  High intensity variance (std={intensity_std:.1f})")
        print("   → Histogram matching + CLAHE recommended / 히스토그램 매칭 + CLAHE 권장")
    else:
        print(f"\n✓  Intensity variance is acceptable (std={intensity_std:.1f})")

    # 해상도 변동성 검사
    res_variance = np.std([w for w, h in resolutions])
    if res_variance > 500:
        print(f"\n⚠️  High resolution variance (std={res_variance:.0f})")
        print("   → Resolution standardization is CRITICAL / 해상도 표준화 필수")
    else:
        print(f"\n✓  Resolution variance is manageable (std={res_variance:.0f})")
    
    # Generate preprocessing config based on analysis / 분석 결과 기반 전처리 설정 생성
    print("\n" + "="*60)
    print("전처리 설정 파일 생성")
    print("Generating preprocessing config")
    print("="*60)

    config = {
        'roi': {
            'method': 'auto',
            'fallback_bbox': None,
            'padding': 20
        },
        'aspect_ratio': {
            'target_ratio': 2.0,
            'method': 'pad',
            'pad_value': 0
        },
        'resolution': {
            'target_width': 2048,
            'target_height': 1024,
            'interpolation': 'lanczos',
            'maintain_aspect': True
        },
        'histogram_matching': {
            'enabled': False,  # Will be updated by script 2
            'reference_path': str(output_path / 'reference_histogram.npy'),
            'method': 'exact'
        },
        'clahe': {
            'enabled': False,
            'clip_limit': 2.0,
            'tile_size': [8, 8]
        },
        'denoising': {
            'enabled': False,
            'method': 'bilateral',
            'strength': 3
        },
        'sharpening': {
            'enabled': False,
            'strength': 1.0
        },
        'output': {
            'format': 'png',
            'quality': 95,
            'bit_depth': 8
        },
        'labels': {
            'format': 'yolo',
            'validate': True,
            'min_bbox_size': 5,
            'filter_invalid': True
        },
        'validation': {
            'enabled': True,
            'min_resolution': [640, 320],
            'max_resolution': [8192, 4096],
            'min_brightness': 10,
            'max_brightness': 245,
            'blur_threshold': 50
        }
    }

    # Adjust based on analysis results / 분석 결과에 따라 조정
    intensity_std = stats['intensity']['global_std']
    if intensity_std > 30:
        print("⚠️  높은 밝기 변동 감지 → 히스토그램 매칭 및 CLAHE 활성화")
        config['histogram_matching']['enabled'] = True
        config['clahe']['enabled'] = True
    elif intensity_std > 15:
        print("⚠️  중간 밝기 변동 감지 → CLAHE 활성화")
        config['clahe']['enabled'] = True
    else:
        print("✓  밝기 변동 낮음 → 추가 처리 불필요")

    res_variance = np.std([w for w, h in resolutions])
    if res_variance > 500:
        print("✓  해상도 표준화 필수 (이미 설정됨)")

    mean_aspect = stats['aspect_ratios']['mean']
    if abs(mean_aspect - 2.0) > 0.3:
        print("⚠️  종횡비 조정 필요 (이미 설정됨)")

    # Save config / 설정 저장
    config_path = output_path / 'preprocessing_config.yaml'
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"\n✓ 전처리 설정 저장: {config_path}")

    print("\n" + "="*60)
    print(f"\n분석 완료! Analysis complete!")
    print(f"결과 저장 위치: {output_path}")
    print("\n다음 단계:")
    print("Next steps:")
    print("1. 분석 결과 검토 / Review the analysis results")
    print("2. 참조 히스토그램 생성 (필요시) / Create reference histogram (if needed)")
    print("   python scripts/2_create_reference.py")
    print("3. 전처리 실행 / Run preprocessing")
    print("   python scripts/3_preprocess_all.py")
    print("="*60)


if __name__ == "__main__":
    # 여기에 분석할 이미지 폴더 절대경로를 지정하세요
    # Specify absolute paths to image folders here
    # 분석할 데이터셋 설정 (경로 + 출력 폴더명)
    # Dataset configurations (path + output folder name)
    data_configs = [
        {
            'data_dir': r"C:\dental_pano\rootinfection\YOLODataset\images\train",
            'output_name': 'rootinfection_train'
        },
        {
            'data_dir': r"C:\dental_pano\rootinfection\YOLODataset\images\val",
            'output_name': 'rootinfection_val'
        },
        {
            'data_dir': r"C:\dental_pano\instance_seg_data\images\train",
            'output_name': 'instance_train'
        },
        {
            'data_dir': r"C:\dental_pano\instance_seg_data\images\val",
            'output_name': 'instance_val'
        },
        {
            'data_dir': r"C:\dental_pano\bone_level_seg\test-1\train\images",
            'output_name': 'bone_train'
        },
        {
            'data_dir': r"C:\dental_pano\bone_level_seg\test-1\valid\images",
            'output_name': 'bone_val'
        },
        {
            'data_dir': r"C:\dental_pano\caries\YOLODataset\images\train",
            'output_name': 'caries_train'
        },
        {
            'data_dir': r"C:\dental_pano\caries\YOLODataset\images\val",
            'output_name': 'caries_val'
        },
    ]

    # 분석 결과를 저장할 기본 디렉토리
    # Base directory to save analysis results
    base_output_dir = "analysis_results"

    # 각 폴더별로 개별 분석
    # Analyze each folder separately
    print("=" * 60)
    print("각 폴더별로 개별 분석을 수행합니다")
    print("Analyzing each folder separately")
    print("=" * 60)

    for config in data_configs:
        data_dir = config['data_dir']
        output_name = config['output_name']
        output_dir = str(Path(base_output_dir) / output_name)

        print(f"\n\n{'='*60}")
        print(f"분석 중: {output_name}")
        print(f"Analyzing: {output_name}")
        print(f"{'='*60}\n")

        analyze_dataset(data_dir, output_dir)