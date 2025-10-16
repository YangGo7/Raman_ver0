"""
Script 0: Dataset Comparison and Analysis
스크립트 0: 데이터셋 비교 및 분석

Compare multiple datasets and visualize differences in resolution, brightness, and contrast
여러 데이터셋을 비교하고 해상도, 밝기, 대비 차이를 시각화합니다
"""

import sys
from pathlib import Path
import yaml
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import Dict, List
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from utils import setup_logging, load_image


def setup_matplotlib_korean():
    """
    Setup matplotlib to display Korean characters
    matplotlib 한글 폰트 설정
    """
    import platform

    # Windows
    if platform.system() == 'Windows':
        # Try common Korean fonts on Windows
        korean_fonts = ['Malgun Gothic', 'NanumGothic', 'Gulim', 'Batang']

        for font_name in korean_fonts:
            try:
                plt.rc('font', family=font_name)
                # Test if it works
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, '한글테스트')
                plt.close(fig)
                print(f"✓ 한글 폰트 설정 완료: {font_name}")
                break
            except:
                continue
        else:
            # Fallback: use DejaVu Sans (no Korean support, but won't crash)
            plt.rc('font', family='DejaVu Sans')
            print("⚠️  한글 폰트를 찾을 수 없습니다. 한글이 깨져 보일 수 있습니다.")

    # macOS
    elif platform.system() == 'Darwin':
        plt.rc('font', family='AppleGothic')

    # Linux
    else:
        plt.rc('font', family='NanumGothic')

    # Prevent minus sign from breaking
    plt.rc('axes', unicode_minus=False)


def analyze_single_dataset(image_dir: Path, dataset_name: str) -> Dict:
    """
    Analyze a single dataset
    단일 데이터셋 분석

    Args:
        image_dir: Directory containing images
        dataset_name: Name of the dataset

    Returns:
        Dictionary with statistics
    """
    print(f"\n분석 중: {dataset_name}")
    print(f"디렉토리: {image_dir}")

    # Find all images
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(list(image_dir.rglob(f'*{ext}')))

    if len(image_paths) == 0:
        print(f"⚠️  이미지를 찾을 수 없습니다!")
        return None

    print(f"찾은 이미지: {len(image_paths)}개")

    # Statistics
    resolutions = []
    intensities_mean = []
    intensities_std = []

    for img_path in tqdm(image_paths, desc=f"  {dataset_name}"):
        img = load_image(str(img_path))
        if img is None:
            continue

        # Ensure grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resolution
        h, w = img.shape
        resolutions.append([w, h])

        # Intensity statistics
        intensities_mean.append(np.mean(img))
        intensities_std.append(np.std(img))

    # Aggregate statistics
    resolutions = np.array(resolutions)

    stats = {
        'name': dataset_name,
        'num_images': len(image_paths),
        'resolutions': {
            'mean': resolutions.mean(axis=0).tolist(),
            'std': resolutions.std(axis=0).tolist(),
            'min': resolutions.min(axis=0).tolist(),
            'max': resolutions.max(axis=0).tolist(),
        },
        'intensity': {
            'global_mean': np.mean(intensities_mean),
            'global_std': np.std(intensities_mean),
            'mean_of_std': np.mean(intensities_std),
            'std_of_std': np.std(intensities_std),
        }
    }

    return stats


def compare_datasets(all_stats: List[Dict], output_dir: Path):
    """
    Compare multiple datasets and visualize
    여러 데이터셋 비교 및 시각화

    Args:
        all_stats: List of statistics dictionaries
        output_dir: Output directory for plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("데이터셋 비교 분석 결과")
    print("="*60)

    # Print comparison
    for stats in all_stats:
        print(f"\n📊 {stats['name']}")
        print(f"  이미지 개수: {stats['num_images']}장")
        print(f"  평균 해상도: {stats['resolutions']['mean'][0]:.1f} × {stats['resolutions']['mean'][1]:.1f}")
        print(f"  평균 밝기: {stats['intensity']['global_mean']:.2f} (±{stats['intensity']['global_std']:.2f})")
        print(f"  평균 대비: {stats['intensity']['mean_of_std']:.2f}")

    # Find brightness differences
    print("\n" + "="*60)
    print("💡 밝기 차이 분석")
    print("="*60)

    means = [s['intensity']['global_mean'] for s in all_stats]
    max_idx = np.argmax(means)
    min_idx = np.argmin(means)

    print(f"가장 밝은 데이터셋: {all_stats[max_idx]['name']} (평균: {means[max_idx]:.2f})")
    print(f"가장 어두운 데이터셋: {all_stats[min_idx]['name']} (평균: {means[min_idx]:.2f})")
    print(f"밝기 차이: {means[max_idx] - means[min_idx]:.2f} ({(means[max_idx] - means[min_idx])/255*100:.1f}%)")

    # Visualize
    visualize_comparison(all_stats, output_dir)


def visualize_comparison(all_stats: List[Dict], output_dir: Path):
    """
    Create comparison visualizations
    비교 시각화 생성
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    names = [s['name'] for s in all_stats]
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_stats)))

    # 1. Resolution comparison
    ax1 = fig.add_subplot(gs[0, 0])
    widths = [s['resolutions']['mean'][0] for s in all_stats]
    heights = [s['resolutions']['mean'][1] for s in all_stats]

    x = np.arange(len(names))
    width = 0.35
    bars1 = ax1.bar(x - width/2, widths, width, label='Width', alpha=0.8, color='steelblue')
    bars2 = ax1.bar(x + width/2, heights, width, label='Height', alpha=0.8, color='coral')

    ax1.set_ylabel('Pixels', fontsize=11)
    ax1.set_title('Average Resolution Comparison\n평균 해상도 비교', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)

    # 2. Brightness comparison
    ax2 = fig.add_subplot(gs[0, 1])
    means = [s['intensity']['global_mean'] for s in all_stats]
    stds = [s['intensity']['global_std'] for s in all_stats]

    bars = ax2.bar(names, means, alpha=0.8, color=colors, edgecolor='black', linewidth=1.5)
    ax2.errorbar(names, means, yerr=stds, fmt='none', color='black', capsize=5, linewidth=2)
    ax2.set_ylabel('Pixel Intensity (0-255)', fontsize=11)
    ax2.set_title('Average Brightness Comparison\n평균 밝기 비교 (± Std)', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 255)
    ax2.axhline(y=128, color='gray', linestyle='--', alpha=0.5, label='Middle (128)', linewidth=2)
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        ax2.text(bar.get_x() + bar.get_width()/2., mean,
                f'{mean:.1f}\n±{std:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 3. Contrast comparison
    ax3 = fig.add_subplot(gs[1, 0])
    contrasts = [s['intensity']['mean_of_std'] for s in all_stats]
    bars = ax3.bar(names, contrasts, alpha=0.8, color=colors, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Standard Deviation', fontsize=11)
    ax3.set_title('Contrast Comparison\n대비(표준편차) 비교', fontsize=12, fontweight='bold')
    ax3.set_xticklabels(names, rotation=45, ha='right')
    ax3.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, contrast in zip(bars, contrasts):
        ax3.text(bar.get_x() + bar.get_width()/2., contrast,
                f'{contrast:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 4. Brightness distribution estimation
    ax4 = fig.add_subplot(gs[1, 1])
    x = np.linspace(0, 255, 1000)

    for i, stats in enumerate(all_stats):
        mean = stats['intensity']['global_mean']
        std = stats['intensity']['mean_of_std']

        # Gaussian distribution
        y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
        ax4.plot(x, y, label=stats['name'], linewidth=2.5, alpha=0.8, color=colors[i])
        ax4.fill_between(x, y, alpha=0.2, color=colors[i])

    ax4.set_xlabel('Pixel Intensity (0-255)', fontsize=11)
    ax4.set_ylabel('Probability Density', fontsize=11)
    ax4.set_title('Estimated Brightness Distribution\n밝기 분포 추정', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)
    ax4.set_xlim(0, 255)

    # 5. Image count comparison
    ax5 = fig.add_subplot(gs[2, 0])
    counts = [s['num_images'] for s in all_stats]
    bars = ax5.bar(names, counts, alpha=0.8, color=colors, edgecolor='black', linewidth=1.5)
    ax5.set_ylabel('Number of Images', fontsize=11)
    ax5.set_title('Dataset Size Comparison\n데이터셋 크기 비교', fontsize=12, fontweight='bold')
    ax5.set_xticklabels(names, rotation=45, ha='right')
    ax5.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, count in zip(bars, counts):
        ax5.text(bar.get_x() + bar.get_width()/2., count,
                f'{count}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 6. Aspect ratio comparison
    ax6 = fig.add_subplot(gs[2, 1])
    aspect_ratios = [s['resolutions']['mean'][0] / s['resolutions']['mean'][1] for s in all_stats]
    bars = ax6.bar(names, aspect_ratios, alpha=0.8, color=colors, edgecolor='black', linewidth=1.5)
    ax6.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='Target (2:1)', linewidth=2)
    ax6.set_ylabel('Width / Height', fontsize=11)
    ax6.set_title('Aspect Ratio Comparison\n종횡비 비교', fontsize=12, fontweight='bold')
    ax6.set_xticklabels(names, rotation=45, ha='right')
    ax6.legend()
    ax6.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, ratio in zip(bars, aspect_ratios):
        ax6.text(bar.get_x() + bar.get_width()/2., ratio,
                f'{ratio:.2f}:1', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.suptitle('Dataset Comparison Analysis\n데이터셋 비교 분석',
                 fontsize=16, fontweight='bold', y=0.995)

    # Save
    output_path = output_dir / 'dataset_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 비교 그래프 저장: {output_path}")

    # Save statistics to YAML
    stats_output = output_dir / 'comparison_stats.yaml'
    with open(stats_output, 'w', encoding='utf-8') as f:
        yaml.dump({'datasets': all_stats}, f, default_flow_style=False, allow_unicode=True)
    print(f"✓ 통계 저장: {stats_output}")

    plt.show()


def generate_recommendations(all_stats: List[Dict], output_dir: Path):
    """
    Generate preprocessing recommendations based on analysis
    분석 결과를 바탕으로 전처리 권장사항 생성
    """
    recommendations = []

    # Check brightness differences
    means = [s['intensity']['global_mean'] for s in all_stats]
    max_diff = max(means) - min(means)

    if max_diff > 30:  # More than 30 units difference
        recommendations.append({
            'issue': 'Large brightness difference detected',
            'issue_ko': '큰 밝기 차이 발견',
            'severity': 'HIGH',
            'description': f'Brightness difference: {max_diff:.1f} (max: {max(means):.1f}, min: {min(means):.1f})',
            'description_ko': f'밝기 차이: {max_diff:.1f} (최대: {max(means):.1f}, 최소: {min(means):.1f})',
            'solution': 'Enable brightness normalization in preprocessing',
            'solution_ko': '전처리에서 밝기 정규화 활성화 필요',
            'config': {
                'brightness_normalization': {
                    'enabled': True,
                    'target_mean': float(np.mean(means)),
                    'target_std': float(np.std(means))
                }
            }
        })

    # Check resolution differences
    resolutions = [s['resolutions']['mean'] for s in all_stats]
    res_std = np.std(resolutions, axis=0)

    if res_std[0] > 500 or res_std[1] > 250:  # Large resolution variance
        recommendations.append({
            'issue': 'Large resolution variance',
            'issue_ko': '큰 해상도 차이',
            'severity': 'MEDIUM',
            'description': f'Resolution std: width={res_std[0]:.1f}, height={res_std[1]:.1f}',
            'description_ko': f'해상도 표준편차: 너비={res_std[0]:.1f}, 높이={res_std[1]:.1f}',
            'solution': 'Use unified target resolution for all datasets',
            'solution_ko': '모든 데이터셋에 통합된 목표 해상도 사용',
            'config': {
                'resolution': {
                    'target_width': 2048,
                    'target_height': 1024
                }
            }
        })

    # Check contrast differences
    contrasts = [s['intensity']['mean_of_std'] for s in all_stats]
    contrast_diff = max(contrasts) - min(contrasts)

    if contrast_diff > 10:
        recommendations.append({
            'issue': 'Contrast variance detected',
            'issue_ko': '대비 차이 발견',
            'severity': 'MEDIUM',
            'description': f'Contrast difference: {contrast_diff:.1f}',
            'description_ko': f'대비 차이: {contrast_diff:.1f}',
            'solution': 'Enable CLAHE for contrast normalization',
            'solution_ko': 'CLAHE를 사용한 대비 정규화 활성화',
            'config': {
                'clahe': {
                    'enabled': True,
                    'clip_limit': 2.0,
                    'tile_size': [8, 8]
                }
            }
        })

    # Print recommendations
    print("\n" + "="*60)
    print("🎯 전처리 권장사항")
    print("="*60)

    if not recommendations:
        print("✓ 데이터셋들이 비교적 균일합니다. 기본 전처리만으로 충분합니다.")
    else:
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. [{rec['severity']}] {rec['issue_ko']}")
            print(f"   문제: {rec['description_ko']}")
            print(f"   해결: {rec['solution_ko']}")

    # Save recommendations
    rec_output = output_dir / 'preprocessing_recommendations.yaml'
    with open(rec_output, 'w', encoding='utf-8') as f:
        yaml.dump({'recommendations': recommendations}, f, default_flow_style=False, allow_unicode=True)
    print(f"\n✓ 권장사항 저장: {rec_output}")


if __name__ == "__main__":
    setup_logging()
    setup_matplotlib_korean()

    # 분석할 데이터셋 경로 설정
    datasets_to_compare = [
        # {
        #     'name': 'RootInfection_Train',
        #     'path': r'C:\dental_pano\rootinfection\YOLODataset\images\train'
        # },
        # {
        #     'name': 'Instance_Train',
        #     'path': r'C:\dental_pano\instance_seg_data\images\train'
        # },
        # {
        #     'name': 'BoneLevel_Train',
        #     'path': r'C:\dental_pano\bone_level_seg\test-1\train\images'
        # },
        # {
        #     'name': 'Caries_Train',
        #     'path': r'C:\dental_pano\caries\YOLODataset\images\train'
        # },
        {
            'name': 'RootInfection_Train',
            'path': r'C:\Users\dwono\OneDrive\바탕 화면\Raman_ver_0\data\preprocessed\rootinfection_train\images\images\train'
        },
        {
            'name': 'Instance_Train',
            'path': r'C:\Users\dwono\OneDrive\바탕 화면\Raman_ver_0\data\preprocessed\instance_train\images\images\train'
        },
        {
            'name': 'BoneLevel_Train',
            'path': r'C:\Users\dwono\OneDrive\바탕 화면\Raman_ver_0\data\preprocessed\bone_train\images\train\images'
        },
        {
            'name': 'Caries_Train',
            'path': r'C:\Users\dwono\OneDrive\바탕 화면\Raman_ver_0\data\preprocessed\caries_train\images\images\train'
        },
    ]

    output_dir = Path('analysis_results/dataset_comparison_preprocessed')

    print("="*60)
    print("데이터셋 비교 분석 시작")
    print("Dataset Comparison Analysis")
    print("="*60)

    # Analyze each dataset
    all_stats = []
    for dataset in datasets_to_compare:
        stats = analyze_single_dataset(Path(dataset['path']), dataset['name'])
        if stats:
            all_stats.append(stats)

    if len(all_stats) < 2:
        print("⚠️  비교할 데이터셋이 부족합니다.")
    else:
        # Compare and visualize
        compare_datasets(all_stats, output_dir)

        # Generate recommendations
        generate_recommendations(all_stats, output_dir)

    print("\n" + "="*60)
    print("분석 완료!")
    print("="*60)
