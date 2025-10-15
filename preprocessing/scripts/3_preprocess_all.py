"""
Script 3: Preprocess All Images
스크립트 3: 모든 이미지 전처리

Apply preprocessing pipeline to entire dataset with label transformation
전체 데이터셋에 전처리 파이프라인을 적용하고 라벨도 함께 변환합니다
"""

import sys
from pathlib import Path
import yaml
import json
from tqdm import tqdm
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from preprocessor import DentalPreprocessor
from coordinate_transform import build_transformer_from_metadata
from label_handler import LabelHandler
from utils import (
    setup_logging, load_image, save_image, 
    visualize_preprocessing_steps, visualize_annotations,
    create_dataset_split, compare_before_after
)


def process_single_image(args):
    """
    Process a single image and its labels
    단일 이미지와 라벨을 전처리합니다

    Args:
        args: Tuple of (image_path, label_path, config_path, output_dir, task_name, visualize)
              (이미지 경로, 라벨 경로, 설정 경로, 출력 디렉토리, 태스크명, 시각화 여부)

    Returns:
        Success status and message
        성공 여부 및 메시지
    """
    image_path, label_path, config_path, output_dir, task_name, visualize = args
    
    try:
        # Initialize (each process needs its own instances)
        preprocessor = DentalPreprocessor(config_path)
        label_handler = LabelHandler(format_type='yolo')
        
        # Load image
        image = load_image(str(image_path))
        if image is None:
            return False, f"Failed to load: {image_path}"
        
        original_size = (image.shape[1], image.shape[0])  # (width, height)
        
        # Preprocess image
        processed_image, metadata = preprocessor.preprocess(image, visualize=visualize)
        
        # Save processed image
        rel_path = Path(image_path).relative_to(Path(image_path).parents[2])  # Get relative path
        output_image_path = Path(output_dir) / task_name / 'images' / rel_path
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(processed_image, str(output_image_path))
        
        # Process labels if they exist
        if label_path is not None and label_path.exists():
            # Read original labels
            annotations = label_handler.read_label(str(label_path))
            
            if len(annotations) > 0:
                # Build coordinate transformer
                transformer = build_transformer_from_metadata(metadata, original_size)
                
                # Transform annotations
                transformed_annotations = label_handler.transform_annotations(
                    annotations, transformer, original_size
                )
                
                # Validate transformed annotations
                final_size = (processed_image.shape[1], processed_image.shape[0])
                valid_annotations, issues = label_handler.validate_annotations(
                    transformed_annotations, final_size
                )
                
                if len(issues) > 0:
                    print(f"⚠️  Validation issues for {image_path.name}: {issues}")
                
                # Save transformed labels
                output_label_path = output_image_path.parent.parent / 'labels' / f"{output_image_path.stem}.txt"
                output_label_path.parent.mkdir(parents=True, exist_ok=True)
                label_handler.write_label(valid_annotations, str(output_label_path))
                
                # Save original labels backup
                backup_label_path = output_image_path.parent.parent / 'labels_original' / f"{output_image_path.stem}.txt"
                backup_label_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(label_path, backup_label_path)
        
        # Save metadata
        metadata_path = output_image_path.parent.parent / 'metadata' / f"{output_image_path.stem}.yaml"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, 'w') as f:
            # Convert numpy types to Python types for YAML
            metadata_clean = {
                'original_shape': list(metadata['original_shape']),
                'final_shape': list(metadata['final_shape']),
                'transforms': metadata['transforms']
            }
            yaml.dump(metadata_clean, f, default_flow_style=False)
        
        # Visualization (only for first few images)
        if visualize:
            viz_dir = Path(output_dir) / task_name / 'visualizations'
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            # Before/after comparison
            compare_before_after(
                image, processed_image,
                str(viz_dir / f"{output_image_path.stem}_comparison.png")
            )
            
            # Preprocessing steps
            if 'intermediate_images' in metadata:
                visualize_preprocessing_steps(
                    metadata['intermediate_images'],
                    str(viz_dir / f"{output_image_path.stem}_steps.png")
                )
            
            # Annotations (if available)
            if label_path is not None and label_path.exists():
                annotations = label_handler.read_label(str(output_label_path))
                if len(annotations) > 0:
                    visualize_annotations(
                        processed_image, annotations,
                        str(viz_dir / f"{output_image_path.stem}_annotations.png"),
                        format_type='yolo'
                    )
        
        return True, f"Processed: {image_path.name}"
    
    except Exception as e:
        return False, f"Error processing {image_path}: {str(e)}"


def preprocess_dataset(data_dir: str,
                      output_dir: str = "data/preprocessed",
                      config_path: str = "config/preprocessing_config.yaml",
                      tasks: list = None,
                      num_workers: int = None,
                      visualize_samples: int = 5):
    """
    Preprocess entire dataset for all tasks
    
    Args:
        data_dir: Root directory containing task subdirectories
        output_dir: Output directory for preprocessed data
        config_path: Path to preprocessing config
        tasks: List of task names (tooth_seg, periodontal, findings)
        num_workers: Number of parallel workers (None = auto)
        visualize_samples: Number of samples to visualize per task
    """
    setup_logging()
    
    if tasks is None:
        tasks = ['tooth_seg', 'periodontal', 'findings']
    
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    
    print(f"\nPreprocessing Pipeline Started")
    print(f"Tasks: {tasks}")
    print(f"Workers: {num_workers}")
    print(f"Output: {output_dir}")
    print("="*60)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Copy config to output directory
    shutil.copy(config_path, Path(output_dir) / 'preprocessing_config.yaml')
    
    # Process each task
    all_stats = {}
    
    for task_name in tasks:
        print(f"\n{'='*60}")
        print(f"Processing task: {task_name}")
        print(f"{'='*60}")
        
        task_dir = Path(data_dir) / task_name
        if not task_dir.exists():
            print(f"⚠️  Task directory not found: {task_dir}")
            continue
        
        # Find all images and labels
        image_dir = task_dir / 'images'
        label_dir = task_dir / 'labels'
        
        if not image_dir.exists():
            print(f"⚠️  Images directory not found: {image_dir}")
            continue
        
        # Get all images
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(list(image_dir.rglob(f'*{ext}')))
        
        print(f"Found {len(image_paths)} images")
        
        # Prepare processing arguments
        process_args = []
        for i, img_path in enumerate(image_paths):
            # Find corresponding label
            label_path = label_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                label_path = None
            
            # Visualize only first N samples
            visualize = (i < visualize_samples)
            
            process_args.append((
                img_path, label_path, config_path, 
                output_dir, task_name, visualize
            ))
        
        # Process in parallel
        print(f"\nProcessing {len(process_args)} images...")
        
        success_count = 0
        failure_count = 0
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_single_image, args) for args in process_args]
            
            # Progress bar
            with tqdm(total=len(futures), desc=f"{task_name}") as pbar:
                for future in as_completed(futures):
                    success, message = future.result()
                    if success:
                        success_count += 1
                    else:
                        failure_count += 1
                        print(f"\n⚠️  {message}")
                    pbar.update(1)
        
        # Statistics
        task_stats = {
            'total_images': len(image_paths),
            'success': success_count,
            'failure': failure_count,
            'success_rate': success_count / len(image_paths) if len(image_paths) > 0 else 0
        }
        all_stats[task_name] = task_stats
        
        print(f"\n✓ {task_name} complete:")
        print(f"  Success: {success_count}/{len(image_paths)}")
        print(f"  Failure: {failure_count}/{len(image_paths)}")
        print(f"  Success rate: {task_stats['success_rate']*100:.1f}%")
    
    # Save overall statistics
    stats_path = Path(output_dir) / 'preprocessing_stats.yaml'
    with open(stats_path, 'w') as f:
        yaml.dump(all_stats, f, default_flow_style=False)
    
    print(f"\n{'='*60}")
    print("PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Statistics: {stats_path}")
    
    for task_name, stats in all_stats.items():
        print(f"\n{task_name}:")
        print(f"  Total: {stats['total_images']}")
        print(f"  Success: {stats['success']} ({stats['success_rate']*100:.1f}%)")
        print(f"  Failure: {stats['failure']}")
    
    print("\nNext steps:")
    print("1. Review visualizations in: <output_dir>/<task>/visualizations/")
    print("2. Validate results with: python scripts/4_validate.py")
    print("3. Start training with preprocessed data")
    print("="*60)


if __name__ == "__main__":
    # 여기에 전처리할 이미지 폴더 절대경로를 지정하세요
    # Specify absolute paths to image folders here
    # 통합 전처리 설정 사용 / Use unified preprocessing config
    # 모든 데이터셋을 동일한 형식으로 표준화 / Standardize all datasets to same format
    UNIFIED_CONFIG = 'config/unified_preprocessing_config.yaml'

    data_configs = [
        {
            'image_dir': r"C:\dental_pano\rootinfection\YOLODataset\images\train",
            'label_dir': r"C:\dental_pano\rootinfection\YOLODataset\labels\train",
            'task_name': 'rootinfection_train',
            'config_path': UNIFIED_CONFIG,  # 통합 설정 사용
        },
        {
            'image_dir': r"C:\dental_pano\rootinfection\YOLODataset\images\val",
            'label_dir': r"C:\dental_pano\rootinfection\YOLODataset\labels\val",
            'task_name': 'rootinfection_val',
            'config_path': UNIFIED_CONFIG,  # 통합 설정 사용
        },
        {
            'image_dir': r"C:\dental_pano\instance_seg_data\images\train",
            'label_dir': r"C:\dental_pano\instance_seg_data\labels\train",
            'task_name': 'instance_train',
            'config_path': UNIFIED_CONFIG,  # 통합 설정 사용
        },
        {
            'image_dir': r"C:\dental_pano\instance_seg_data\images\val",
            'label_dir': r"C:\dental_pano\instance_seg_data\labels\val",
            'task_name': 'instance_val',
            'config_path': UNIFIED_CONFIG,  # 통합 설정 사용
        },
        {
            'image_dir': r"C:\dental_pano\bone_level_seg\test-1\train\images",
            'label_dir': r"C:\dental_pano\bone_level_seg\test-1\train\labels",
            'task_name': 'bone_train',
            'config_path': UNIFIED_CONFIG,  # 통합 설정 사용
        },
        {
            'image_dir': r"C:\dental_pano\bone_level_seg\test-1\valid\images",
            'label_dir': r"C:\dental_pano\bone_level_seg\test-1\valid\labels",
            'task_name': 'bone_val',
            'config_path': UNIFIED_CONFIG,  # 통합 설정 사용
        },
        {
            'image_dir': r"C:\dental_pano\caries\YOLODataset\images\train",
            'label_dir': r"C:\dental_pano\caries\YOLODataset\labels\train",
            'task_name': 'caries_train',
            'config_path': UNIFIED_CONFIG,  # 통합 설정 사용
        },
        {
            'image_dir': r"C:\dental_pano\caries\YOLODataset\images\val",
            'label_dir': r"C:\dental_pano\caries\YOLODataset\labels\val",
            'task_name': 'caries_val',
            'config_path': UNIFIED_CONFIG,  # 통합 설정 사용
        },
    ]

    # 설정
    # Settings
    base_output_dir = "data/preprocessed"  # 기본 출력 디렉토리
    num_workers = None  # None = 자동 (CPU 코어 수 - 1)
    visualize_samples = 5  # 태스크당 시각화할 샘플 수

    setup_logging()

    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    print("\n" + "="*60)
    print("전처리 파이프라인 시작")
    print("Preprocessing Pipeline Started")
    print("="*60)
    print(f"총 {len(data_configs)}개 데이터셋 처리")
    print(f"병렬 작업자: {num_workers}")
    print(f"출력 디렉토리: {base_output_dir}")
    print("="*60)

    # 전체 통계
    all_stats = {}

    # 각 데이터셋 처리
    for config in data_configs:
        image_dir = Path(config['image_dir'])
        label_dir = Path(config['label_dir']) if config.get('label_dir') else None
        task_name = config['task_name']
        config_path = config.get('config_path', 'config/preprocessing_config.yaml')

        print(f"\n{'='*60}")
        print(f"처리 중: {task_name}")
        print(f"Processing: {task_name}")
        print(f"이미지 디렉토리: {image_dir}")
        print(f"라벨 디렉토리: {label_dir}")
        print(f"{'='*60}")

        # 이미지 디렉토리 확인
        if not image_dir.exists():
            print(f"⚠️  이미지 디렉토리를 찾을 수 없습니다: {image_dir}")
            continue

        # 설정 파일 확인 (없으면 기본 설정 사용)
        if not Path(config_path).exists():
            print(f"⚠️  설정 파일 없음, 기본 설정 사용: config/preprocessing_config.yaml")
            config_path = 'config/preprocessing_config.yaml'

        # 모든 이미지 찾기
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        image_paths = []
        for ext in image_extensions:
            found = list(image_dir.rglob(f'*{ext}'))
            image_paths.extend(found)

        print(f"찾은 이미지: {len(image_paths)}개")

        if len(image_paths) == 0:
            print(f"⚠️  이미지를 찾을 수 없습니다!")
            continue

        # 처리 인자 준비
        process_args = []
        for i, img_path in enumerate(image_paths):
            # 대응하는 라벨 찾기
            if label_dir and label_dir.exists():
                label_path = label_dir / f"{img_path.stem}.txt"
                if not label_path.exists():
                    label_path = None
            else:
                label_path = None

            # 처음 N개만 시각화
            visualize = (i < visualize_samples)

            process_args.append((
                img_path, label_path, config_path,
                base_output_dir, task_name, visualize
            ))

        # 병렬 처리
        print(f"\n{len(process_args)}개 이미지 처리 중...")

        success_count = 0
        failure_count = 0

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_single_image, args) for args in process_args]

            # 진행률 표시
            with tqdm(total=len(futures), desc=f"{task_name}") as pbar:
                for future in as_completed(futures):
                    success, message = future.result()
                    if success:
                        success_count += 1
                    else:
                        failure_count += 1
                        print(f"\n⚠️  {message}")
                    pbar.update(1)

        # 통계
        task_stats = {
            'total_images': len(image_paths),
            'success': success_count,
            'failure': failure_count,
            'success_rate': success_count / len(image_paths) if len(image_paths) > 0 else 0
        }
        all_stats[task_name] = task_stats

        print(f"\n✓ {task_name} 완료:")
        print(f"  성공: {success_count}/{len(image_paths)}")
        print(f"  실패: {failure_count}/{len(image_paths)}")
        print(f"  성공률: {task_stats['success_rate']*100:.1f}%")

    # 전체 통계 저장
    stats_path = Path(base_output_dir) / 'preprocessing_stats.yaml'
    with open(stats_path, 'w') as f:
        yaml.dump(all_stats, f, default_flow_style=False)

    print(f"\n{'='*60}")
    print("전처리 완료!")
    print("PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"출력 디렉토리: {base_output_dir}")
    print(f"통계 파일: {stats_path}")

    for task_name, stats in all_stats.items():
        print(f"\n{task_name}:")
        print(f"  전체: {stats['total_images']}")
        print(f"  성공: {stats['success']} ({stats['success_rate']*100:.1f}%)")
        print(f"  실패: {stats['failure']}")

    print("\n다음 단계:")
    print("Next steps:")
    print("1. 시각화 결과 확인: <output_dir>/<task>/visualizations/")
    print("2. 검증 실행: python scripts/4_validate.py")
    print("3. 전처리된 데이터로 학습 시작")
    print("="*60)