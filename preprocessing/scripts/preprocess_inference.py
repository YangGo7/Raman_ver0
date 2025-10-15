"""
Inference Preprocessing Script
추론용 전처리 스크립트

새로운 이미지를 학습된 모델에 넣기 전에 전처리합니다.
통합 설정을 사용하므로 1번만 전처리하면 모든 모델에 사용 가능합니다.
"""

import sys
from pathlib import Path
import argparse

# Add src to path / src 경로 추가
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from preprocessor import DentalPreprocessor
from utils import setup_logging, load_image, save_image


def preprocess_for_inference(input_path: str,
                             output_path: str = None,
                             config_path: str = "config/unified_preprocessing_config.yaml",
                             visualize: bool = False):
    """
    Preprocess a single image for inference
    추론을 위해 단일 이미지를 전처리합니다

    Args:
        input_path: Input image path / 입력 이미지 경로
        output_path: Output image path (optional) / 출력 이미지 경로 (선택)
        config_path: Preprocessing config path / 전처리 설정 파일 경로
        visualize: Whether to save visualization / 시각화 저장 여부

    Returns:
        preprocessed_image, metadata
        전처리된 이미지, 메타데이터
    """
    setup_logging()

    print(f"\n{'='*60}")
    print(f"추론용 이미지 전처리")
    print(f"Preprocessing image for inference")
    print(f"{'='*60}")
    print(f"입력: {input_path}")
    print(f"설정: {config_path}")

    # Load image / 이미지 로드
    image = load_image(input_path)
    if image is None:
        raise ValueError(f"Failed to load image: {input_path}")

    print(f"원본 크기: {image.shape[1]}x{image.shape[0]}")

    # Initialize preprocessor / 전처리기 초기화
    preprocessor = DentalPreprocessor(config_path)

    # Preprocess / 전처리 수행
    processed_image, metadata = preprocessor.preprocess(image, visualize=visualize)

    print(f"전처리 후 크기: {processed_image.shape[1]}x{processed_image.shape[0]}")
    print(f"✓ 전처리 완료!")

    # Save if output path provided / 출력 경로가 주어지면 저장
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        save_image(processed_image, str(output_file))
        print(f"✓ 저장 완료: {output_file}")

        # Save visualization if requested / 시각화 요청 시 저장
        if visualize and 'intermediate_images' in metadata:
            from utils import visualize_preprocessing_steps, compare_before_after

            viz_dir = output_file.parent / 'visualizations'
            viz_dir.mkdir(exist_ok=True)

            # Before/after comparison / 전후 비교
            compare_before_after(
                image, processed_image,
                str(viz_dir / f"{output_file.stem}_comparison.png")
            )
            print(f"✓ 시각화 저장: {viz_dir}")

    print(f"{'='*60}\n")

    return processed_image, metadata


def preprocess_batch(input_dir: str,
                    output_dir: str,
                    config_path: str = "config/unified_preprocessing_config.yaml",
                    visualize: bool = False):
    """
    Preprocess multiple images in a directory
    디렉토리 내 여러 이미지를 전처리합니다

    Args:
        input_dir: Input directory / 입력 디렉토리
        output_dir: Output directory / 출력 디렉토리
        config_path: Preprocessing config / 전처리 설정
        visualize: Save visualizations / 시각화 저장 여부
    """
    setup_logging()

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Find all images / 모든 이미지 찾기
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(list(input_path.rglob(f'*{ext}')))

    print(f"\n{'='*60}")
    print(f"배치 전처리")
    print(f"Batch Preprocessing")
    print(f"{'='*60}")
    print(f"입력 디렉토리: {input_dir}")
    print(f"출력 디렉토리: {output_dir}")
    print(f"찾은 이미지: {len(image_paths)}개")
    print(f"{'='*60}\n")

    if len(image_paths) == 0:
        print("⚠️  이미지를 찾을 수 없습니다!")
        return

    # Process each image / 각 이미지 처리
    success_count = 0
    for i, img_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] 처리 중: {img_path.name}")

        try:
            # Determine output path / 출력 경로 결정
            rel_path = img_path.relative_to(input_path)
            out_path = output_path / rel_path

            # Preprocess / 전처리
            preprocess_for_inference(
                str(img_path),
                str(out_path),
                config_path,
                visualize=(i <= 5)  # First 5 only / 처음 5개만 시각화
            )
            success_count += 1

        except Exception as e:
            print(f"⚠️  Error: {e}")
            continue

    print(f"\n{'='*60}")
    print(f"배치 전처리 완료!")
    print(f"Batch Preprocessing Complete")
    print(f"{'='*60}")
    print(f"성공: {success_count}/{len(image_paths)}")
    print(f"출력: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Preprocess images for inference / 추론용 이미지 전처리'
    )

    # Mode selection / 모드 선택
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'batch'],
                       help='Processing mode: single image or batch / 처리 모드: 단일 또는 배치')

    # Single mode arguments / 단일 모드 인자
    parser.add_argument('--input', type=str,
                       help='Input image path / 입력 이미지 경로')
    parser.add_argument('--output', type=str,
                       help='Output image path / 출력 이미지 경로')

    # Batch mode arguments / 배치 모드 인자
    parser.add_argument('--input_dir', type=str,
                       help='Input directory for batch mode / 배치 모드 입력 디렉토리')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory for batch mode / 배치 모드 출력 디렉토리')

    # Common arguments / 공통 인자
    parser.add_argument('--config', type=str,
                       default='config/unified_preprocessing_config.yaml',
                       help='Preprocessing config file / 전처리 설정 파일')
    parser.add_argument('--visualize', action='store_true',
                       help='Save visualization / 시각화 저장')

    args = parser.parse_args()

    # Execute based on mode / 모드에 따라 실행
    if args.mode == 'single':
        if not args.input:
            print("❌ Error: --input is required for single mode")
            print("예시: python preprocess_inference.py --mode single --input test.png --output output.png")
            sys.exit(1)

        preprocess_for_inference(
            args.input,
            args.output,
            args.config,
            args.visualize
        )

    elif args.mode == 'batch':
        if not args.input_dir or not args.output_dir:
            print("❌ Error: --input_dir and --output_dir are required for batch mode")
            print("예시: python preprocess_inference.py --mode batch --input_dir ./test_images --output_dir ./preprocessed")
            sys.exit(1)

        preprocess_batch(
            args.input_dir,
            args.output_dir,
            args.config,
            args.visualize
        )
