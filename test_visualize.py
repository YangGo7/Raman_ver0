"""
전처리된 이미지와 라벨 시각화 스크립트
Visualize preprocessed images with their labels to verify alignment
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import platform

def setup_matplotlib_korean():
    """한글 폰트 설정"""
    if platform.system() == 'Windows':
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
    plt.rc('axes', unicode_minus=False)


def load_image_korean_path(image_path):
    """한글 경로를 지원하는 이미지 로딩"""
    try:
        with open(image_path, 'rb') as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        return image
    except Exception as e:
        print(f"❌ 이미지 로딩 실패: {image_path}")
        print(f"   에러: {e}")
        return None


def load_yolo_label(label_path):
    """YOLO 포맷 라벨 파일 로드"""
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        annotations = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                annotations.append({
                    'class_id': class_id,
                    'bbox': [x_center, y_center, width, height]
                })

        return annotations
    except Exception as e:
        print(f"❌ 라벨 로딩 실패: {label_path}")
        print(f"   에러: {e}")
        return []


def draw_yolo_annotations(image, annotations, class_names=None):
    """
    이미지에 YOLO 어노테이션 그리기

    Args:
        image: 그레이스케일 이미지
        annotations: YOLO 포맷 어노테이션 리스트
        class_names: 클래스 이름 딕셔너리
    """
    # 그레이스케일을 BGR로 변환 (컬러로 박스 그리기 위해)
    if len(image.shape) == 2:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_image = image.copy()

    h, w = image.shape[:2]

    # 클래스별 색상
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]

    for ann in annotations:
        class_id = ann['class_id']
        x_center, y_center, bbox_w, bbox_h = ann['bbox']

        # YOLO normalized coordinates -> pixel coordinates
        x1 = int((x_center - bbox_w / 2) * w)
        y1 = int((y_center - bbox_h / 2) * h)
        x2 = int((x_center + bbox_w / 2) * w)
        y2 = int((y_center + bbox_h / 2) * h)

        # 색상 선택
        color = colors[class_id % len(colors)]

        # 박스 그리기
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

        # 클래스 레이블 텍스트
        if class_names and class_id in class_names:
            label = class_names[class_id]
        else:
            label = f"Class {class_id}"

        # 텍스트 배경 그리기
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(vis_image, (x1, y1 - text_size[1] - 10),
                     (x1 + text_size[0], y1), color, -1)

        # 텍스트 그리기
        cv2.putText(vis_image, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return vis_image


def visualize_dataset_samples(dataset_path, num_samples=9, output_dir='visualization_results'):
    """
    데이터셋에서 샘플을 선택하여 시각화

    Args:
        dataset_path: 전처리된 데이터셋 경로
        num_samples: 시각화할 샘플 수
        output_dir: 결과 저장 디렉토리
    """
    dataset_path = Path(dataset_path)

    # 이미지와 라벨 경로 찾기
    images_dir = dataset_path / 'images' / 'images' / 'train'
    labels_dir = dataset_path / 'images' / 'images' / 'labels'

    if not images_dir.exists():
        print(f"❌ 이미지 디렉토리를 찾을 수 없습니다: {images_dir}")
        return

    if not labels_dir.exists():
        print(f"❌ 라벨 디렉토리를 찾을 수 없습니다: {labels_dir}")
        return

    # 이미지 파일 목록
    image_files = list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg'))

    if len(image_files) == 0:
        print(f"❌ 이미지 파일을 찾을 수 없습니다: {images_dir}")
        return

    print(f"\n✓ 총 {len(image_files)}개의 이미지 발견")

    # 랜덤 샘플 선택
    sample_images = random.sample(image_files, min(num_samples, len(image_files)))

    # 시각화 준비
    rows = int(np.ceil(np.sqrt(num_samples)))
    cols = int(np.ceil(num_samples / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    if num_samples == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # 각 샘플 시각화
    for idx, img_path in enumerate(sample_images):
        # 이미지 로드
        image = load_image_korean_path(str(img_path))

        if image is None:
            axes[idx].text(0.5, 0.5, 'Failed to load', ha='center', va='center')
            axes[idx].axis('off')
            continue

        # 대응하는 라벨 파일 찾기
        label_path = labels_dir / f"{img_path.stem}.txt"

        annotations = []
        if label_path.exists():
            annotations = load_yolo_label(str(label_path))

        # 어노테이션 그리기
        vis_image = draw_yolo_annotations(image, annotations)

        # RGB로 변환 (matplotlib 표시용)
        if len(vis_image.shape) == 3:
            vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        else:
            vis_image_rgb = vis_image

        # 플롯
        axes[idx].imshow(vis_image_rgb, cmap='gray' if len(vis_image_rgb.shape) == 2 else None)
        axes[idx].set_title(f"{img_path.name}\n{len(annotations)} labels", fontsize=10)
        axes[idx].axis('off')

    # 빈 subplot 숨기기
    for idx in range(len(sample_images), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    # 결과 저장
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_name = dataset_path.name
    output_file = output_path / f"{dataset_name}_visualization.png"

    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ 시각화 저장 완료: {output_file}")

    plt.show()
    plt.close()


def compare_original_vs_preprocessed(original_img_path, preprocessed_img_path,
                                     label_path, output_path='comparison.png'):
    """
    원본과 전처리된 이미지 비교 시각화

    Args:
        original_img_path: 원본 이미지 경로
        preprocessed_img_path: 전처리된 이미지 경로
        label_path: 라벨 파일 경로
        output_path: 저장 경로
    """
    # 이미지 로드
    original = load_image_korean_path(str(original_img_path))
    preprocessed = load_image_korean_path(str(preprocessed_img_path))

    if original is None or preprocessed is None:
        print("❌ 이미지를 로드할 수 없습니다")
        return

    # 라벨 로드
    annotations = load_yolo_label(str(label_path))

    # 어노테이션 그리기
    original_vis = draw_yolo_annotations(original, annotations)
    preprocessed_vis = draw_yolo_annotations(preprocessed, annotations)

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # 원본
    axes[0].imshow(cv2.cvtColor(original_vis, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'원본 이미지\n{original.shape[1]}x{original.shape[0]}', fontsize=14)
    axes[0].axis('off')

    # 전처리됨
    axes[1].imshow(cv2.cvtColor(preprocessed_vis, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'전처리된 이미지\n{preprocessed.shape[1]}x{preprocessed.shape[0]}', fontsize=14)
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 비교 시각화 저장 완료: {output_path}")
    plt.show()
    plt.close()


if __name__ == "__main__":
    # 한글 폰트 설정
    setup_matplotlib_korean()

    print("=" * 60)
    print("전처리된 데이터셋 시각화")
    print("=" * 60)

    # 전처리된 데이터셋 경로들
    base_path = Path(r"C:\Users\dwono\OneDrive\바탕 화면\Raman_ver_0\data\preprocessed")

    datasets = [
        # 'rootinfection_train',
        # 'rootinfection_val',
        'instance_train',
        'instance_val',
        # 'caries_train',
        # 'caries_val',
        # 'bone_train',
        # 'bone_val'
    ]

    # 결과 저장 디렉토리
    output_dir = Path(r"C:\Users\dwono\OneDrive\바탕 화면\Raman_ver_0\visualization_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 각 데이터셋 시각화
    for dataset_name in datasets:
        dataset_path = base_path / dataset_name

        if not dataset_path.exists():
            print(f"\n⚠️ 데이터셋을 찾을 수 없습니다: {dataset_name}")
            continue

        print(f"\n{'=' * 60}")
        print(f"데이터셋 시각화: {dataset_name}")
        print(f"{'=' * 60}")

        visualize_dataset_samples(
            dataset_path=dataset_path,
            num_samples=9,
            output_dir=output_dir
        )

    print("\n" + "=" * 60)
    print("✓ 모든 시각화 완료!")
    print(f"✓ 결과 저장 위치: {output_dir}")
    print("=" * 60)
