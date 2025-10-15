"""
Coordinate Transformation for Labels
Automatically transforms bounding boxes and segmentation masks
"""

import numpy as np
from typing import List, Dict, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class CoordinateTransformer:
    """
    Transforms annotation coordinates based on preprocessing operations
    Supports: YOLO format, COCO format, segmentation masks

    전처리 작업에 따라 어노테이션 좌표를 변환합니다
    지원 형식: YOLO 형식, COCO 형식, 세그멘테이션 마스크
    """

    def __init__(self):
        """Initialize coordinate transformer

        좌표 변환기를 초기화합니다
        """
        self.transform_chain = []
    
    def add_transform(self, transform_type: str, params: Dict):
        """
        Add a transformation to the chain
        변환 체인에 변환 작업을 추가합니다

        Args:
            transform_type: Type of transform (roi_crop, pad, resize)
                           변환 타입 (roi_crop, pad, resize)
            params: Transform parameters
                   변환 파라미터
        """
        self.transform_chain.append({
            'type': transform_type,
            'params': params
        })
    
    def transform_yolo_bbox(self, bbox: List[float],
                            original_size: Tuple[int, int]) -> List[float]:
        """
        Transform YOLO format bounding box
        YOLO 형식의 바운딩 박스를 변환합니다

        Args:
            bbox: [x_center, y_center, width, height] in normalized coords
                  정규화된 좌표로 [중심x, 중심y, 너비, 높이]
            original_size: (width, height) of original image
                          원본 이미지의 (너비, 높이)

        Returns:
            Transformed bounding box in normalized coords
            정규화된 좌표로 변환된 바운딩 박스
        """
        # Convert to absolute coordinates / 절대 좌표로 변환
        orig_w, orig_h = original_size
        x_center = bbox[0] * orig_w
        y_center = bbox[1] * orig_h
        width = bbox[2] * orig_w
        height = bbox[3] * orig_h

        # Convert to corners / 모서리 좌표로 변환
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        # Apply transform chain / 변환 체인 적용
        x1, y1, x2, y2 = self._transform_corners(x1, y1, x2, y2, original_size)

        # Get final image size / 최종 이미지 크기 계산
        final_size = self._get_final_size(original_size)
        final_w, final_h = final_size

        # Convert back to YOLO format / YOLO 형식으로 다시 변환
        x_center = (x1 + x2) / 2 / final_w
        y_center = (y1 + y2) / 2 / final_h
        width = (x2 - x1) / final_w
        height = (y2 - y1) / final_h
        
        return [x_center, y_center, width, height]
    
    def transform_polygon(self, polygon: List[List[float]],
                          original_size: Tuple[int, int]) -> List[List[float]]:
        """
        Transform polygon/segmentation coordinates
        폴리곤/세그멘테이션 좌표를 변환합니다

        Args:
            polygon: List of [x, y] points in normalized coords
                    정규화된 좌표로 [x, y] 점들의 리스트
            original_size: (width, height) of original image
                          원본 이미지의 (너비, 높이)

        Returns:
            Transformed polygon in normalized coords
            정규화된 좌표로 변환된 폴리곤
        """
        orig_w, orig_h = original_size

        # Convert to absolute / 절대 좌표로 변환
        abs_polygon = [[x * orig_w, y * orig_h] for x, y in polygon]

        # Apply transforms / 변환 적용
        for point in abs_polygon:
            point[0], point[1] = self._transform_point(point[0], point[1], original_size)

        # Get final size and normalize / 최종 크기 계산 및 정규화
        final_size = self._get_final_size(original_size)
        final_w, final_h = final_size

        normalized_polygon = [[x / final_w, y / final_h] for x, y in abs_polygon]
        
        return normalized_polygon
    
    def transform_mask(self, mask: np.ndarray,
                       original_size: Tuple[int, int]) -> np.ndarray:
        """
        Transform binary segmentation mask
        이진 세그멘테이션 마스크를 변환합니다

        Args:
            mask: Binary mask (H, W)
                 이진 마스크 (높이, 너비)
            original_size: (width, height) of original image
                          원본 이미지의 (너비, 높이)

        Returns:
            Transformed mask
            변환된 마스크
        """
        import cv2
        
        for transform in self.transform_chain:
            if transform['type'] == 'roi_crop':
                # Crop mask / 마스크 크롭
                bbox = transform['params']['bbox']
                x1, y1, x2, y2 = bbox
                mask = mask[y1:y2, x1:x2]

            elif transform['type'] == 'pad':
                # Pad mask / 마스크 패딩
                pad = transform['params']['pad']
                top, bottom, left, right = pad
                mask = cv2.copyMakeBorder(mask, top, bottom, left, right,
                                         cv2.BORDER_CONSTANT, value=0)

            elif transform['type'] == 'resize':
                # Resize mask / 마스크 리사이즈
                target_size = transform['params']['target_size']
                target_w, target_h = target_size
                mask = cv2.resize(mask, (target_w, target_h),
                                 interpolation=cv2.INTER_NEAREST)
        
        return mask
    
    def _transform_corners(self, x1: float, y1: float, x2: float, y2: float,
                          original_size: Tuple[int, int]) -> Tuple[float, float, float, float]:
        """Apply transform chain to bounding box corners

        바운딩 박스의 모서리 좌표에 변환 체인을 적용합니다
        """
        
        for transform in self.transform_chain:
            if transform['type'] == 'roi_crop':
                # Adjust for crop / 크롭에 따른 좌표 조정
                bbox = transform['params']['bbox']
                crop_x1, crop_y1, crop_x2, crop_y2 = bbox

                x1 = x1 - crop_x1
                y1 = y1 - crop_y1
                x2 = x2 - crop_x1
                y2 = y2 - crop_y1

                # Clip to crop boundaries / 크롭 경계에 맞춰 자르기
                x1 = max(0, min(x1, crop_x2 - crop_x1))
                y1 = max(0, min(y1, crop_y2 - crop_y1))
                x2 = max(0, min(x2, crop_x2 - crop_x1))
                y2 = max(0, min(y2, crop_y2 - crop_y1))

            elif transform['type'] == 'pad':
                # Adjust for padding / 패딩에 따른 좌표 조정
                pad = transform['params']['pad']
                top, bottom, left, right = pad

                x1 = x1 + left
                x2 = x2 + left
                y1 = y1 + top
                y2 = y2 + top

            elif transform['type'] == 'resize':
                # Scale for resize / 리사이즈에 따른 스케일 조정
                scale_x = transform['params']['scale_x']
                scale_y = transform['params']['scale_y']

                x1 = x1 * scale_x
                x2 = x2 * scale_x
                y1 = y1 * scale_y
                y2 = y2 * scale_y
        
        return x1, y1, x2, y2
    
    def _transform_point(self, x: float, y: float,
                        original_size: Tuple[int, int]) -> Tuple[float, float]:
        """Apply transform chain to a single point

        단일 점에 변환 체인을 적용합니다
        """
        
        for transform in self.transform_chain:
            if transform['type'] == 'roi_crop':
                # ROI 크롭 적용
                bbox = transform['params']['bbox']
                crop_x1, crop_y1, crop_x2, crop_y2 = bbox

                x = x - crop_x1
                y = y - crop_y1

                x = max(0, min(x, crop_x2 - crop_x1))
                y = max(0, min(y, crop_y2 - crop_y1))

            elif transform['type'] == 'pad':
                # 패딩 적용
                pad = transform['params']['pad']
                top, bottom, left, right = pad

                x = x + left
                y = y + top

            elif transform['type'] == 'resize':
                # 리사이즈 스케일 적용
                scale_x = transform['params']['scale_x']
                scale_y = transform['params']['scale_y']

                x = x * scale_x
                y = y * scale_y
        
        return x, y
    
    def _get_final_size(self, original_size: Tuple[int, int]) -> Tuple[int, int]:
        """Calculate final image size after all transforms

        모든 변환 후 최종 이미지 크기를 계산합니다
        """
        w, h = original_size

        for transform in self.transform_chain:
            if transform['type'] == 'roi_crop':
                # 크롭 후 크기
                bbox = transform['params']['bbox']
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1

            elif transform['type'] == 'pad':
                # 패딩 후 크기
                pad = transform['params']['pad']
                top, bottom, left, right = pad
                w = w + left + right
                h = h + top + bottom

            elif transform['type'] == 'resize':
                # 리사이즈 후 크기
                target_size = transform['params']['target_size']
                w, h = target_size
        
        return w, h
    
    def reset(self):
        """Clear transform chain

        변환 체인을 초기화합니다
        """
        self.transform_chain = []
    
    def get_transform_matrix(self, original_size: Tuple[int, int]) -> np.ndarray:
        """
        Get cumulative transformation matrix (for visualization)
        누적 변환 행렬을 가져옵니다 (시각화용)

        Returns:
            3x3 transformation matrix
            3x3 변환 행렬
        """
        # Start with identity / 단위 행렬로 시작
        M = np.eye(3)

        for transform in self.transform_chain:
            if transform['type'] == 'roi_crop':
                bbox = transform['params']['bbox']
                x1, y1, _, _ = bbox
                # Translation matrix / 이동 행렬
                T = np.array([
                    [1, 0, -x1],
                    [0, 1, -y1],
                    [0, 0, 1]
                ])
                M = T @ M

            elif transform['type'] == 'pad':
                pad = transform['params']['pad']
                top, _, left, _ = pad
                # Translation matrix / 이동 행렬
                T = np.array([
                    [1, 0, left],
                    [0, 1, top],
                    [0, 0, 1]
                ])
                M = T @ M

            elif transform['type'] == 'resize':
                scale_x = transform['params']['scale_x']
                scale_y = transform['params']['scale_y']
                # Scale matrix / 스케일 행렬
                S = np.array([
                    [scale_x, 0, 0],
                    [0, scale_y, 0],
                    [0, 0, 1]
                ])
                M = S @ M
        
        return M


def build_transformer_from_metadata(metadata: Dict,
                                    original_size: Tuple[int, int]) -> CoordinateTransformer:
    """
    Build CoordinateTransformer from preprocessing metadata
    전처리 메타데이터로부터 CoordinateTransformer를 생성합니다

    Args:
        metadata: Preprocessing metadata dict
                 전처리 메타데이터 딕셔너리
        original_size: (width, height) of original image
                      원본 이미지의 (너비, 높이)

    Returns:
        Configured CoordinateTransformer
        설정된 CoordinateTransformer
    """
    transformer = CoordinateTransformer()

    transforms = metadata.get('transforms', {})

    # Add ROI crop if present / ROI 크롭이 있으면 추가
    if 'roi' in transforms and transforms['roi']['method'] != 'fallback':
        transformer.add_transform('roi_crop', {
            'bbox': transforms['roi']['bbox']
        })

    # Add padding if present / 패딩이 있으면 추가
    if 'aspect_ratio' in transforms:
        aspect_info = transforms['aspect_ratio']
        if aspect_info['method'] in ['pad_width', 'pad_height']:
            transformer.add_transform('pad', {
                'pad': aspect_info['pad']
            })

    # Add resize / 리사이즈 추가
    if 'resolution' in transforms:
        res_info = transforms['resolution']
        transformer.add_transform('resize', {
            'target_size': res_info['target_size'],
            'scale_x': res_info['scale_x'],
            'scale_y': res_info['scale_y']
        })

    return transformer