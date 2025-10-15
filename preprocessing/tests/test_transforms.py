"""
Unit tests for coordinate transformations
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from coordinate_transform import CoordinateTransformer, build_transformer_from_metadata


class TestCoordinateTransformer:
    """Test coordinate transformation functions"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.transformer = CoordinateTransformer()
        self.original_size = (2000, 1000)  # width, height
    
    def test_identity_transform(self):
        """Test that no transforms returns original coordinates"""
        bbox = [0.5, 0.5, 0.2, 0.2]  # YOLO format
        result = self.transformer.transform_yolo_bbox(bbox, self.original_size)
        
        # Should be unchanged
        np.testing.assert_array_almost_equal(result, bbox, decimal=6)
    
    def test_crop_transform(self):
        """Test ROI crop transformation"""
        # Add crop transform: crop left 100px, top 50px
        self.transformer.add_transform('roi_crop', {
            'bbox': [100, 50, 1900, 950]  # x1, y1, x2, y2
        })
        
        # Original bbox in full image
        bbox = [0.5, 0.5, 0.2, 0.2]  # center of image
        
        # Transform
        result = self.transformer.transform_yolo_bbox(bbox, self.original_size)
        
        # After crop, the center should shift
        # Original center: (1000, 500)
        # After crop: (1000-100, 500-50) = (900, 450)
        # New image size: 1800x900
        # New normalized: (900/1800, 450/900) = (0.5, 0.5)
        
        assert result[0] == pytest.approx(0.5, abs=0.01)  # x_center
        assert result[1] == pytest.approx(0.5, abs=0.01)  # y_center
    
    def test_padding_transform(self):
        """Test padding transformation"""
        self.transformer.add_transform('pad', {
            'pad': [50, 50, 100, 100]  # top, bottom, left, right
        })
        
        # Original bbox
        bbox = [0.5, 0.5, 0.2, 0.2]
        
        # Transform
        result = self.transformer.transform_yolo_bbox(bbox, self.original_size)
        
        # After padding:
        # Original center: (1000, 500)
        # After pad: (1000+100, 500+50) = (1100, 550)
        # New size: (2000+200, 1000+100) = (2200, 1100)
        # New normalized: (1100/2200, 550/1100) = (0.5, 0.5)
        
        assert result[0] == pytest.approx(0.5, abs=0.01)
        assert result[1] == pytest.approx(0.5, abs=0.01)
    
    def test_resize_transform(self):
        """Test resize transformation"""
        self.transformer.add_transform('resize', {
            'target_size': [2048, 1024],
            'scale_x': 2048 / 2000,
            'scale_y': 1024 / 1000
        })
        
        # Original bbox
        bbox = [0.5, 0.5, 0.2, 0.2]
        
        # Transform
        result = self.transformer.transform_yolo_bbox(bbox, self.original_size)
        
        # Normalized coordinates should remain the same after resize
        np.testing.assert_array_almost_equal(result, bbox, decimal=6)
    
    def test_combined_transforms(self):
        """Test multiple transforms in sequence"""
        # Crop
        self.transformer.add_transform('roi_crop', {
            'bbox': [100, 50, 1900, 950]
        })
        
        # Pad
        self.transformer.add_transform('pad', {
            'pad': [50, 50, 100, 100]
        })
        
        # Resize
        self.transformer.add_transform('resize', {
            'target_size': [2048, 1024],
            'scale_x': 2048 / 2000,
            'scale_y': 1024 / 1000
        })
        
        # Original bbox
        bbox = [0.5, 0.5, 0.2, 0.2]
        
        # Should not raise exception
        result = self.transformer.transform_yolo_bbox(bbox, self.original_size)
        
        # Result should be valid normalized coordinates
        assert 0 <= result[0] <= 1
        assert 0 <= result[1] <= 1
        assert 0 <= result[2] <= 1
        assert 0 <= result[3] <= 1
    
    def test_polygon_transform(self):
        """Test polygon transformation"""
        self.transformer.add_transform('resize', {
            'target_size': [2048, 1024],
            'scale_x': 2048 / 2000,
            'scale_y': 1024 / 1000
        })
        
        # Square polygon in normalized coords
        polygon = [
            [0.4, 0.4],
            [0.6, 0.4],
            [0.6, 0.6],
            [0.4, 0.6]
        ]
        
        result = self.transformer.transform_polygon(polygon, self.original_size)
        
        # Should remain similar after proportional resize
        np.testing.assert_array_almost_equal(result, polygon, decimal=2)
    
    def test_reset(self):
        """Test resetting transformer"""
        self.transformer.add_transform('crop', {'bbox': [0, 0, 100, 100]})
        assert len(self.transformer.transform_chain) == 1
        
        self.transformer.reset()
        assert len(self.transformer.transform_chain) == 0
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Bbox at image edge
        bbox = [0.0, 0.0, 0.1, 0.1]
        result = self.transformer.transform_yolo_bbox(bbox, self.original_size)
        assert all(0 <= x <= 1 for x in result)
        
        # Bbox at other edge
        bbox = [1.0, 1.0, 0.1, 0.1]
        result = self.transformer.transform_yolo_bbox(bbox, self.original_size)
        assert all(0 <= x <= 1 for x in result)


class TestMetadataTransformer:
    """Test building transformer from metadata"""
    
    def test_build_from_metadata(self):
        """Test building transformer from preprocessing metadata"""
        metadata = {
            'transforms': {
                'roi': {
                    'method': 'otsu',
                    'bbox': [100, 50, 1900, 950]
                },
                'aspect_ratio': {
                    'method': 'pad_height',
                    'pad': [50, 50, 0, 0]
                },
                'resolution': {
                    'target_size': [2048, 1024],
                    'scale_x': 1.024,
                    'scale_y': 1.024
                }
            }
        }
        
        original_size = (2000, 1000)
        transformer = build_transformer_from_metadata(metadata, original_size)
        
        # Should have 3 transforms
        assert len(transformer.transform_chain) == 3
        
        # Test transform works
        bbox = [0.5, 0.5, 0.2, 0.2]
        result = transformer.transform_yolo_bbox(bbox, original_size)
        
        # Should return valid coordinates
        assert all(0 <= x <= 1 for x in result)
    
    def test_metadata_without_roi(self):
        """Test metadata without ROI transform"""
        metadata = {
            'transforms': {
                'aspect_ratio': {
                    'method': 'none',
                    'pad': [0, 0, 0, 0]
                },
                'resolution': {
                    'target_size': [2048, 1024],
                    'scale_x': 1.024,
                    'scale_y': 1.024
                }
            }
        }
        
        original_size = (2000, 1000)
        transformer = build_transformer_from_metadata(metadata, original_size)
        
        # Should have only resize transform
        assert len(transformer.transform_chain) == 1


class TestTransformAccuracy:
    """Test transformation accuracy with known examples"""
    
    def test_known_transform_case_1(self):
        """Test specific known case"""
        transformer = CoordinateTransformer()
        
        # Crop 10% from each side
        transformer.add_transform('roi_crop', {
            'bbox': [200, 100, 1800, 900]  # From 2000x1000
        })
        
        # Resize to target
        transformer.add_transform('resize', {
            'target_size': [2048, 1024],
            'scale_x': 2048 / 1600,
            'scale_y': 1024 / 800
        })
        
        original_size = (2000, 1000)
        
        # Bbox in the center of original image
        # Center at (1000, 500) with size (400, 200)
        bbox = [0.5, 0.5, 0.2, 0.2]
        
        result = transformer.transform_yolo_bbox(bbox, original_size)
        
        # After crop: center should still be at center of cropped image
        # Cropped center: (1000-200, 500-100) = (800, 400) in 1600x800
        # That's (0.5, 0.5) normalized
        
        assert result[0] == pytest.approx(0.5, abs=0.05)
        assert result[1] == pytest.approx(0.5, abs=0.05)
    
    def test_conservation_of_area_ratio(self):
        """Test that bbox area ratio is approximately conserved"""
        transformer = CoordinateTransformer()
        
        # Only resize (no crop/pad)
        transformer.add_transform('resize', {
            'target_size': [2048, 1024],
            'scale_x': 2048 / 2000,
            'scale_y': 1024 / 1000
        })
        
        original_size = (2000, 1000)
        bbox = [0.5, 0.5, 0.3, 0.4]  # 30% width, 40% height
        
        result = transformer.transform_yolo_bbox(bbox, original_size)
        
        # Area ratio should be conserved
        original_area = 0.3 * 0.4
        result_area = result[2] * result[3]
        
        assert result_area == pytest.approx(original_area, rel=0.01)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v'])