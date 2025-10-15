"""
Label Handler for Dental X-ray Annotations
Supports YOLO, COCO, and Pascal VOC formats
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import xml.etree.ElementTree as ET
import logging

logger = logging.getLogger(__name__)


class LabelHandler:
    """
    Handle reading, transforming, and writing labels
    Supports multiple formats and tasks
    """
    
    def __init__(self, format_type: str = 'yolo'):
        """
        Initialize label handler
        
        Args:
            format_type: 'yolo', 'coco', or 'pascal_voc'
        """
        self.format_type = format_type
    
    def read_label(self, label_path: str, image_size: Optional[Tuple[int, int]] = None) -> List[Dict]:
        """
        Read label file and parse annotations
        
        Args:
            label_path: Path to label file
            image_size: (width, height) for formats that need it
            
        Returns:
            List of annotation dicts with keys: class_id, bbox/polygon, confidence
        """
        if self.format_type == 'yolo':
            return self._read_yolo(label_path)
        elif self.format_type == 'coco':
            return self._read_coco(label_path)
        elif self.format_type == 'pascal_voc':
            return self._read_pascal_voc(label_path, image_size)
        else:
            raise ValueError(f"Unsupported format: {self.format_type}")
    
    def write_label(self, annotations: List[Dict], output_path: str, 
                    image_size: Optional[Tuple[int, int]] = None):
        """
        Write annotations to file
        
        Args:
            annotations: List of annotation dicts
            output_path: Path to save label file
            image_size: (width, height) for formats that need it
        """
        if self.format_type == 'yolo':
            self._write_yolo(annotations, output_path)
        elif self.format_type == 'coco':
            self._write_coco(annotations, output_path)
        elif self.format_type == 'pascal_voc':
            self._write_pascal_voc(annotations, output_path, image_size)
        else:
            raise ValueError(f"Unsupported format: {self.format_type}")
    
    def _read_yolo(self, label_path: str) -> List[Dict]:
        """
        Read YOLO format label
        Format: class_id x_center y_center width height [polygon_points...]
        """
        annotations = []
        
        if not Path(label_path).exists():
            logger.warning(f"Label file not found: {label_path}")
            return annotations
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                class_id = int(parts[0])
                
                # Check if it's segmentation (more than 5 values)
                if len(parts) > 5:
                    # Segmentation format
                    polygon = []
                    coords = [float(x) for x in parts[1:]]
                    for i in range(0, len(coords), 2):
                        if i + 1 < len(coords):
                            polygon.append([coords[i], coords[i + 1]])
                    
                    annotation = {
                        'class_id': class_id,
                        'type': 'segmentation',
                        'polygon': polygon
                    }
                else:
                    # Bounding box format
                    bbox = [float(x) for x in parts[1:5]]
                    annotation = {
                        'class_id': class_id,
                        'type': 'bbox',
                        'bbox': bbox  # [x_center, y_center, width, height]
                    }
                
                annotations.append(annotation)
        
        return annotations
    
    def _write_yolo(self, annotations: List[Dict], output_path: str):
        """Write annotations in YOLO format"""
        with open(output_path, 'w') as f:
            for ann in annotations:
                class_id = ann['class_id']
                
                if ann['type'] == 'bbox':
                    bbox = ann['bbox']
                    line = f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
                    f.write(line)
                
                elif ann['type'] == 'segmentation':
                    polygon = ann['polygon']
                    coords = []
                    for point in polygon:
                        coords.extend([f"{point[0]:.6f}", f"{point[1]:.6f}"])
                    line = f"{class_id} " + " ".join(coords) + "\n"
                    f.write(line)
    
    def _read_coco(self, json_path: str) -> List[Dict]:
        """Read COCO format JSON"""
        with open(json_path, 'r') as f:
            coco_data = json.load(f)
        
        annotations = []
        for ann in coco_data.get('annotations', []):
            if 'bbox' in ann:
                # COCO bbox: [x, y, width, height] (absolute)
                bbox = ann['bbox']
                annotations.append({
                    'class_id': ann['category_id'],
                    'type': 'bbox',
                    'bbox': bbox,
                    'id': ann.get('id'),
                    'image_id': ann.get('image_id')
                })
            
            if 'segmentation' in ann:
                # COCO segmentation: list of polygons
                for seg in ann['segmentation']:
                    polygon = []
                    for i in range(0, len(seg), 2):
                        polygon.append([seg[i], seg[i + 1]])
                    
                    annotations.append({
                        'class_id': ann['category_id'],
                        'type': 'segmentation',
                        'polygon': polygon,
                        'id': ann.get('id'),
                        'image_id': ann.get('image_id')
                    })
        
        return annotations
    
    def _write_coco(self, annotations: List[Dict], output_path: str):
        """Write COCO format JSON"""
        coco_data = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        for i, ann in enumerate(annotations):
            coco_ann = {
                'id': ann.get('id', i),
                'image_id': ann.get('image_id', 0),
                'category_id': ann['class_id'],
                'iscrowd': 0
            }
            
            if ann['type'] == 'bbox':
                coco_ann['bbox'] = ann['bbox']
                coco_ann['area'] = ann['bbox'][2] * ann['bbox'][3]
            
            elif ann['type'] == 'segmentation':
                polygon = ann['polygon']
                seg = []
                for point in polygon:
                    seg.extend(point)
                coco_ann['segmentation'] = [seg]
            
            coco_data['annotations'].append(coco_ann)
        
        with open(output_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
    
    def _read_pascal_voc(self, xml_path: str, image_size: Tuple[int, int]) -> List[Dict]:
        """Read Pascal VOC XML format"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        annotations = []
        
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            bndbox = obj.find('bndbox')
            
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            annotations.append({
                'class_name': class_name,
                'type': 'bbox',
                'bbox': [xmin, ymin, xmax, ymax]  # absolute coordinates
            })
        
        return annotations
    
    def _write_pascal_voc(self, annotations: List[Dict], output_path: str, 
                          image_size: Tuple[int, int]):
        """Write Pascal VOC XML format"""
        root = ET.Element('annotation')
        
        size = ET.SubElement(root, 'size')
        ET.SubElement(size, 'width').text = str(image_size[0])
        ET.SubElement(size, 'height').text = str(image_size[1])
        ET.SubElement(size, 'depth').text = '1'
        
        for ann in annotations:
            if ann['type'] == 'bbox':
                obj = ET.SubElement(root, 'object')
                ET.SubElement(obj, 'name').text = ann.get('class_name', str(ann['class_id']))
                
                bndbox = ET.SubElement(obj, 'bndbox')
                bbox = ann['bbox']
                ET.SubElement(bndbox, 'xmin').text = str(int(bbox[0]))
                ET.SubElement(bndbox, 'ymin').text = str(int(bbox[1]))
                ET.SubElement(bndbox, 'xmax').text = str(int(bbox[2]))
                ET.SubElement(bndbox, 'ymax').text = str(int(bbox[3]))
        
        tree = ET.ElementTree(root)
        tree.write(output_path)
    
    def transform_annotations(self, annotations: List[Dict], 
                            transformer, 
                            original_size: Tuple[int, int]) -> List[Dict]:
        """
        Transform all annotations using CoordinateTransformer
        
        Args:
            annotations: List of annotation dicts
            transformer: CoordinateTransformer instance
            original_size: (width, height) of original image
            
        Returns:
            Transformed annotations
        """
        transformed = []
        
        for ann in annotations:
            transformed_ann = ann.copy()
            
            if ann['type'] == 'bbox':
                # Transform bounding box
                bbox = ann['bbox']
                
                # Normalize if needed (for YOLO)
                if self.format_type == 'yolo':
                    # Already normalized
                    new_bbox = transformer.transform_yolo_bbox(bbox, original_size)
                else:
                    # Convert to YOLO format, transform, convert back
                    w, h = original_size
                    x_center = (bbox[0] + bbox[2]) / 2 / w
                    y_center = (bbox[1] + bbox[3]) / 2 / h
                    width = (bbox[2] - bbox[0]) / w
                    height = (bbox[3] - bbox[1]) / h
                    
                    yolo_bbox = [x_center, y_center, width, height]
                    new_yolo_bbox = transformer.transform_yolo_bbox(yolo_bbox, original_size)
                    
                    # Convert back to absolute
                    final_size = transformer._get_final_size(original_size)
                    fw, fh = final_size
                    x_center = new_yolo_bbox[0] * fw
                    y_center = new_yolo_bbox[1] * fh
                    width = new_yolo_bbox[2] * fw
                    height = new_yolo_bbox[3] * fh
                    
                    new_bbox = [
                        x_center - width / 2,
                        y_center - height / 2,
                        x_center + width / 2,
                        y_center + height / 2
                    ]
                
                transformed_ann['bbox'] = new_bbox
            
            elif ann['type'] == 'segmentation':
                # Transform polygon
                polygon = ann['polygon']
                new_polygon = transformer.transform_polygon(polygon, original_size)
                transformed_ann['polygon'] = new_polygon
            
            transformed.append(transformed_ann)
        
        return transformed
    
    def validate_annotations(self, annotations: List[Dict], 
                           image_size: Tuple[int, int]) -> Tuple[List[Dict], List[str]]:
        """
        Validate annotations and remove invalid ones
        
        Args:
            annotations: List of annotations
            image_size: (width, height) of image
            
        Returns:
            valid_annotations: Filtered annotations
            issues: List of validation issues
        """
        valid = []
        issues = []
        
        w, h = image_size
        
        for i, ann in enumerate(annotations):
            is_valid = True
            
            if ann['type'] == 'bbox':
                bbox = ann['bbox']
                
                if self.format_type == 'yolo':
                    # YOLO: normalized coordinates
                    x_center, y_center, width, height = bbox
                    
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                        issues.append(f"Annotation {i}: bbox center out of bounds")
                        is_valid = False
                    
                    if width <= 0 or height <= 0 or width > 1 or height > 1:
                        issues.append(f"Annotation {i}: invalid bbox dimensions")
                        is_valid = False
                else:
                    # Absolute coordinates
                    x1, y1, x2, y2 = bbox
                    
                    if x1 >= x2 or y1 >= y2:
                        issues.append(f"Annotation {i}: invalid bbox corners")
                        is_valid = False
                    
                    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                        issues.append(f"Annotation {i}: bbox out of image bounds")
                        is_valid = False
            
            elif ann['type'] == 'segmentation':
                polygon = ann['polygon']
                
                if len(polygon) < 3:
                    issues.append(f"Annotation {i}: polygon has < 3 points")
                    is_valid = False
                
                # Check if all points are in bounds
                for point in polygon:
                    if self.format_type == 'yolo':
                        if not (0 <= point[0] <= 1 and 0 <= point[1] <= 1):
                            issues.append(f"Annotation {i}: polygon point out of bounds")
                            is_valid = False
                            break
                    else:
                        if point[0] < 0 or point[0] > w or point[1] < 0 or point[1] > h:
                            issues.append(f"Annotation {i}: polygon point out of bounds")
                            is_valid = False
                            break
            
            if is_valid:
                valid.append(ann)
        
        return valid, issues