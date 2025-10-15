"""
Script 4: Validate Preprocessing Results
Validate preprocessed dataset and generate quality report
"""

import sys
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils import setup_logging, load_image, validate_image_quality
from label_handler import LabelHandler


def validate_preprocessing(preprocessed_dir: str, 
                          output_report: str = "validation_report",
                          target_size: tuple = (2048, 1024)):
    """
    Validate preprocessed dataset
    
    Args:
        preprocessed_dir: Directory containing preprocessed data
        output_report: Output directory for validation report
        target_size: Expected image size (width, height)
    """
    setup_logging()
    
    print("\n" + "="*60)
    print("PREPROCESSING VALIDATION")
    print("="*60)
    
    # Create output directory
    report_dir = Path(output_report)
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all tasks
    preprocessed_path = Path(preprocessed_dir)
    tasks = [d.name for d in preprocessed_path.iterdir() if d.is_dir()]
    
    print(f"Found tasks: {tasks}")
    
    all_validation_results = {}
    
    for task_name in tasks:
        print(f"\n{'='*60}")
        print(f"Validating task: {task_name}")
        print(f"{'='*60}")
        
        task_dir = preprocessed_path / task_name
        image_dir = task_dir / 'images'
        label_dir = task_dir / 'labels'
        
        if not image_dir.exists():
            print(f"⚠️  Image directory not found: {image_dir}")
            continue
        
        # Find all images
        image_paths = list(image_dir.rglob('*.png')) + list(image_dir.rglob('*.jpg'))
        print(f"Found {len(image_paths)} images")
        
        # Validation metrics
        resolution_check = []
        aspect_ratio_check = []
        intensity_stats = []
        label_validity = []
        missing_labels = 0
        
        # Process each image
        print("\nValidating images...")
        for img_path in tqdm(image_paths):
            # Load image
            image = load_image(str(img_path))
            if image is None:
                continue
            
            h, w = image.shape
            
            # 1. Resolution check
            resolution_ok = (w == target_size[0] and h == target_size[1])
            resolution_check.append(resolution_ok)
            
            # 2. Aspect ratio check
            aspect_ratio = w / h
            expected_ratio = target_size[0] / target_size[1]
            aspect_ok = abs(aspect_ratio - expected_ratio) < 0.01
            aspect_ratio_check.append(aspect_ok)
            
            # 3. Intensity statistics
            intensity_stats.append({
                'mean': np.mean(image),
                'std': np.std(image),
                'min': np.min(image),
                'max': np.max(image)
            })
            
            # 4. Label validation
            label_path = label_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                label_handler = LabelHandler(format_type='yolo')
                annotations = label_handler.read_label(str(label_path))
                
                if len(annotations) > 0:
                    valid_anns, issues = label_handler.validate_annotations(
                        annotations, (w, h)
                    )
                    label_validity.append({
                        'path': str(label_path),
                        'total': len(annotations),
                        'valid': len(valid_anns),
                        'issues': issues
                    })
            else:
                missing_labels += 1
        
        # Compute validation statistics
        validation_stats = {
            'total_images': len(image_paths),
            'resolution': {
                'target': list(target_size),
                'correct': sum(resolution_check),
                'incorrect': len(resolution_check) - sum(resolution_check),
                'accuracy': sum(resolution_check) / len(resolution_check) if resolution_check else 0
            },
            'aspect_ratio': {
                'correct': sum(aspect_ratio_check),
                'incorrect': len(aspect_ratio_check) - sum(aspect_ratio_check),
                'accuracy': sum(aspect_ratio_check) / len(aspect_ratio_check) if aspect_ratio_check else 0
            },
            'intensity': {
                'mean_avg': float(np.mean([s['mean'] for s in intensity_stats])),
                'mean_std': float(np.std([s['mean'] for s in intensity_stats])),
                'std_avg': float(np.mean([s['std'] for s in intensity_stats])),
                'range': [
                    float(np.min([s['min'] for s in intensity_stats])),
                    float(np.max([s['max'] for s in intensity_stats]))
                ]
            },
            'labels': {
                'with_labels': len(label_validity),
                'missing_labels': missing_labels,
                'total_annotations': sum([lv['total'] for lv in label_validity]),
                'valid_annotations': sum([lv['valid'] for lv in label_validity]),
                'issues_count': sum([len(lv['issues']) for lv in label_validity])
            }
        }
        
        all_validation_results[task_name] = validation_stats
        
        # Print summary
        print(f"\n✓ Validation results for {task_name}:")
        print(f"  Images: {validation_stats['total_images']}")
        print(f"  Resolution accuracy: {validation_stats['resolution']['accuracy']*100:.1f}%")
        print(f"  Aspect ratio accuracy: {validation_stats['aspect_ratio']['accuracy']*100:.1f}%")
        print(f"  Mean intensity: {validation_stats['intensity']['mean_avg']:.1f} ± {validation_stats['intensity']['mean_std']:.1f}")
        print(f"  Labels with annotations: {validation_stats['labels']['with_labels']}")
        print(f"  Missing labels: {validation_stats['labels']['missing_labels']}")
        print(f"  Valid annotations: {validation_stats['labels']['valid_annotations']}/{validation_stats['labels']['total_annotations']}")
        
        # Create visualizations
        print("\nCreating validation visualizations...")
        
        # 1. Intensity distribution
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        means = [s['mean'] for s in intensity_stats]
        stds = [s['std'] for s in intensity_stats]
        
        axes[0, 0].hist(means, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Mean Intensity')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title(f'Mean Intensity Distribution\n(avg={np.mean(means):.1f})')
        axes[0, 0].axvline(np.mean(means), color='r', linestyle='--', label='Mean')
        axes[0, 0].legend()
        
        axes[0, 1].hist(stds, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Standard Deviation')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title(f'Std Deviation Distribution\n(avg={np.mean(stds):.1f})')
        axes[0, 1].axvline(np.mean(stds), color='r', linestyle='--', label='Mean')
        axes[0, 1].legend()
        
        # 2. Resolution/Aspect validation
        categories = ['Resolution', 'Aspect Ratio']
        correct = [
            validation_stats['resolution']['correct'],
            validation_stats['aspect_ratio']['correct']
        ]
        incorrect = [
            validation_stats['resolution']['incorrect'],
            validation_stats['aspect_ratio']['incorrect']
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, correct, width, label='Correct', color='green', alpha=0.7)
        axes[1, 0].bar(x + width/2, incorrect, width, label='Incorrect', color='red', alpha=0.7)
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Validation Results')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(categories)
        axes[1, 0].legend()
        
        # 3. Label statistics
        if validation_stats['labels']['total_annotations'] > 0:
            label_data = [
                validation_stats['labels']['valid_annotations'],
                validation_stats['labels']['issues_count']
            ]
            label_labels = ['Valid', 'Issues']
            axes[1, 1].pie(label_data, labels=label_labels, autopct='%1.1f%%',
                          colors=['green', 'orange'], startangle=90)
            axes[1, 1].set_title(f'Label Validation\n(Total: {validation_stats["labels"]["total_annotations"]})')
        else:
            axes[1, 1].text(0.5, 0.5, 'No labels found', 
                          ha='center', va='center', fontsize=14)
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(report_dir / f'{task_name}_validation.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Visualization saved: {report_dir / f'{task_name}_validation.png'}")
        
        # Save label issues if any
        if any(len(lv['issues']) > 0 for lv in label_validity):
            issues_file = report_dir / f'{task_name}_label_issues.txt'
            with open(issues_file, 'w') as f:
                f.write(f"Label Validation Issues - {task_name}\n")
                f.write("="*60 + "\n\n")
                for lv in label_validity:
                    if len(lv['issues']) > 0:
                        f.write(f"File: {lv['path']}\n")
                        f.write(f"Total annotations: {lv['total']}\n")
                        f.write(f"Valid annotations: {lv['valid']}\n")
                        f.write("Issues:\n")
                        for issue in lv['issues']:
                            f.write(f"  - {issue}\n")
                        f.write("\n")
            print(f"⚠️  Label issues saved: {issues_file}")
    
    # Save overall validation report
    report_path = report_dir / 'validation_summary.yaml'
    with open(report_path, 'w') as f:
        yaml.dump(all_validation_results, f, default_flow_style=False)
    
    print(f"\n{'='*60}")
    print("VALIDATION COMPLETE")
    print(f"{'='*60}")
    print(f"Report saved: {report_path}")
    
    # Overall summary
    print("\nOverall Summary:")
    for task_name, stats in all_validation_results.items():
        print(f"\n{task_name}:")
        res_acc = stats['resolution']['accuracy']
        asp_acc = stats['aspect_ratio']['accuracy']
        
        status = "✓ PASS" if res_acc > 0.99 and asp_acc > 0.99 else "⚠️  REVIEW NEEDED"
        print(f"  Status: {status}")
        print(f"  Resolution: {res_acc*100:.1f}% correct")
        print(f"  Aspect ratio: {asp_acc*100:.1f}% correct")
        print(f"  Intensity: {stats['intensity']['mean_avg']:.1f} ± {stats['intensity']['mean_std']:.1f}")
    
    print("\n" + "="*60)
    print("All validation reports and visualizations saved to:")
    print(f"{report_dir}")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate preprocessed dataset')
    parser.add_argument('--preprocessed_dir', type=str, required=True,
                       help='Directory containing preprocessed data')
    parser.add_argument('--output', type=str, default='validation_report',
                       help='Output directory for validation report')
    parser.add_argument('--target_width', type=int, default=2048,
                       help='Expected image width')
    parser.add_argument('--target_height', type=int, default=1024,
                       help='Expected image height')
    
    args = parser.parse_args()
    
    validate_preprocessing(
        args.preprocessed_dir,
        args.output,
        (args.target_width, args.target_height)
    )