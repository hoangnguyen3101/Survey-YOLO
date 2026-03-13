import argparse
import csv
from ultralytics import YOLO
import os

def main():
    parser = argparse.ArgumentParser(description='YOLO Validation Script')
    parser.add_argument('--model', type=str, required=True, help='path to model weights (e.g., runs/exp/weights/best.pt)')
    parser.add_argument('--data', type=str, default='data/bdd100k.yaml', help='path to data yaml')
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--conf', type=float, default=0.001, help='object confidence threshold for detection')
    parser.add_argument('--iou', type=float, default=0.6, help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', type=str, default='0', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--project', type=str, default='runs/val', help='project name')
    parser.add_argument('--name', type=str, default='val_exp', help='experiment name')

    args = parser.parse_args()

    # Load a model
    model = YOLO(args.model)

    # Validate the model
    results = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=args.project,
        name=args.name
    )

    # Save results to CSV
    save_dir = results.save_dir
    csv_path = os.path.join(save_dir, 'val_results.csv')
    
    class_names = results.names  # dict {id: name}
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'images', 'instances', 'precision', 'recall', 'mAP50', 'mAP50-95'])
        
        # Per-class metrics
        for i, cls_name in class_names.items():
            if i < len(results.box.p):
                writer.writerow([
                    cls_name,
                    results.box.n if hasattr(results.box, 'n') else '',
                    '',
                    f'{results.box.p[i]:.4f}',
                    f'{results.box.r[i]:.4f}',
                    f'{results.box.ap50[i]:.4f}',
                    f'{results.box.ap[i]:.4f}'
                ])
        
        # Overall metrics
        writer.writerow([
            'all',
            '',
            '',
            f'{results.box.mp:.4f}',
            f'{results.box.mr:.4f}',
            f'{results.box.map50:.4f}',
            f'{results.box.map:.4f}'
        ])
    
    print(f"\nResults saved to CSV: {csv_path}")

if __name__ == '__main__':
    main()

