import argparse
from ultralytics import YOLO
import os

def main():
    parser = argparse.ArgumentParser(description='YOLO Inference Script')
    parser.add_argument('--model', type=str, required=True, help='model path')
    parser.add_argument('--source', type=str, default='inference/source', help='source path (file, directory, URL, etc.)')
    parser.add_argument('--imgsz', type=int, default=640, help='inference size')
    parser.add_argument('--conf', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--project', type=str, default='inference', help='project name')
    parser.add_argument('--name', type=str, default='exp', help='experiment name')
    parser.add_argument('--device', type=str, default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save', action='store_true', default=True, help='save results')
    
    args = parser.parse_args()
    
    # Load a model
    model = YOLO(args.model)
    
    # Run inference
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        project=args.project,
        name=args.name,
        device=args.device,
        save=args.save
    )

if __name__ == '__main__':
    main()
