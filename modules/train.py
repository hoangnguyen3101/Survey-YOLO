import os
import argparse
from ultralytics import YOLO
import os

def main():
    parser = argparse.ArgumentParser(description='YOLO Training Script')
    parser.add_argument('--model', type=str, default='yolov8s.pt', help='model path')
    parser.add_argument('--data', type=str, default='data/bdd100k.yaml', help='data yaml path')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    parser.add_argument('--project', type=str, default='runs', help='project name (local)')
    parser.add_argument('--name', type=str, default='exp', help='experiment name')
    parser.add_argument('--device', type=str, default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--wandb', type=str, default='YOLO_BDD100K', help='wandb project name')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr0', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01, help='final learning rate (fraction of lr0)')
    parser.add_argument('--cos_lr', action='store_true', default=True, help='use cosine LR scheduler')
    args = parser.parse_args()
    
    # Set WANDB project name via environment variable
    if args.wandb:
        os.environ["WANDB_PROJECT"] = args.wandb

    # Load a model
    model = YOLO(args.model)
    
    # Train the model
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        device=args.device,
        batch=args.batch,
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=args.cos_lr
    )

if __name__ == '__main__':
    main()
