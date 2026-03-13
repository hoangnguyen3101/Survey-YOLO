import os
import random
from PIL import Image, ImageDraw, ImageFont

def draw_yolo(root_dir, split, num_samples=3):
    img_dir = os.path.join(root_dir, split, 'images')
    label_dir = os.path.join(root_dir, split, 'labels')
    output_dir = os.path.join(root_dir, 'debug_viz', split)
    os.makedirs(output_dir, exist_ok=True)

    # Get class names from data.yaml if exists, otherwise use BDD names
    names = ['bicycle', 'bus', 'car', 'caravan', 'motorcycle', 'pedestrian', 'rider', 'trailer', 'train', 'truck']
    
    image_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
    samples = random.sample(image_files, min(num_samples, len(image_files)))

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255), (0, 255, 128), (128, 128, 128)]

    for img_name in samples:
        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + '.txt')

        if not os.path.exists(label_path):
            continue

        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) != 5: continue
                cid, x, y, bw, bh = map(float, parts)
                cid = int(cid)

                # Convert YOLO to pixel
                left = (x - bw / 2) * w
                top = (y - bh / 2) * h
                right = (x + bw / 2) * w
                bottom = (y + bh / 2) * h

                color = colors[cid % len(colors)]
                draw.rectangle([left, top, right, bottom], outline=color, width=3)
                draw.text((left, top - 10), names[cid], fill=color)

        out_path = os.path.join(output_dir, img_name)
        img.save(out_path)
        print(f"Saved visualization: {out_path}")

if __name__ == "__main__":
    root = "/home/hoangnv/YOLO/data/bdd100k-10k"
    print("Visualizing train samples...")
    draw_yolo(root, "train")
    print("Visualizing val samples...")
    draw_yolo(root, "val")
