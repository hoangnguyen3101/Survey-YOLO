import json
import os
import glob
from pathlib import Path

def convert_to_yolo(root_dir):
    # 1. Load class mapping from meta.json
    meta_path = os.path.join(root_dir, 'meta.json')
    if not os.path.exists(meta_path):
        print(f"Error: meta.json not found at {meta_path}")
        return

    with open(meta_path, 'r') as f:
        meta_data = json.load(f)
    
    classes = [c['title'] for c in meta_data['classes']]
    class_to_id = {title: i for i, title in enumerate(classes)}
    print(f"Classes found: {classes}")

    # 2. Iterate through splits
    splits = ['train', 'val', 'test']
    for split in splits:
        split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(split_dir):
            print(f"Skipping split {split}: directory not found.")
            continue
        
        ann_dir = os.path.join(split_dir, 'ann')
        # Standard YOLO expects 'images' and 'labels'
        img_dir = os.path.join(split_dir, 'img')
        images_dir = os.path.join(split_dir, 'images')
        labels_dir = os.path.join(split_dir, 'labels')
        
        # Rename img to images if it exists
        if os.path.exists(img_dir) and not os.path.exists(images_dir):
            print(f"Renaming {img_dir} to {images_dir}...")
            os.rename(img_dir, images_dir)
        
        if not os.path.exists(ann_dir):
            print(f"Skipping {split}: ann/ directory not found.")
            continue
            
        os.makedirs(labels_dir, exist_ok=True)
        
        json_files = glob.glob(os.path.join(ann_dir, '*.json'))
        print(f"Processing {len(json_files)} files in {split}...")
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            img_width = data['size']['width']
            img_height = data['size']['height']
            
            yolo_lines = set()
            for obj in data.get('objects', []):
                class_title = obj['classTitle']
                if class_title not in class_to_id:
                    continue
                
                cid = class_to_id[class_title]
                
                # Calculate bounding box from points
                # Geometry type can be 'polygon' or 'rectangle'
                exterior_points = obj.get('points', {}).get('exterior', [])
                if not exterior_points:
                    continue
                
                xs = [p[0] for p in exterior_points]
                ys = [p[1] for p in exterior_points]
                
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                
                # YOLO format: class_id x_center y_center width height (all normalized)
                x_center = (min_x + max_x) / 2.0 / img_width
                y_center = (min_y + max_y) / 2.0 / img_height
                w = (max_x - min_x) / float(img_width)
                h = (max_y - min_y) / float(img_height)
                
                # Clamp values to [0, 1] just in case
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                w = max(0, min(1, w))
                h = max(0, min(1, h))
                
                yolo_lines.add(f"{cid} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
            
            # Save to labels/ directory
            base_name = os.path.basename(json_file).replace('.jpg.json', '').replace('.png.json', '')
            label_file = os.path.join(labels_dir, f"{base_name}.txt")
            
            with open(label_file, 'w') as f:
                f.write('\n'.join(list(yolo_lines)))

    # 3. Create data.yaml with relative path
    yaml_content = f"""path: .
train: train/images
val: val/images
test: test/images

names:
"""
    for i, name in enumerate(classes):
        yaml_content += f"  {i}: {name}\n"
    
    with open(os.path.join(root_dir, 'data.yaml'), 'w') as f:
        f.write(yaml_content)
    print(f"Created data.yaml at {os.path.join(root_dir, 'data.yaml')}")

if __name__ == "__main__":
    # Use the directory where the script is located as the default root_dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    convert_to_yolo(script_dir)
