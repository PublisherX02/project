import os
import json
import random
import shutil
from pathlib import Path

random.seed(42)

def prepare_dataset():
    base_dir = Path("C:/Users/moham/GOMYCODE/Insurance-AI-Chatbot")
    target_dir = base_dir / "ml_models" / "car_damage_unified"
    
    if target_dir.exists():
        shutil.rmtree(target_dir)

    for split in ['train', 'val', 'test']:
        os.makedirs(target_dir / split / 'images', exist_ok=True)
        os.makedirs(target_dir / split / 'labels', exist_ok=True)
        
    print("Converting COCO to YOLO and merging datasets...")
    
    # Map COCO category id (1-6) to YOLO class id (0-4)
    coco_to_yolo = {
        1: 0, 2: 1, 3: 1, 4: 2, 5: 3, 6: 4
    }
    class_names = {
        0: 'dent', 1: 'surface_damage', 2: 'glass shatter', 
        3: 'lamp broken', 4: 'tire flat'
    }
    
    class_counts = {0:0, 1:0, 2:0, 3:0, 4:0}
    all_data = [] # List of tuples: (image path, list of yolo labels)

    for src_split in ['train', 'val', 'test']:
        json_file = base_dir / f"{src_split}.json"
        img_dir = base_dir / src_split
        
        if not json_file.exists() or not img_dir.exists():
            continue
            
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # mapping image_id to index
        img_dict = {img['id']: img for img in data['images']}
        
        # map image_id to annotations
        anno_dict = {img_id: [] for img_id in img_dict.keys()}
        for anno in data.get('annotations', []):
            anno_dict[anno['image_id']].append(anno)
            
        for img_id, img_info in img_dict.items():
            img_path = img_dir / img_info['file_name']
            if not img_path.exists():
                continue
                
            yolo_labels = []
            width = float(img_info['width'])
            height = float(img_info['height'])
            
            for anno in anno_dict[img_id]:
                cat_id = anno['category_id']
                if cat_id not in coco_to_yolo:
                    continue
                yolo_c = coco_to_yolo[cat_id]
                
                # COCO bbox: [x_min, y_min, width, height]
                x, y, w, h = anno['bbox']
                
                # YOLO format: x_center, y_center, width, height (normalized)
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                w_norm = w / width
                h_norm = h / height
                
                # Clamp between 0 and 1
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                w_norm = max(0.0, min(1.0, w_norm))
                h_norm = max(0.0, min(1.0, h_norm))
                
                yolo_labels.append(f"{yolo_c} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
                class_counts[yolo_c] += 1
                
            all_data.append((img_path, yolo_labels))
            
    print(f"Total valid images fetched: {len(all_data)}")
    
    # Shuffle and split 80/10/10
    random.shuffle(all_data)
    n = len(all_data)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    
    splits = {
        'train': all_data[:n_train],
        'val': all_data[n_train:n_train+n_val],
        'test': all_data[n_train+n_val:]
    }
    
    # Write to new dirs
    for dst_split, items in splits.items():
        for img_path, labels in items:
            # Copy image
            new_img_path = target_dir / dst_split / 'images' / img_path.name
            shutil.copy2(img_path, new_img_path)
            
            # Write labels (even if empty, ultralytics supports background images)
            new_label_path = target_dir / dst_split / 'labels' / img_path.with_suffix('.txt').name
            with open(new_label_path, 'w') as f:
                f.write('\n'.join(labels))
                
    # Write yaml config
    yaml_content = f"""path: {target_dir.absolute().as_posix()}
train: train/images
val: val/images
test: test/images

names:
  0: dent
  1: surface_damage
  2: glass shatter
  3: lamp broken
  4: tire flat
"""
    yaml_path = base_dir / "ml_models" / "car_damage.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
        
    print("\n=== Dataset Conversion Summary ===")
    print(f"Train split: {len(splits['train'])} images")
    print(f"Val split: {len(splits['val'])} images")
    print(f"Test split: {len(splits['test'])} images")
    print("\nClass Distribution:")
    for yolo_c, count in class_counts.items():
        print(f"  {class_names[yolo_c]} (ID {yolo_c}): {count} annotations")
    print(f"\nCreated YOLO config at {yaml_path}")
    print("Step 2 fully completed.")

if __name__ == '__main__':
    prepare_dataset()
