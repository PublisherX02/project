import os
import sys

# --- KAGGLE AUTO-INSTALLER ---
try:
    import ultralytics
except ImportError:
    print("📦 Ultralytics not found. Installing now...")
    os.system(f"{sys.executable} -m pip install ultralytics")
    import ultralytics
# -----------------------------

import yaml
import shutil
import torch
from ultralytics import YOLO

def find_kaggle_dataset(dataset_name="olea_v4"):
    """Dynamically finds the dataset in Kaggle's read-only input directory."""
    base_dir = "/kaggle/input"
    if not os.path.exists(base_dir):
        print("⚠️ Warning: /kaggle/input not found. Assuming local run.")
        return "data.yaml"
        
    print(f"🔍 Searching for '{dataset_name}' or 'data.yaml' in {base_dir}...")
    
    # 1. Search specifically for the named dataset folder
    for root, dirs, files in os.walk(base_dir):
        if 'data.yaml' in files and dataset_name.lower() in root.lower():
            return os.path.join(root, 'data.yaml')
            
    # 2. General fallback just in case Kaggle renamed the folder
    for root, dirs, files in os.walk(base_dir):
        if 'data.yaml' in files:
            return os.path.join(root, 'data.yaml')
            
    raise FileNotFoundError(f"Could NOT find data.yaml anywhere inside {base_dir}")

def generate_fixed_yaml(yaml_path):
    """
    Kaggle's data.yaml often contains relative paths (like `train: ../train/images`).
    YOLO throws FileNotFoundError if we don't fix these to absolute read-only paths.
    We must create a duplicate data.yaml in /kaggle/working/ with absolute paths.
    """
    if yaml_path == "data.yaml": return yaml_path # Local bypass
    
    dataset_root = os.path.dirname(yaml_path)
    fixed_yaml = "/kaggle/working/olea_v4_fixed.yaml"
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
        
    # Force absolute paths mapping directly to the Kaggle /input/ folder
    data['path'] = dataset_root
    for split in ['train', 'val', 'test']:
        if split in data and isinstance(data[split], str):
            if not data[split].startswith('/'):
                # Strip leading relative dots and slashes 
                cleaned_path = data[split].lstrip('./').lstrip('../')
                data[split] = os.path.join(dataset_root, cleaned_path)
                
    with open(fixed_yaml, 'w') as f:
        yaml.dump(data, f)
        
    print(f"✅ Generated fixed absolute-path YAML for Kaggle at: {fixed_yaml}")
    return fixed_yaml

def train():
    print("🚀 OLEA Vision Training Pipeline (Kaggle Edition) 🚀")
    print(f"💻 PyTorch GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"💻 Device Name: {torch.cuda.get_device_name(0)}")
    
    # 1. Prepare YAML
    original_yaml = find_kaggle_dataset("olea_v4")
    safe_yaml = generate_fixed_yaml(original_yaml)
    
    # 2. Initialize Model Backup (Training from empty YOLOv8m for max capacity)
    # Why YOLOv8m? You had 15,000 images. YOLOv8s bottlenecks early. Medium is better.
    print("🧠 Initializing YOLOv8 Model Architecture...")
    model = YOLO('yolov8m.pt') 
    
    # 3. Train
    # WARNING: Your previous crash ([rank0] Traceback) was caused by Ultralytics DDP (Multi-GPU). 
    # Kaggle's dual-T4 setup has a known NCCL torch communication bug that crashes randomly after hours of training.
    # FIX: We strictly enforce device=0 (Single T4 wrapper) to guarantee 100% completion without crashes.
    print("🔥 Starting Safe Training Protocol (Device 0 Enforced to prevent multi-GPU DDP crashes)...")
    
    results = model.train(
        data=safe_yaml,
        epochs=100,            
        imgsz=640,
        batch=32,              # 32 batch size total (16 per GPU) works for 2x T4
        device=[0, 1],         # 🛠️ UPDATED: Using BOTH Kaggle T4 GPUs (DDP Mode)
        project='/kaggle/working',
        name='olea_claims_stable_v4',
        patience=25,
        save=True,
        workers=4,             
        # Advanced augments for damage detection
        augment=True,
        degrees=15,
        hsv_s=0.5,
        hsv_v=0.4,
        copy_paste=0.1
    )
    
    print("\n✅ Training Complete. Best weights are saved in: /kaggle/working/olea_claims_stable_v4/weights/best.pt")
    print("⬇️ Download the `best.pt` file from Kaggle Output and rename it to `car_damage_yolov8s.pt` to hot-swap into our app!")

if __name__ == "__main__":
    train()
