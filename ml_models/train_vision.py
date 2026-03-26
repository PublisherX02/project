from ultralytics import YOLO
import shutil
import sys
import torch
from pathlib import Path

print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("CUDA not available. Stopping execution.")
    sys.exit(1)
print(f"GPU: {torch.cuda.get_device_name(0)}")

def train():
    data_yaml = Path("C:/Users/moham/GOMYCODE/Insurance-AI-Chatbot/ml_models/car_damage.yaml")
    model_path = Path("C:/Users/moham/GOMYCODE/Insurance-AI-Chatbot/yolov8s.pt")
    
    model = YOLO(model_path.as_posix())
    
    print("Starting YOLOv8n fine-tuning on unified dataset...")
    
    # Train
    results = model.train(
        data=data_yaml.as_posix(),
        epochs=75,
        imgsz=640,
        batch=16,  # Reduced from 32 to prevent RTX 4060 Laptop OOM
        patience=10,
        device=0,
        seed=42,
        project="C:/Users/moham/GOMYCODE/Insurance-AI-Chatbot/ml_models",
        name="car_damage_run_v2",
        exist_ok=True
    )
    
    best_weights = Path("C:/Users/moham/GOMYCODE/Insurance-AI-Chatbot/ml_models/car_damage_run_v2/weights/best.pt")
    target_weights = Path("C:/Users/moham/GOMYCODE/Insurance-AI-Chatbot/ml_models/car_damage_yolov8s.pt")
    
    if best_weights.exists():
        shutil.copy2(best_weights, target_weights)
        print(f"Saved best weights to {target_weights}")
    
    print("\nRunning evaluation on test split...")
    metrics = model.val(data=data_yaml.as_posix(), split="test")
    
    print(f"\nOverall mAP@50: {metrics.box.map50:.4f}")
    
    print("\nPer-class mAP@50:")
    # ultralytics ap_class_index maps to the metric vectors
    class_indices = metrics.box.ap_class_index
    for i, c_idx in enumerate(class_indices):
        class_name = model.names[c_idx]
        ap50 = metrics.box.ap50[i]
        print(f"  {class_name}: {ap50:.4f}")

if __name__ == "__main__":
    train()
