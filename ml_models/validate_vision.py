from ultralytics import YOLO
import random
from pathlib import Path

def validate():
    # Fix: Resolve all paths relative to the script location to ensure it works anywhere
    script_dir = Path(__file__).parent
    weights_path = script_dir / "car_damage_yolov8s.pt"
    data_yaml = script_dir / "car_damage.yaml"
    img_dir = script_dir / "car_damage_unified/test/images"

    try:
        model = YOLO(str(weights_path))
    except Exception as e:
        print(f"Failed to load tuned model from {weights_path}: {e}")
        return

    print("\n--- TEST SPLIT VALIDATION ---")
    # Data YAML must be passed as a string for ultralytics val()
    metrics = model.val(data=str(data_yaml), split="test")
    
    print(f"\nOverall mAP@50: {metrics.box.map50:.4f}")
    
    print("\nPer-class mAP@50:")
    class_indices = metrics.box.ap_class_index
    for i, c_idx in enumerate(class_indices):
        class_name = model.names[c_idx]
        if len(metrics.box.ap50) > i:
            ap50 = metrics.box.ap50[i]
            print(f"  {class_name}: {ap50:.4f}")
        
    print("\n--- SAMPLE INFERENCE CHECK ---")
    imgs = list(img_dir.glob("*.jpg"))
    
    if not imgs:
        print(f"No test images found in {img_dir}!")
        return
        
    for img_path in random.sample(imgs, min(3, len(imgs))):
        res = model(str(img_path), verbose=False)
        boxes = res[0].boxes
        if len(boxes) == 0:
            print(f"Image {img_path.name}: [No damage detected]")
        else:
            detections = [f"{model.names[int(c)]} ({conf:.2f})" for c, conf in zip(boxes.cls, boxes.conf)]
            print(f"Image {img_path.name}: {detections}")

if __name__ == '__main__':
    validate()
