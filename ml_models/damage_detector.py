import io
import base64
from PIL import Image
from pathlib import Path
from ultralytics import YOLO

# Load model globally (once per module load)
MODEL_PATH = Path("C:/Users/moham/GOMYCODE/Insurance-AI-Chatbot/ml_models/car_damage_yolov8n.pt")
# Fallback to base yolov8n if fine-tuned is still training/missing
if not MODEL_PATH.exists():
    MODEL_PATH = Path("C:/Users/moham/GOMYCODE/Insurance-AI-Chatbot/yolov8n.pt")

try:
    _vision_model = YOLO(MODEL_PATH.as_posix())
except Exception as e:
    _vision_model = None
    print(f"Warning: Failed to load YOLO model: {e}")

def detect_damage(base64_img: str) -> dict:
    if not _vision_model:
        return {"detected": False, "damages": [], "summary": "Vision model failed to load."}
        
    try:
        # Strip data URI scheme if present
        if "base64," in base64_img:
            base64_img = base64_img.split("base64,")[1]
            
        img_data = base64.b64decode(base64_img)
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        
        # Run inference
        results = _vision_model(img, verbose=False)
        
        damages = []
        summary_sentences = []
        
        # Ultralytics results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_idx = int(box.cls[0].item())
                class_name = _vision_model.names[cls_idx]
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                damages.append({
                    "class": class_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })
                summary_sentences.append(f"{class_name.capitalize()} detected ({int(conf*100)}%).")
                
        if not damages:
            return {"detected": False, "damages": [], "summary": "No visible damage detected."}
            
        return {
            "detected": True,
            "damages": damages,
            "summary": " ".join(summary_sentences)
        }
    except Exception as e:
        print(f"Error in detect_damage: {e}")
        # Never crash the API
        return {"detected": False, "damages": [], "summary": "Error analyzing image."}
