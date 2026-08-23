# models/detector.py
from ultralytics import YOLO
import datetime

class WeaponDetector:
    def __init__(self, model_path='models/weapon_detect_best.pt'):
        self.model = YOLO(model_path)
    
    def detect(self, frame):
        results = self.model(frame)
        detections = []
        
        for r in results:
            for box in r.boxes:
                detections.append({
                    "weapon": self.model.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "timestamp": datetime.datetime.now().isoformat()
                })
        return detections