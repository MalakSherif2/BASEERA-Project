# src/threat_viz.py
import cv2
import numpy as np

class ThreatVisualizer:
    def __init__(self):
        print("🔄 Initializing Explainable Threat Visualizer...")

    def draw_threat_overlay(self, frame, tracking_id, behavior, confidence, weapon_detected, bbox, mask=None):
        """
        ترسم تفاصيل التهديد، الـ Bounding Box، الـ Mask، وحالة السلاح على الفريم مباشرة
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # اختيار اللون بناءً على حالة التهديد (أحمر لو فيه سلاح أو شجار، أخضر لو عادي)
        color = (0, 0, 255) if (weapon_detected or behavior == "Fighting") else (0, 255, 0)
        
        # لو فيه ماسك (من SAM)، نقدر نعمله Blending خفيف على الفريم
        if mask is not None:
            colored_mask = np.zeros_like(frame, dtype=np.uint8)
            colored_mask[mask > 0] = color
            frame = cv2.addWeighted(frame, 1.0, colored_mask, 0.4, 0)

        # رسم الـ Bounding Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # كتابة تفاصيل التهديد الواضحة (Explainable Details)
        threat_status = "WEAPON DETECTED" if weapon_detected else "NORMAL"
        label = f"ID: {x1} | {behavior} | {threat_status} ({confidence:.1f}%)"
        
        cv2.putText(frame, label, (x1, max(35, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame

if __name__ == "__main__":
    visualizer = ThreatVisualizer()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_bbox = [100, 100, 300, 400]
    result_frame = visualizer.draw_threat_overlay(dummy_frame, 4, "Fighting", 94.7, True, dummy_bbox)
    print(f"✅ Threat Visualization Test Successful! Frame Shape: {result_frame.shape}")