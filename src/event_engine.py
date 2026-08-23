# src/event_engine.py
import json
import time
from datetime import datetime

class AdvancedEventEngine:
    def __init__(self):
        print("🔄 Initializing Advanced Event Intelligence Engine...")
        self.active_events = {}

    def process_event(self, person_id, behavior, weapon_detected, confidence, bbox, segmentation_info=None):
        """
        معالجة وتوليد حدث ذكي ومتقدم يحتوي على كافة تفاصيل التهديد والخطورة
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # تحديد مستوى الخطورة (Threat Severity)
        if weapon_detected:
            severity = "CRITICAL"
        elif behavior == "Fighting":
            severity = "HIGH"
        else:
            severity = "MEDIUM"

        event_id = f"EVT-{int(time.time())}-{person_id}"
        
        event_data = {
            "event_id": event_id,
            "timestamp": timestamp,
            "person_id": int(person_id),
            "behavior": behavior,
            "weapon_detected": bool(weapon_detected),
            "confidence": float(confidence),
            "severity": severity,
            "bounding_box": [int(x) for x in bbox],
            "segmentation_active": bool(segmentation_info is not None)
        }
        
        self.active_events[event_id] = event_data
        return event_data

    def export_events_json(self, output_path="events/advanced_event_history.json"):
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(list(self.active_events.values()), f, indent=4, ensure_ascii=False)
        print(f"✅ Exported {len(self.active_events)} intelligent events to {output_path}")

if __name__ == "__main__":
    engine = AdvancedEventEngine()
    sample_event = engine.process_event(
        person_id=4, 
        behavior="Fighting", 
        weapon_detected=True, 
        confidence=94.7, 
        bbox=[100, 100, 300, 400],
        segmentation_info="mask_generated"
    )
    print("✅ Advanced Event Generated Successfully:")
    print(json.dumps(sample_event, indent=4))
    engine.export_events_json()