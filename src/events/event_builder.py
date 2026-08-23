from pathlib import Path
import json
import pandas as pd

# 1. Root Directory
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# 2. Paths
TRACKING_JSON = ROOT_DIR / "data" / "outputs" / "tracking_results.json"
EVENTS_OUTPUT = ROOT_DIR / "events" / "event_history" / "parsed_events.json"

# Make sure output directory exists
EVENTS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

def build_event_timeline():
    if not TRACKING_JSON.exists():
        print(f"❌ Tracking data not found at: {TRACKING_JSON}")
        return

    with open(TRACKING_JSON, "r") as f:
        tracking_data = json.load(f)

    # تحويل البيانات إلى Events مجمعة لكل شخص/عنصر
    events_summary = {}
    
    for entry in tracking_data:
        p_id = entry["person_id"]
        frame = entry["frame"]
        obj_class = entry["class"]
        
        if p_id not in events_summary:
            events_summary[p_id] = {
                "id": p_id,
                "class": obj_class,
                "first_seen_frame": frame,
                "last_seen_frame": frame,
                "total_frames_present": 0
            }
        
        events_summary[p_id]["last_seen_frame"] = frame
        events_summary[p_id]["total_frames_present"] += 1

    # حفظ الأحداث المجمعة
    output_data = list(events_summary.values())
    
    with open(EVENTS_OUTPUT, "w") as f:
        json.dump(output_data, f, indent=4)
        
    print(f"✅ Event Timeline Built Successfully!\n- Saved to: {EVENTS_OUTPUT}")

if __name__ == "__main__":
    build_event_timeline()