import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from ultralytics import YOLO

# 1. Setup Project Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_VIDEO = ROOT_DIR / "data" / "inputs" / "weapon_test.mp4"
OUTPUT_JSON = ROOT_DIR / "events" / "event_history" / "parsed_events.json"
OUTPUT_VIDEO = ROOT_DIR / "data" / "outputs" / "annotated_pipeline_out.mp4"
TEMPORAL_MODEL_PATH = ROOT_DIR / "models" / "temporal_behavior_model.pt"

# 👇 هنا تم تحديث المسار ليعمل مع الموديل الجديد اللي نزلتيه من كولاب
WEAPON_MODEL_PATH = ROOT_DIR / "models" / "weapon_detect_best.pt"

# Ensure Directories Exist
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {0: "Fighting", 1: "Robbery", 2: "Stealing"}

# 2. Temporal Behavior Classifier Definition
class TemporalBehaviorClassifier(nn.Module):
    def __init__(self, input_size=34, hidden_size=64, num_classes=3, num_layers=2):
        super(TemporalBehaviorClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def load_temporal_behavior_model():
    if not TEMPORAL_MODEL_PATH.exists():
        print(f"⚠️ Temporal model not found at {TEMPORAL_MODEL_PATH}.")
        return None
    
    model = TemporalBehaviorClassifier()
    model.load_state_dict(torch.load(TEMPORAL_MODEL_PATH))
    model.eval()
    return model

def load_weapon_detector():
    if WEAPON_MODEL_PATH.exists():
        print(f"✅ Loading Weapon Detector from {WEAPON_MODEL_PATH}...")
        return YOLO(str(WEAPON_MODEL_PATH))
    print("ℹ️ Weapon detector weights not found yet. Running Pose + Temporal detection only.")
    return None

def predict_person_behavior(temporal_model, kpts_sequence, max_seq_len=30):
    if temporal_model is None or len(kpts_sequence) < 5:
        return "Normal Movement", 0.0

    seq = list(kpts_sequence)
    if len(seq) < max_seq_len:
        padding = [[0.0] * 34] * (max_seq_len - len(seq))
        seq.extend(padding)
    else:
        seq = seq[:max_seq_len]

    inp_tensor = torch.tensor([seq], dtype=torch.float32)
    with torch.no_grad():
        outputs = temporal_model(inp_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_cls = torch.max(probs, dim=1)

    predicted_label = LABEL_MAP.get(pred_cls.item(), "Normal Movement")
    confidence = float(conf.item())

    if confidence < 0.4:
        return "Normal Movement", confidence

    return predicted_label, confidence

def run_visionguard_pipeline(video_source=str(INPUT_VIDEO)):
    if isinstance(video_source, str) and video_source.isdigit():
        video_source = int(video_source)

    print("🚀 Initializing VISIONGUARD Pipeline Models...")
    pose_model = YOLO("yolo11n-pose.pt")
    temporal_model = load_temporal_behavior_model()
    weapon_model = load_weapon_detector()

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"❌ Failed to open video source: {video_source}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, fps, (width, height))

    tracking_history = {}
    person_kpt_sequences = {}
    frame_count = 0

    print(f"🎬 Processing Stream / Video Source: {video_source}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 1. Pose Tracking
        results = pose_model.track(
            frame, 
            persist=True, 
            tracker="bytetrack.yaml", 
            conf=0.25, 
            verbose=False
        )

        annotated_frame = results[0].plot()

        # 2. Weapon Detector Inference (سيستخدم موديلك الجديد تلقائياً هنا)
        weapons_detected_in_frame = False
        if weapon_model is not None:
            weapon_results = weapon_model(frame, conf=0.4, verbose=False)
            if weapon_results and len(weapon_results[0].boxes) > 0:
                weapons_detected_in_frame = True
                annotated_frame = weapon_results[0].plot(img=annotated_frame)

        out_writer.write(annotated_frame)

        # 3. Track Aggregation
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            cls_ids = results[0].boxes.cls.int().cpu().tolist()

            keypoints_data = None
            if results[0].keypoints is not None:
                keypoints_data = results[0].keypoints.data.cpu().numpy()

            for i, p_id in enumerate(track_ids):
                bbox = boxes[i].tolist()
                obj_cls = pose_model.names[cls_ids[i]]

                flat_kpts = []
                if keypoints_data is not None:
                    flat_kpts = keypoints_data[i][:, :2].flatten().tolist()

                if p_id not in person_kpt_sequences:
                    person_kpt_sequences[p_id] = []

                if len(flat_kpts) == 34:
                    person_kpt_sequences[p_id].append(flat_kpts)

                if p_id not in tracking_history:
                    tracking_history[p_id] = {
                        "person_id": p_id,
                        "class": obj_cls,
                        "start_frame": frame_count,
                        "end_frame": frame_count,
                        "total_frames": 0,
                        "last_bbox": bbox,
                        "has_pose": len(flat_kpts) > 0,
                        "detected_behavior": "Normal Movement",
                        "behavior_confidence": 0.0,
                        "weapon_detected": False
                    }

                tracking_history[p_id]["end_frame"] = frame_count
                tracking_history[p_id]["total_frames"] += 1
                tracking_history[p_id]["last_bbox"] = bbox
                if weapons_detected_in_frame:
                    tracking_history[p_id]["weapon_detected"] = True

    cap.release()
    out_writer.release()

    # 4. Consolidate Events & Behaviors
    events_list = list(tracking_history.values())
    for ev in events_list:
        p_id = ev["person_id"]
        kpt_seq = person_kpt_sequences.get(p_id, [])
        behavior, conf = predict_person_behavior(temporal_model, kpt_seq)
        
        ev["detected_behavior"] = behavior
        ev["behavior_confidence"] = round(conf, 2)
        ev["start_time_sec"] = round(ev["start_frame"] / fps, 2)
        ev["end_time_sec"] = round(ev["end_frame"] / fps, 2)
        ev["duration_sec"] = round(ev["total_frames"] / fps, 2)

    output_schema = {
        "summary": {
            "total_frames": frame_count,
            "total_unique_persons": len(events_list),
            "fps": fps
        },
        "tracked_persons": {f"person_{ev['person_id']}": ev for ev in events_list}
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_schema, f, indent=4)

    print(f"\n✅ Stream / Pipeline Run Complete!")
    print(f"📹 Output Video Saved: {OUTPUT_VIDEO}")
    print(f"📊 Events Telemetry Saved: {OUTPUT_JSON}")

if __name__ == "__main__":
    run_visionguard_pipeline(str(INPUT_VIDEO))