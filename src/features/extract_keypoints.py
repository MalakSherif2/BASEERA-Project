import os
import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# 1. Root Directory & Paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = ROOT_DIR / "data" / "inputs" / "datasets" / "UCF_Crime_Clips"
OUTPUT_FEATURES_DIR = ROOT_DIR / "data" / "processed_features"

OUTPUT_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

def extract_sequence_from_video(video_path, model):
    """
    Extracts keypoint sequences per tracked person_id from a video clip.
    """
    cap = cv2.VideoCapture(str(video_path))
    person_sequences = {} # {person_id: [[x1, y1, conf1, ...], ...]}

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO Pose with ByteTrack
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            keypoints = results[0].keypoints

            if boxes.id is not None and keypoints is not None:
                track_ids = boxes.id.int().cpu().tolist()
                kpts_data = keypoints.data.cpu().numpy() # Shape: (N, 17, 3) -> (x, y, conf)

                for person_id, kpts in zip(track_ids, kpts_data):
                    p_key = f"person_{person_id}"
                    if p_key not in person_sequences:
                        person_sequences[p_key] = []
                    
                    # Store 17 keypoints (x, y) coordinates flattened
                    xy_kpts = kpts[:, :2].flatten().tolist()
                    person_sequences[p_key].append({
                        "frame": frame_idx,
                        "keypoints": xy_kpts
                    })

        frame_idx += 1

    cap.release()
    return person_sequences

def process_ucf_dataset():
    if not DATASET_DIR.exists():
        print(f"❌ Dataset folder not found at: {DATASET_DIR}")
        return

    print("🚀 Initializing YOLO11 Pose Model...")
    model = YOLO("models/yolo11n-pose.pt") # Uses pre-trained pose model

    categories = ["Fighting", "Robbery", "Stealing"]
    dataset_records = []

    for category in categories:
        cat_path = DATASET_DIR / category
        if not cat_path.exists():
            continue

        output_cat_dir = OUTPUT_FEATURES_DIR / category
        output_cat_dir.mkdir(parents=True, exist_ok=True)

        video_files = list(cat_path.glob("*.mp4")) + list(cat_path.glob("*.avi"))
        print(f"📂 Processing Category [{category}]: Found {len(video_files)} videos.")

        for vid_path in video_files:
            print(f"  🎬 Extracting Pose Sequences: {vid_path.name}...")
            sequences = extract_sequence_from_video(vid_path, model)

            # Save extracted feature JSON for this video
            out_file = output_cat_dir / f"{vid_path.stem}_features.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump({
                    "video_name": vid_path.name,
                    "category": category,
                    "persons": sequences
                }, f, indent=4)

            dataset_records.append({
                "video_name": vid_path.name,
                "category": category,
                "num_persons_tracked": len(sequences),
                "feature_file": str(out_file.relative_to(ROOT_DIR))
            })

    # Save dataset index
    index_file = OUTPUT_FEATURES_DIR / "dataset_index.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(dataset_records, f, indent=4)

    print("\n==========================================")
    print("✅ Keypoint Feature Extraction Completed!")
    print(f"📍 Features saved to: {OUTPUT_FEATURES_DIR}")
    print("==========================================")

if __name__ == "__main__":
    process_ucf_dataset()