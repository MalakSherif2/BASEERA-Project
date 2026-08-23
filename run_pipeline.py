from pathlib import Path
import cv2
import torch
import numpy as np

from detector import WeaponDetector
from src.models.temporal_classifier import (
    TemporalBehaviorClassifier,
    INV_LABEL_MAP
)
from src.models.sam_segmenter import SAMSegmentationEnhancer
from src.threat_viz import ThreatVisualizer
from src.event_engine import AdvancedEventEngine


# =========================
# Paths
# =========================

ROOT_DIR = Path(__file__).resolve().parent

POSE_MODEL_PATH = ROOT_DIR / "models" / "yolo11n-pose.pt"
BEHAVIOR_MODEL_PATH = ROOT_DIR / "models" / "temporal_behavior_model.pt"
WEAPON_MODEL_PATH = ROOT_DIR / "models" / "weapon_detect_best.pt"

OUTPUT_DIR = ROOT_DIR / "data" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO_PATH = OUTPUT_DIR / "annotated_pipeline_out.mp4"
EVENTS_PATH = ROOT_DIR / "events" / "pipeline_final_report.json"


# =========================
# Device
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Load Models
# =========================

def load_models():

    print("🚀 Loading VisionGuard models...")

    # YOLO Pose
    from ultralytics import YOLO

    pose_model = YOLO(str(POSE_MODEL_PATH))

    # Weapon Detector
    weapon_detector = WeaponDetector(
        model_path=str(WEAPON_MODEL_PATH)
    )

    # Temporal Behavior Model
    behavior_model = TemporalBehaviorClassifier()

    behavior_model.load_state_dict(
        torch.load(
            BEHAVIOR_MODEL_PATH,
            map_location=DEVICE
        )
    )

    behavior_model.to(DEVICE)
    behavior_model.eval()

    # SAM wrapper
    segmenter = SAMSegmentationEnhancer()

    # Visualization
    visualizer = ThreatVisualizer()

    # Event Engine
    event_engine = AdvancedEventEngine()

    print("✅ All models loaded successfully!")

    return (
        pose_model,
        weapon_detector,
        behavior_model,
        segmenter,
        visualizer,
        event_engine
    )


# =========================
# Behavior Prediction
# =========================

def predict_behavior(
    behavior_model,
    keypoint_sequence
):
    """
    keypoint_sequence:
        shape = (30, 34)
    """

    sequence = np.array(
        keypoint_sequence,
        dtype=np.float32
    )

    tensor = torch.tensor(
        sequence,
        dtype=torch.float32
    )

    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(DEVICE)

    with torch.no_grad():

        output = behavior_model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = (
            probabilities[0][predicted_class]
            .item()
            * 100
        )

    behavior = INV_LABEL_MAP.get(
        predicted_class,
        "Unknown"
    )

    return behavior, confidence


# =========================
# Main Pipeline
# =========================

def run_vision_guard_pipeline(video_path):

    print("\n🚀 Starting VisionGuard Pipeline")
    print(f"🎥 Input: {video_path}")

    (
        pose_model,
        weapon_detector,
        behavior_model,
        segmenter,
        visualizer,
        event_engine
    ) = load_models()

    # -------------------------
    # Open Video
    # -------------------------

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"❌ Cannot open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print(f"📐 Resolution: {width}x{height}")
    print(f"⚡ FPS: {fps}")
    print(f"🎬 Total Frames: {total_frames}")

    # -------------------------
    # Output Video
    # -------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO_PATH),
        fourcc,
        fps,
        (width, height)
    )

    # -------------------------
    # Store keypoints
    # -------------------------

    person_sequences = {}

    # -------------------------
    # Frame Loop
    # -------------------------

    frame_idx = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_idx += 1

        # =====================
        # YOLO Pose + Tracking
        # =====================

        results = pose_model.track(
            frame,
            persist=True,
            verbose=False
        )

        result = results[0]

        # =====================
        # Persons
        # =====================

        if (
            result.keypoints is not None
            and len(result.keypoints) > 0
        ):

            track_ids = None

            if (
                result.boxes is not None
                and result.boxes.id is not None
            ):
                track_ids = (
                    result.boxes.id
                    .cpu()
                    .numpy()
                    .astype(int)
                )

            for person_idx in range(
                len(result.keypoints)
            ):

                # ---------------------
                # Person ID
                # ---------------------

                if track_ids is not None:

                    person_id = int(
                        track_ids[person_idx]
                    )

                else:

                    person_id = person_idx

                # ---------------------
                # Keypoints
                # ---------------------

                kpts_xy = (
                    result
                    .keypoints
                    .xy[person_idx]
                    .cpu()
                    .numpy()
                )

                # 17 × 2 → 34
                keypoints = (
                    kpts_xy
                    .flatten()
                    .astype(np.float32)
                )

                # ---------------------
                # Bounding Box
                # ---------------------

                if result.boxes is not None:

                    bbox = (
                        result
                        .boxes
                        .xyxy[person_idx]
                        .cpu()
                        .numpy()
                        .astype(int)
                        .tolist()
                    )

                else:

                    bbox = [
                        0,
                        0,
                        width,
                        height
                    ]

                # =====================
                # Sequence Management
                # =====================

                if person_id not in person_sequences:

                    person_sequences[
                        person_id
                    ] = []

                person_sequences[
                    person_id
                ].append(keypoints)

                # Keep latest 30 frames
                if len(
                    person_sequences[person_id]
                ) > 30:

                    person_sequences[
                        person_id
                    ] = person_sequences[
                        person_id
                    ][-30:]

                # =====================
                # Default Values
                # =====================

                behavior = "Analyzing..."
                behavior_confidence = 0.0

                # =====================
                # Behavior Classification
                # =====================

                if len(
                    person_sequences[person_id]
                ) == 30:

                    behavior, behavior_confidence = (
                        predict_behavior(
                            behavior_model,
                            person_sequences[
                                person_id
                            ]
                        )
                    )

                # =====================
                # Weapon Detection
                # =====================

                weapon_detections = (
                    weapon_detector.detect(frame)
                )

                weapon_detected = (
                    len(weapon_detections) > 0
                )

                # =====================
                # SAM / Segmentation
                # =====================

                mask = segmenter.segment_bbox(
                    frame,
                    bbox
                )

                # =====================
                # Visualization
                # =====================

                frame = (
                    visualizer.draw_threat_overlay(
                        frame=frame,
                        tracking_id=person_id,
                        behavior=behavior,
                        confidence=behavior_confidence,
                        weapon_detected=weapon_detected,
                        bbox=bbox,
                        mask=mask
                    )
                )

                # =====================
                # Event Engine
                # =====================

                if (
                    weapon_detected
                    or behavior in [
                        "Fighting",
                        "Robbery",
                        "Stealing"
                    ]
                ):

                    event_engine.process_event(
                        person_id=person_id,
                        behavior=behavior,
                        weapon_detected=weapon_detected,
                        confidence=behavior_confidence,
                        bbox=bbox,
                        segmentation_info="mask_applied"
                    )

        # =====================
        # Write Frame
        # =====================

        writer.write(frame)

        # Progress
        if frame_idx % 30 == 0:

            print(
                f"🔄 Processed "
                f"{frame_idx}/{total_frames} frames"
            )

    # =========================
    # Cleanup
    # =========================

    cap.release()
    writer.release()

    # =========================
    # Export Events
    # =========================

    event_engine.export_events_json(
        str(EVENTS_PATH)
    )

    print("\n" + "=" * 50)
    print("✅ VISIONGUARD PIPELINE COMPLETED")
    print("=" * 50)

    print(
        f"📹 Output Video: "
        f"{OUTPUT_VIDEO_PATH}"
    )

    print(
        f"🚨 Events: "
        f"{EVENTS_PATH}"
    )

    return {
        "output_video": str(
            OUTPUT_VIDEO_PATH
        ),
        "events": str(
            EVENTS_PATH
        ),
        "total_frames": frame_idx
    }


# =========================
# Run from Terminal
# =========================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print(
            "Usage:\n"
            "python run_pipeline.py "
            "path/to/video.mp4"
        )

        sys.exit(1)

    video_path = sys.argv[1]

    run_vision_guard_pipeline(
        video_path
    )