from pathlib import Path
import cv2
import json
import pandas as pd
from ultralytics import YOLO

# 1. تحديد الجذر الرئيسي للمشروع (VISIONGUARD)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# 2. تحديد المسارات الديناميكية المحدثة
MODEL_PATH = ROOT_DIR / "models" / "yolo11n.pt"  # أو ROOT_DIR / "models" / "best.pt"
VIDEO_PATH = ROOT_DIR / "data" / "inputs" / "sample_video.mp4"

OUTPUT_VIDEO_PATH = ROOT_DIR / "data" / "outputs" / "tracking_output.mp4"
TRACKING_JSON_PATH = ROOT_DIR / "data" / "outputs" / "tracking_results.json"
TRACKING_CSV_PATH = ROOT_DIR / "data" / "outputs" / "tracking_results.csv"

# التأكد من وجود مجلد المخرجات
OUTPUT_VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)

# 3. تحميل النموذج
model = YOLO(str(MODEL_PATH))

# 4. فتح الفيديو
cap = cv2.VideoCapture(str(VIDEO_PATH))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# إعداد Video Writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(OUTPUT_VIDEO_PATH), fourcc, fps, (width, height))

frame_count = 0
tracking_history = []

print("Starting ByteTrack Tracking...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # 5. تشغيل YOLO + ByteTrack
    results = model.track(
        frame, persist=True, tracker="bytetrack.yaml", conf=0.25
    )

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()

        # استخراج البيانات
        for box, track_id, cls_id, conf in zip(
            boxes, track_ids, cls_ids, confs
        ):
            x1, y1, x2, y2 = map(float, box)
            class_name = model.names[cls_id]

            # حفظ البيانات للـ Event Engine
            tracking_history.append(
                {
                    "frame": frame_count,
                    "person_id": int(track_id),
                    "class": class_name,
                    "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    "confidence": round(float(conf), 2),
                }
            )

    # رسم النتائج وحفظ الفريم
    annotated_frame = results[0].plot(line_width=1, font_size=0.4)
    out.write(annotated_frame)

cap.release()
out.release()

# 6. تصدير البيانات للـ Event Engine
with open(TRACKING_JSON_PATH, "w") as f:
    json.dump(tracking_history, f, indent=4)

df = pd.DataFrame(tracking_history)
df.to_csv(TRACKING_CSV_PATH, index=False)

print(
    f"✅ Tracking Complete!\n- Video saved to: {OUTPUT_VIDEO_PATH}\n- JSON saved to: {TRACKING_JSON_PATH}\n- CSV saved to: {TRACKING_CSV_PATH}"
)