from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")

results = model(
    "data/inputs/sample_video.mp4",
    conf=0.10,
    save=True,
    verbose=True
)

print("✅ Pose test finished")