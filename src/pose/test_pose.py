from pathlib import Path
from ultralytics import YOLO

# 1. تحديد المسار الرئيسي للمشروع (VISIONGUARD)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# 2. تحديد المسارات (سيتم تحميل yolo11n-pose.pt تلقائياً إذا لم يكن موجوداً)
MODEL_PATH = ROOT_DIR / "models" / "yolo11n-pose.pt"
IMAGE_PATH = ROOT_DIR / "data" / "inputs" / "street_test.jpg"
OUTPUT_PATH = ROOT_DIR / "data" / "outputs" / "pose_test.jpg"

# 3. تحميل نموذج Pose
model = YOLO(str(MODEL_PATH))

# 4. تشغيل الاستنتاج لاستخراج الهيكل العظمي والنقاط
results = model(str(IMAGE_PATH), conf=0.15)

# 5. طباعة عدد الأشخاص والنقاط المفصلية المكتشفة
for result in results:
    if result.keypoints is not None:
        print(f" Detected keypoints shape: {result.keypoints.data.shape}")
    result.save(filename=str(OUTPUT_PATH))

print(f"✅ Pose Estimation test passed! Output saved to: {OUTPUT_PATH}")