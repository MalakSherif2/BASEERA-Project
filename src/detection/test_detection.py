from pathlib import Path
from ultralytics import YOLO

# 1. تحديد المسار الرئيسي للمشروع ديناميكياً
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# 2. تحديد مسارات النموذج والصورة المحدثة
MODEL_PATH = ROOT_DIR / "models" / "yolo11n.pt"  # أو best.pt حسب التجربة
IMAGE_PATH = ROOT_DIR / "data" / "inputs" / "street_test.jpg"
OUTPUT_PATH = ROOT_DIR / "data" / "outputs" / "detection_test.jpg"

# التأكد من وجود مجلد المخرجات
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# 3. تحميل النموذج وتشغيله
model = YOLO(str(MODEL_PATH))
results = model(str(IMAGE_PATH), conf=0.25)

# 4. حفظ النتيجة للاختبار
for result in results:
    result.save(filename=str(OUTPUT_PATH))

print(f"✅ Detection test passed! Output saved to: {OUTPUT_PATH}")