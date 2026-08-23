# استخدام نسخة خفيفة من Python
FROM python:3.10-slim

# ضبط مسار العمل داخل الحاوية
WORKDIR /app

# تثبيت المكتبات الأساسية لمعالجة الصور (OpenCV)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# نسخ ملفات التبعيات وتثبيتها
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ باقي ملفات المشروع
COPY . .

# تشغيل السيرفر
CMD ["python", "server.py"]