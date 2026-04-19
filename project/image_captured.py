import cv2
import time
import os

IP = "192.168.0.77"
CANDIDATE_URLS = [
    f"http://{IP}:8080/video",
    f"http://{IP}:8080/mjpeg",
    f"http://{IP}:4747/video",  # بعض التطبيقات (مثل DroidCam/بدائل)
]

cap = None
used_url = None

for url in CANDIDATE_URLS:
    test = cv2.VideoCapture(url)
    if test.isOpened():
        cap = test
        used_url = url
        break
    test.release()

if cap is None:
    print("❌ لم يتم فتح الكاميرا. جرّب فتح رابط الكاميرا في المتصفح للتأكد من المسار والمنفذ.")
    exit()

print("✅ تم فتح الكاميرا:", used_url)

os.makedirs("captured_images", exist_ok=True)
i = 201

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("⚠️ فشل قراءة الإطار")
            break

        filename = f"captured_images/SHAMLAN/img_SHAMLAN_{i}.jpg"
        cv2.imwrite(filename, frame)
        print("✅ حفظ:", filename)

        i += 1
        time.sleep(1)

except KeyboardInterrupt:
    print("\n⛔ تم الإيقاف")

cap.release()
