# pip install ultralytics opencv-python
from ultralytics import YOLO
import cv2

# استخدم وزن PyTorch أو ONNX (نفس السطر يقبل الاثنين)
model = YOLO("yolov8n.pt")          # أو "yolov8n.onnx"

cap = cv2.VideoCapture(0)            # كاميرا الويب (بدّل 0 لو عندك أكثر من كاميرا)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    results = model(frame, imgsz=640, conf=0.25)
    annotated = results[0].plot()    # يرسم الصناديق والتسميات

    cv2.imshow("YOLO Test", annotated)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC للخروج
        break

cap.release()
cv2.destroyAllWindows()
