import time
import cv2
import numpy as np
import onnxruntime
from ultralytics import YOLO
import sys
if 'google.colab' in sys.modules:
    !pip install ultralytics onnxruntime opencv-python numpy
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
model_path = "yolov8n.onnx"  
yolov8_detector = YOLO(model_path)

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    results = yolov8_detector(frame)  
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            conf = box[4].item()
            if conf > 0.7: 
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Conf: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("Detected Objects", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()