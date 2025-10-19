from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/fruit-detection/weights/best.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở webcam")
    exit()

print("Bắt đầu nhận diện (nhấn Q để thoát)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc frame từ webcam")
        break

    results = model.predict(
        source=frame,  
        conf=0.88,      
        verbose=False
    )

    annotated_frame = results[0].plot()

    cv2.imshow("Fruit Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Đã dừng nhận diện")
