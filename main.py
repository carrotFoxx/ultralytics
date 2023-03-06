from ultralytics import YOLO
import cv2
model = YOLO("yolov5nu.pt")  # fast
# model = YOLO("best.pt")  # fast
# model = YOLO("yolov8n.pt")  # fast
# model = YOLO("path/to/best.pt")  # load a custom model


cap = cv2.VideoCapture(0) # Используем камеру с номером 0
# results = model("test1.mp4")  # predict on an image
# frame = cv2.imread('bus.jpg')


while True:
    ret, frame = cap.read()
    results = model(frame)  # predict on an image
    for box, clss, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
        class_name = results[0].names[int(clss)]
        x1, y1, x2, y2 = box.int().tolist()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f'{class_name} {conf:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # print('pass')
    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
