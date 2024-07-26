import cv2  # Ensure you have YOLOv8 installed and imported correctly

def run_yolov8_on_webcam_0():
    # model = YOLO('yolov8n.pt')  # Replace with your YOLOv8 model path
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # results = model.track(frame)
        # frame = results[0].plot()
        cv2.imshow('Webcam 0', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_yolov8_on_webcam_0()
