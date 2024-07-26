import cv2
from ultralytics import YOLO, solutions

# Load YOLO model
model = YOLO("yolov8n.pt")

# Define the classes we want to detect
desired_classes = ['car', 'truck', 'bus']

# Get the indices of the desired classes
class_indices = [i for i, name in model.names.items() if name in desired_classes]
print(f"Class indices: {class_indices}")

cap = cv2.VideoCapture("D:/code/Cropper/video.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter("D:/code/Cropper/speed_estimation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

line_pts = [(0, 360), (1280, 360)]

# Init speed-estimation obj
speed_obj = solutions.SpeedEstimator(
    reg_pts=line_pts,
    names=model.names,  # Use the full names dictionary from the model
    view_img=True,
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Run YOLO tracking with class filtering
    tracks = model.track(im0, persist=True, show=False, classes=class_indices)

    im0 = speed_obj.estimate_speed(im0, tracks)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()