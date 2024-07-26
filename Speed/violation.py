# import cv2
# import os
# from ultralytics import YOLO, solutions
# from datetime import datetime

# # Load YOLO model
# model = YOLO("yolov8n.pt")

# # Define the classes we want to detect
# desired_classes = ['car', 'truck', 'bus']

# # Get the indices of the desired classes
# class_indices = [i for i, name in model.names.items() if name in desired_classes]
# print(f"Class indices: {class_indices}")

# cap = cv2.VideoCapture("D:/code/Cropper/video.mp4")
# assert cap.isOpened(), "Error reading video file"
# w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# # Video writer
# video_writer = cv2.VideoWriter("D:/code/Cropper/speed_estimation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# line_pts = [(0, 360), (1280, 360)]

# # Set speed limit (km/h)
# SPEED_LIMIT = 40

# # Create a directory to save violation images
# violation_dir = "D:/code/Cropper/speed/violations"
# os.makedirs(violation_dir, exist_ok=True)

# # Function to save violation image and report
# def report_violation(image, speed, vehicle_type, track_id):
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"{violation_dir}/violation_{vehicle_type}_{track_id}_{timestamp}.jpg"
#     cv2.imwrite(filename, image)
    
#     report = f"Violation Report:\nTime: {timestamp}\nVehicle Type: {vehicle_type}\nSpeed: {speed} km/h\nTrack ID: {track_id}\n"
#     with open(f"{violation_dir}/violation_report.txt", "a") as f:
#         f.write(report + "\n")
    
#     print(f"Violation recorded: {filename}")

# # Init speed-estimation obj
# speed_obj = solutions.SpeedEstimator(
#     reg_pts=line_pts,
#     names=model.names,
#     view_img=True,
# )

# # Keep track of reported violations
# reported_violations = set()

# while cap.isOpened():
#     success, im0 = cap.read()
#     if not success:
#         print("Video frame is empty or video processing has been successfully completed.")
#         break

#     # Run YOLO tracking with class filtering
#     tracks = model.track(im0, persist=True, show=False, classes=class_indices)

#     im0 = speed_obj.estimate_speed(im0, tracks)
    
#     # Check for speed violations
#     for track in tracks:
#         if track.id is None:
#             continue
        
#         track_id = int(track.id)
#         if track_id in speed_obj.dist_data:
#             speed = speed_obj.dist_data[track_id]
#             if speed > SPEED_LIMIT and track_id not in reported_violations:
#                 # Get vehicle type
#                 cls = int(track.boxes.cls[0])
#                 vehicle_type = model.names[cls]
                
#                 # Crop the image to the vehicle
#                 x1, y1, x2, y2 = map(int, track.boxes.xyxy[0])
#                 vehicle_img = im0[y1:y2, x1:x2]
                
#                 # Report violation
#                 report_violation(vehicle_img, speed, vehicle_type, track_id)
#                 reported_violations.add(track_id)

#     video_writer.write(im0)

# cap.release()
# video_writer.release()
# cv2.destroyAllWindows()

# print(f"Violation reports saved in {violation_dir}")



import cv2
import os
from ultralytics import YOLO, solutions
from datetime import datetime

# Load YOLO model
model = YOLO("yolov8n.pt")

# Define the classes we want to detect
desired_classes = ['car', 'truck', 'bus', 'motorcycle']

# Get the indices of the desired classes
class_indices = [i for i, name in model.names.items() if name in desired_classes]
print(f"Class indices: {class_indices}")

cap = cv2.VideoCapture("D:/code/Cropper/video.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter("D:/code/Cropper/speed_estimation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

line_pts = [(0, 360), (1280, 360)]

# Set speed limit (km/h)
SPEED_LIMIT = 20

# Create a directory to save violation images
violation_dir = "D:/code/Cropper/speed/violations2"
os.makedirs(violation_dir, exist_ok=True)

# Function to save violation image and report
def report_violation(image, speed, vehicle_type, track_id):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{violation_dir}/violation_{vehicle_type}_{track_id}_{timestamp}.jpg"
    cv2.imwrite(filename, image)
    
    report = f"Violation Report:\nTime: {timestamp}\nVehicle Type: {vehicle_type}\nSpeed: {speed} km/h\nTrack ID: {track_id}\n"
    with open(f"{violation_dir}/violation_report.txt", "a") as f:
        f.write(report + "\n")
    
    print(f"Violation recorded: {filename}")

# Init speed-estimation obj
speed_obj = solutions.SpeedEstimator(
    reg_pts=line_pts,
    names=model.names,
    view_img=True,
)

# Keep track of reported violations
reported_violations = set()

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    original_frame = im0.copy()

    # Run YOLO tracking with class filtering
    tracks = model.track(im0, persist=True, show=False, classes=class_indices, verbose=True, imgsz=1280)

    im0 = speed_obj.estimate_speed(im0, tracks)
    
    # Check for speed violations
    if tracks[0].boxes.id is not None:
        for box, cls, track_id in zip(tracks[0].boxes.xyxy, tracks[0].boxes.cls, tracks[0].boxes.id):
            track_id = int(track_id)
            if track_id in speed_obj.dist_data:
                speed = speed_obj.dist_data[track_id]
                if speed > SPEED_LIMIT and track_id not in reported_violations:
                    # Get vehicle type
                    vehicle_type = model.names[int(cls)]
                    
                    # Crop the image to the vehicle
                    x1, y1, x2, y2 = map(int, box)
                    vehicle_img = original_frame[y1:y2, x1:x2]
                    
                    # Report violation
                    report_violation(vehicle_img, speed, vehicle_type, track_id)
                    reported_violations.add(track_id)

    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()

print(f"Violation reports saved in {violation_dir}")