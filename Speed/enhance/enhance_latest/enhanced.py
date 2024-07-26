import cv2
import numpy as np
import os
import shutil
from ultralytics import YOLO, solutions
from datetime import datetime, timedelta

# Load YOLO model
model = YOLO("yolov8n.pt")

desired_classes = ['car', 'truck', 'bus', 'motorcycle'] 
class_indices = [i for i, name in model.names.items() if name in desired_classes]
print(f"Class indices: {class_indices}")

cap = cv2.VideoCapture("D:/code/Cropper/speed/video_final.mp4")

assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

video_writer = cv2.VideoWriter("D:/code/Cropper/speed/enhance_latest/speed_estimation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

line_pts = [(0, 270), (720, 270)]
SPEED_LIMIT = 10
violation_dir = "D:/code/Cropper/speed/enhance_latest/violations3"
temp_dir = "D:/code/Cropper/speed/enhance_latest/temp_vehicles"
os.makedirs(violation_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)

def report_violation(image, speed, vehicle_type, track_id):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{violation_dir}/violation_{vehicle_type}_{track_id}_{timestamp}.jpg"
    cv2.imwrite(filename, image)
    
    report = f"Violation Report:\nTime: {timestamp}\nVehicle Type: {vehicle_type}\nSpeed: {speed} km/h\nTrack ID: {track_id}\n"
    with open(f"{violation_dir}/violation_report.txt", "a") as f:
        f.write(report + "\n")
    
    print(f"Violation recorded: {filename}")

def del_temp(vehicle_id):
    vehicle_folder = os.path.join(temp_dir, f"vehicle_{vehicle_id}")
    if os.path.exists(vehicle_folder):
        shutil.rmtree(vehicle_folder)
        print(f"Deleted temporary folder for vehicle_{vehicle_id}")
    
    if vehicle_id in speed_obj.dist_data:
        del speed_obj.dist_data[vehicle_id]
    
    saved_ids.discard(vehicle_id)
    # all_ids.discard(vehicle_id)

def cleanup_temp_files(current_vehicles, frame_count, last_seen_threshold=5):
    for vehicle_id in list(saved_ids):  # Use list() to avoid modifying set during iteration
        if vehicle_id not in current_vehicles:
            if vehicle_id not in vehicle_last_seen:
                vehicle_last_seen[vehicle_id] = frame_count
            elif (frame_count - vehicle_last_seen[vehicle_id]) > last_seen_threshold:
                if vehicle_id not in reported_violations:
                    del_temp(vehicle_id)
                    del vehicle_last_seen[vehicle_id]
                    print(f"Removed vehicle {vehicle_id} due to inactivity")
        else:
            vehicle_last_seen[vehicle_id] = frame_count

def enhance_low_res_image(image):
    # Upscale the image
    height, width = image.shape[:2]
    upscaled = cv2.resize(image, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
    
    # Convert to grayscale
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equalized = clahe.apply(gray)
    
    # Denoise the image
    denoised = cv2.fastNlMeansDenoising(equalized, None, 10, 7, 21)
    
    # Sharpen the image
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    # Apply threshold to make text more visible
    _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Convert back to BGR color space
    enhanced = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    return enhanced

speed_obj = solutions.SpeedEstimator(reg_pts=line_pts, names=model.names, view_img=True)

MAX_SAVED_IMAGES = 30
reported_violations = set()
vehicle_last_seen = {}
frame_count = 0
all_ids = set()
saved_ids = set()

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video processing completed.")
        break

    frame_count += 1
    if frame_count % 5 != 0:
        continue

    print("frame : - ", frame_count)
    print("all_id : - ",  all_ids)
    print("saved_ids : - ",  saved_ids)

    im0 = cv2.resize(im0, (720, 540))
    original_frame = im0.copy()
    tracks = model.track(im0, persist=True, show=False, classes=class_indices, verbose=True, imgsz=1280, show_labels=False)
    im0 = speed_obj.estimate_speed(im0, tracks)
    
    current_vehicles = set()
    
    if tracks[0].boxes.id is not None:
        for box, cls, track_id in zip(tracks[0].boxes.xyxy, tracks[0].boxes.cls, tracks[0].boxes.id):
            track_id = int(track_id)
            current_vehicles.add(track_id)
            
            vehicle_folder = os.path.join(temp_dir, f"vehicle_{track_id}")
            if track_id not in all_ids:
                all_ids.add(track_id)
                saved_ids.add(track_id)
                os.makedirs(vehicle_folder, exist_ok=True)
                print(f"Folder created for id: {track_id}")


            if os.path.exists(vehicle_folder):
                saved_images = os.listdir(vehicle_folder)
                if len(saved_images) >= MAX_SAVED_IMAGES and track_id not in reported_violations:
                    print(f"Deleting {track_id} due to max images limit reached")
                    del_temp(track_id)
                else:
                    vehicle_type = model.names[int(cls)]
                    x1, y1, x2, y2 = map(int, box)
                    if x2 - x1 > 30 and y2 - y1 > 30 : 
                        vehicle_img = original_frame[y1:y2, x1:x2]
                        enhanced_img = enhance_low_res_image(vehicle_img)
                        cv2.imwrite(os.path.join(temp_dir, f"vehicle_{track_id}", f"frame_{frame_count}.jpg"), vehicle_img)

            if track_id in speed_obj.dist_data:
                speed = speed_obj.dist_data[track_id]
                if speed > SPEED_LIMIT and track_id not in reported_violations:
                    report_violation(vehicle_img, speed, vehicle_type, track_id)
                    reported_violations.add(track_id)
                elif speed < SPEED_LIMIT and track_id not in reported_violations:
                    print(f"Deleting {track_id} due to speed less than speed limit")
                    del_temp(track_id)

    print("current : - ", current_vehicles)
    cleanup_temp_files(current_vehicles, frame_count)

    annotated_frame = tracks[0].plot()
    video_writer.write(annotated_frame)
    cv2.imshow("frame", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()

print(f"Violation reports saved in {violation_dir}")
print(f"Temporary vehicle images saved in {temp_dir}")