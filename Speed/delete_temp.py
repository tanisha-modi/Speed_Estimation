import cv2
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

video_writer = cv2.VideoWriter("D:/code/Cropper/speed_estimation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

line_pts = [(0, 270), (720, 270)]
SPEED_LIMIT = 10
violation_dir = "D:/code/Cropper/speed/violations3"
temp_dir = "D:/code/Cropper/speed/temp_vehicles"
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
            print("deleted ",  f"vehicle_{vehicle_id}")

        if vehicle_id in speed_obj.dist_data:
            del speed_obj.dist_data[vehicle_id]
        # del vehicle_last_seen[vehicle_id]

speed_obj = solutions.SpeedEstimator(reg_pts=line_pts, names=model.names, view_img=True)

reported_violations = set()
vehicle_last_seen = {}
frame_count = 0
all_ids = set()

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video processing completed.")
        break

    frame_count += 1

    if frame_count % 5 != 0:
        continue

    print("frame : - ", frame_count)
    im0=cv2.resize(im0 , (720, 540)) 
    original_frame = im0.copy()
    tracks = model.track(im0, persist=True, show=False, classes=class_indices, verbose=True, imgsz=1280, show_labels=False)
    im0 = speed_obj.estimate_speed(im0, tracks)
    
    if tracks[0].boxes.id is not None:
        for box, cls, track_id in zip(tracks[0].boxes.xyxy, tracks[0].boxes.cls, tracks[0].boxes.id):
            track_id = int(track_id)

            vehicle_folder = os.path.join(temp_dir, f"vehicle_{track_id}")
            if track_id not in all_ids :
                print("folder created for id : ", track_id)
                os.makedirs(vehicle_folder, exist_ok=True)

            
            all_ids.add(track_id)
            print(all_ids)
            vehicle_type = model.names[int(cls)]
            
            if os.path.isdir(vehicle_folder):
                x1, y1, x2, y2 = map(int, box)
                vehicle_img = original_frame[y1:y2, x1:x2]
                cv2.imwrite(os.path.join(vehicle_folder, f"frame_{frame_count}.jpg"), vehicle_img)

            
            if track_id in speed_obj.dist_data:
                speed = speed_obj.dist_data[track_id]
                if speed > SPEED_LIMIT and track_id not in reported_violations:
                    report_violation(vehicle_img, speed, vehicle_type, track_id)
                    reported_violations.add(track_id)
                elif speed < SPEED_LIMIT :
                    del_temp(track_id)
                    

    annotated_frame = tracks[0].plot()

    video_writer.write(annotated_frame)
    cv2.imshow("frame", annotated_frame)

cap.release()
video_writer.release()
cv2.destroyAllWindows()

print(f"Violation reports saved in {violation_dir}")
print(f"Temporary vehicle images saved in {temp_dir}")