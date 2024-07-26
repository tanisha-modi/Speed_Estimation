import cv2
import os
import shutil
from ultralytics import YOLO, solutions
from datetime import datetime, timedelta


print("welcome to video processing script")
# Load YOLO model
model = YOLO("yolov8n.pt")

desired_classes = ['car', 'truck', 'bus', 'motorcycle'] 
class_indices = [i for i, name in model.names.items() if name in desired_classes]
# print(f"Class indices: {class_indices}")

cap = cv2.VideoCapture("D:/code/Cropper/Speed/video/v4.mp4")

assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

video_writer = cv2.VideoWriter("D:/code/Cropper/Speed/video/v5/speed_estimation.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (540, 860))

line_pts = [(0, 420), (720, 360)]
SPEED_LIMIT = 5
violation_dir = "D:/code/Cropper/Speed/video/v5/violations3"
violation_img = "D:/code/Cropper/Speed/video/v5/violations4"
temp_dir = "D:/code/Cropper/Speed/video/v5/temp_vehicles"
os.makedirs(violation_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(violation_img, exist_ok=True)

def report_violation(image, speed, vehicle_type, track_id):
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{violation_dir}/violation_{vehicle_type}_{track_id}_{timestamp}.jpg"
    # cv2.imwrite(filename, image)
    
    report = f"Violation Report:\nTime: {timestamp}\nVehicle Type: {vehicle_type}\nSpeed: {speed} km/h\nTrack ID: {track_id}\n"
    with open(f"{violation_dir}/violation_report.txt", "a") as f:
        f.write(report + "\n")
    
    print(f"Violation recorded: {filename}")

def del_temp(vehicle_id):
    vehicle_folder = os.path.join(temp_dir, f"vehicle_{vehicle_id}")
    if os.path.exists(vehicle_folder) and vehicle_id not in reported_violations:
        shutil.rmtree(vehicle_folder)
        print(f"Deleted temporary folder for vehicle_{vehicle_id}")
        
    if vehicle_id in speed_obj.dist_data:
        del speed_obj.dist_data[vehicle_id]
    
    saved_ids.discard(vehicle_id)
    # all_ids.discard(vehicle_id)

def transfer_violation_folder(source_dir, destination_dir, violation_id):
    try:
        # Search for the folder with the specific ID
        for folder in os.listdir(source_dir):
            violation_folder = f"vehicle_{violation_id}"
            if folder == violation_folder:
                source_path = os.path.join(source_dir, folder)
                destination_path = os.path.join(destination_dir, folder)
                
                # Move the folder
                shutil.move(source_path, destination_path)
                print(f"Folder {violation_id} moved successfully from {source_path} to {destination_path}")
                return True
        
        print(f"Folder with ID {violation_id} not found in {source_dir}")
        return False
    
    except FileNotFoundError:
        print(f"Source directory {source_dir} not found.")
    except PermissionError:
        print("Permission denied. Make sure you have the necessary permissions.")
    except shutil.Error as e:
        print(f"An error occurred: {e}")
    
    return False



def cleanup_temp_files(current_vehicles, frame_count, last_seen_threshold=10):
    for vehicle_id in list(saved_ids):  # Use list() to avoid modifying set during iteration
        if vehicle_id not in current_vehicles:
            last_frame, _, _ = vehicle_last_seen[vehicle_id]
            if (frame_count - last_frame) > last_seen_threshold:
                if vehicle_id not in reported_violations:
                    del_temp(vehicle_id)
                    del vehicle_last_seen[vehicle_id]
                    print(f"Removed vehicle {vehicle_id} due to inactivity")
                else :
                    saved_ids.discard(vehicle_id)
                    transfer_violation_folder(temp_dir, violation_img, vehicle_id)

speed_obj = solutions.SpeedEstimator(reg_pts=line_pts, names=model.names, view_img=True)

MAX_SAVED_IMAGES = 30
reported_violations = set()
vehicle_last_seen = {}
frame_count = 0
saved_ids = set()

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video processing completed.")
        break

    frame_count += 1
    if frame_count % 4 != 0:
        continue

    # print("frame : - ", frame_count)
    print("saved_ids : - ",  saved_ids)
    print("violation : - ", reported_violations)

    im0 = cv2.resize(im0, (540, 860))
    original_frame = im0.copy()
    tracks = model.track(im0, persist=True, show=False, classes=class_indices, verbose=True, imgsz=1280, show_labels=False)
    im0 = speed_obj.estimate_speed(im0, tracks)
    
    current_vehicles = set()
    
    if tracks[0].boxes.id is not None:
        for box, cls, track_id in zip(tracks[0].boxes.xyxy, tracks[0].boxes.cls, tracks[0].boxes.id):
            track_id = int(track_id)
            current_vehicles.add(track_id)
            
            vehicle_folder = os.path.join(temp_dir, f"vehicle_{track_id}")
            
            if track_id not in vehicle_last_seen.keys():
                saved_ids.add(track_id)
                os.makedirs(vehicle_folder, exist_ok=True)
                # print(f"Folder created for id: {track_id}")


            vehicle_type = model.names[int(cls)]
            x1, y1, x2, y2 = map(int, box)
            vehicle_img = original_frame[y1:y2, x1:x2]
            

            if os.path.exists(vehicle_folder):
                saved_images = os.listdir(vehicle_folder)
                if len(saved_images) >= MAX_SAVED_IMAGES and track_id not in reported_violations:
                    print(f"Deleting {track_id} due to max images limit reached")
                    del_temp(track_id)
                else:
                    if x2 - x1 > 30 and y2 - y1 > 30 : 
                        vehicle_img = original_frame[y1:y2, x1:x2]
                        if track_id in vehicle_last_seen.keys():
                            last_frame, last_x, last_y = vehicle_last_seen[track_id]
                            if x1 != last_x and y1 != last_y :
                                cv2.imwrite(os.path.join(temp_dir, f"vehicle_{track_id}", f"frame_{frame_count}.jpg"), vehicle_img)
                            else :
                                # print( track_id, " not saved due to fixed position")
                                pass
                        else :
                            cv2.imwrite(os.path.join(temp_dir, f"vehicle_{track_id}", f"frame_{frame_count}.jpg"), vehicle_img)
                    # print("not saved image, small size : -", track_id, " frame - ", frame_count)


            vehicle_last_seen[track_id] = [frame_count, x1, y1]

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