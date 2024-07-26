import cv2
import os
import shutil
import csv
from ultralytics import YOLO, solutions
from datetime import datetime

# Function to report violation
def report_violation(image, speed, vehicle_type, track_id, violation_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Log to CSV
    csv_file = os.path.join(violation_dir, "violation_report.csv")
    header = ['Timestamp', 'Vehicle Type', 'Speed (km/h)', 'Track ID']
    data = [timestamp, vehicle_type, speed, track_id]
    
    # Check if the CSV file already exists
    file_exists = os.path.isfile(csv_file)
    
    # Write the data to CSV
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)  # Write the header if the file is new
        writer.writerow(data)
    
    # Print statement to confirm logging
    print(f"Violation logged: {data}")

# Function to delete temporary vehicle data
def del_temp(vehicle_id, temp_dir, reported_violations, speed_obj, saved_ids):
    vehicle_folder = os.path.join(temp_dir, f"vehicle_{vehicle_id}")
    if os.path.exists(vehicle_folder) and vehicle_id not in reported_violations:
        shutil.rmtree(vehicle_folder)
        print(f"Deleted temporary folder for vehicle_{vehicle_id}")

    if vehicle_id in speed_obj.dist_data:
        del speed_obj.dist_data[vehicle_id]

    saved_ids.discard(vehicle_id)

# Function to transfer violation folder
def transfer_violation_folder(source_dir, destination_dir, violation_id):
    try:
        violation_folder = f"vehicle_{violation_id}"
        source_path = os.path.join(source_dir, violation_folder)
        destination_path = os.path.join(destination_dir, violation_folder)

        if os.path.exists(source_path):
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

# Function to clean up temporary files
def cleanup_temp_files(current_vehicles, frame_count, last_seen_threshold, temp_dir, reported_violations, vehicle_last_seen, saved_ids, violation_img, speed_obj):
    for vehicle_id in list(saved_ids):
        if vehicle_id not in current_vehicles:
            last_frame, _, _ = vehicle_last_seen[vehicle_id]
            if (frame_count - last_frame) > last_seen_threshold:
                if vehicle_id not in reported_violations:
                    del_temp(vehicle_id, temp_dir, reported_violations, speed_obj, saved_ids)
                    del vehicle_last_seen[vehicle_id]
                    print(f"Removed vehicle {vehicle_id} due to inactivity")
                else:
                    saved_ids.discard(vehicle_id)
                    transfer_violation_folder(temp_dir, violation_img, vehicle_id)

# Main function
def main():
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
    SPEED_LIMIT = 14
    violation_dir = "D:/code/Cropper/speed/final/violations"
    violation_img = "D:/code/Cropper/speed/final/violations4"
    temp_dir = "D:/code/Cropper/speed/final/temp_vehicles"
    os.makedirs(violation_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(violation_img, exist_ok=True)

    speed_obj = solutions.SpeedEstimator(reg_pts=line_pts, names=model.names, view_img=True)

    MAX_SAVED_IMAGES = 50
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

        print(f"Frame: {frame_count}")
        print(f"Saved IDs: {saved_ids}")
        print(f"Violations: {reported_violations}")

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

                if track_id not in vehicle_last_seen.keys():
                    saved_ids.add(track_id)
                    os.makedirs(vehicle_folder, exist_ok=True)
                    print(f"Folder created for ID: {track_id}")

                vehicle_type = model.names[int(cls)]
                x1, y1, x2, y2 = map(int, box)
                vehicle_img = original_frame[y1:y2, x1:x2]

                if os.path.exists(vehicle_folder):
                    saved_images = os.listdir(vehicle_folder)
                    if len(saved_images) >= MAX_SAVED_IMAGES and track_id not in reported_violations:
                        print(f"Deleting {track_id} due to max images limit reached")
                        del_temp(track_id, temp_dir, reported_violations, speed_obj, saved_ids)
                    else:
                        if x2 - x1 > 30 and y2 - y1 > 30:
                            if track_id in vehicle_last_seen.keys():
                                last_frame, last_x, last_y = vehicle_last_seen[track_id]
                                if x1 != last_x or y1 != last_y:
                                    cv2.imwrite(os.path.join(temp_dir, f"vehicle_{track_id}", f"frame_{frame_count}.jpg"), vehicle_img)
                                else:
                                    print(f"{track_id} not saved due to fixed position")
                            else:
                                cv2.imwrite(os.path.join(temp_dir, f"vehicle_{track_id}", f"frame_{frame_count}.jpg"), vehicle_img)

                vehicle_last_seen[track_id] = [frame_count, x1, y1]

                if track_id in speed_obj.dist_data:
                    speed = speed_obj.dist_data[track_id]
                    if speed > SPEED_LIMIT and track_id not in reported_violations:
                        report_violation(vehicle_img, speed, vehicle_type, track_id, violation_dir)
                        reported_violations.add(track_id)
                    elif speed < SPEED_LIMIT and track_id not in reported_violations:
                        print(f"Deleting {track_id} due to speed less than speed limit")
                        del_temp(track_id, temp_dir, reported_violations, speed_obj, saved_ids)

        print(f"Current: {current_vehicles}")
        cleanup_temp_files(current_vehicles, frame_count, 10, temp_dir, reported_violations, vehicle_last_seen, saved_ids, violation_img, speed_obj)

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

if __name__ == "__main__":
    main()
