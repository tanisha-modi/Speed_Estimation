import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import subprocess
import csv

def run_script(script_path, *args):
    command = ['python', script_path] + list(args)
    subprocess.run(command)


def run_script_1():
    # Activate conda environment and run script_1.py
    subprocess.run(["conda", "run", "--name", "ANPR", "python", r"ANPR/Automatic_Number_Plate_Detection_Recognition_YOLOv8/ultralytics/yolo/v8/detect/predict7.py"], check=True)

def run_script_2():
    # Activate conda environment and run script_2.py
    subprocess.run(["conda", "run", "--name", "nodes", "python", r"Speed\fixed_position.py"], check=True)

def create_csv(file_name, header):
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

class FolderHandler(FileSystemEventHandler):
    def __init__(self, main_folder, processed_folders):
        self.main_folder = main_folder
        self.processed_folders = processed_folders
        super().__init__()

    def on_created(self, event):
        if event.is_directory:
            print(f"New folder detected: {event.src_path}")
            self.process_folder(event.src_path)

    def process_folder(self, folder_path):
        if folder_path not in self.processed_folders:
            self.processed_folders.add(folder_path)
            print(f"Processing folder: {folder_path}")
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                # print(f" - Found file: {file_path}")

            # Run the script on the newly created folder
            script1 = 'ANPR/Automatic_Number_Plate_Detection_Recognition_YOLOv8/ultralytics/yolo/v8/detect/predict7.py'
            model_path = 'ANPR/Automatic_Number_Plate_Detection_Recognition_YOLOv8/ultralytics/yolo/v8/detect/best.pt'
            args1 = [f'model={model_path}', f'source={folder_path}']
            run_script(script1, *args1)
            # run_script_1()
            t3 = threading.Thread(target=run_script_1, args=(script1,) + tuple(args1))
            t3.start()
            t3.join()

def monitor_folder(main_folder):
    processed_folders = set()
    event_handler = FolderHandler(main_folder, processed_folders)
    observer = Observer()
    observer.schedule(event_handler, main_folder, recursive=True)
    observer.start()
    print(f"Monitoring folder: {main_folder}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

def process_video(video_path, output_folder):
    # Replace this with your actual video processing code
    print(f"Processing video: {video_path}")
    # Simulate video processing and saving images
    time.sleep(5)
    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, 'dummy_violation.jpg'), 'w') as f:
        f.write("Dummy violation image")
    print(f"Saved violation images to: {output_folder}")

if __name__ == '__main__':
    file_name = 'D:/code/Cropper/Speed/final/violation_report.csv'
    header = ['image_folder', 'number_plates']
    create_csv(file_name, header)

    main_folder = 'D:/code/Cropper/Speed/final/violations4'
    video_path = 'path/to/your/video.mp4'  # Replace with your video path

    # monitor_folder(main_folder)
    # script2 = 'Speed\csv_write_final.py'
    # t1 = threading.Thread(target=run_script, args=(script2,))
    t1 = threading.Thread(target=run_script_2)
    t2 = threading.Thread(target=monitor_folder, args=(main_folder,))

    # Start the video processing thread
    t1.start()

    # Start the folder monitoring thread
    t2.start()

    # Wait for both threads to complete
    t1.join()
    t2.join()