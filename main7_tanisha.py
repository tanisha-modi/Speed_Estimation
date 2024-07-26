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

def run_script_1(folder_path):
    # Activate conda environment and run predict7.py
    script1 = r"ANPR/Automatic_Number_Plate_Detection_Recognition_YOLOv8/ultralytics/yolo/v8/detect/predict7.py"
    model_path = r"ANPR/Automatic_Number_Plate_Detection_Recognition_YOLOv8/ultralytics/yolo/v8/detect/best.pt"
    args1 = [f'model={model_path}', f'source={folder_path}']
    command = ["conda", "run", "--name", "ANPR", "python", script1] + args1
    subprocess.run(command, check=True)

def run_script_2():
    # Activate conda environment and run fixed_position.py
    script2 = r"Speed\fixed_position.py"
    command = ["conda", "run", "--name", "nodes", "python", script2]
    subprocess.run(command, check=True)

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
                print(f" - Found file: {file_path}")

            # Run the script on the newly created folder
            t3 = threading.Thread(target=run_script_1, args=(folder_path,))
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

if __name__ == '__main__':
    # file_name = 'D:/code/Cropper/Speed/final/violation_report.csv'
    file_name = 'D:/code/Cropper/Speed/video/v4/violation_report.csv'
    header = ['image_folder', 'number_plates']
    create_csv(file_name, header)

    # main_folder = r'D:/code/Cropper/Speed/video/v1/violations4'
    main_folder = r'D:/code/Cropper/Speed/video/v5/violations4'

    # Start the video processing thread
    t1 = threading.Thread(target=run_script_2)

    # Start the folder monitoring thread
    t2 = threading.Thread(target=monitor_folder, args=(main_folder,))

    # Start both threads
    t1.start()
    t2.start()

    # Wait for both threads to complete
    t1.join()
    t2.join()
