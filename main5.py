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


def create_csv(file_name, header):
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # Write the header


class FolderHandler(FileSystemEventHandler):
    def __init__(self, main_folder, processed_folders):
        self.main_folder = main_folder
        self.processed_folders = processed_folders
        super().__init__()

    def on_any_event(self, event):
        if event.is_directory and event.event_type == 'created':
            print(f"New folder detected: {event.src_path}")
            self.iterate_folders(event.src_path)

    def iterate_folders(self, folder_path):
        if folder_path not in self.processed_folders:
            self.processed_folders.add(folder_path)
            print(f"Processing folder: {folder_path}")
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                print(f" - Found file: {file_path}")

            # Run the script on the newly created folder
            script1 = 'ANPR/Automatic_Number_Plate_Detection_Recognition_YOLOv8/ultralytics/yolo/v8/detect/predict7.py'
            model_path = 'ANPR/Automatic_Number_Plate_Detection_Recognition_YOLOv8/ultralytics/yolo/v8/detect/best.pt'
            args1 = [f'model={model_path}', f'source={folder_path}']
            t2 = threading.Thread(target=run_script, args=(script1,) + tuple(args1))
            t2.start()
            t2.join()


def monitor_folder(main_folder):
    processed_folders = set()  # To keep track of processed folders
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

    print("inside main5")
    file_name = 'D:/code/Cropper/Speed/final/violation_report.csv'
    header = ['image_folder', 'number_plates']

    create_csv(file_name, header)

    main_folder = 'D:/code/Cropper/Speed/final/violations4/'  # Replace with the path to your base folder

    monitor_folder(main_folder)
