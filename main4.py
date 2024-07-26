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
        # writer.writerows(data)   # Write the initial data

class FolderHandler(FileSystemEventHandler):
    def on_any_event(self, event):
        if event.is_directory and event.event_type == 'created':
            print(f"New folder detected: {event.src_path}")
            self.iterate_folders()

    def iterate_folders(self):
        for root, dirs, files in os.walk(main_folder):
            for folder in dirs:
                folder_path = os.path.join(root, folder)
                if folder_path not in processed_folders:
                    processed_folders.add(folder_path)

                    
                    print(f"Processing folder: {folder_path}")
                    for file in os.listdir(folder_path):
                        file_path = os.path.join(folder_path, file)
                        print(f" - Found file: {file_path}")

def monitor_folder(main_folder):
    event_handler = FolderHandler()
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


    file_name = 'D:/code/Cropper/Speed/final/violation_report.csv'
    header = ['image_folder', 'number_plates']

    create_csv(file_name, header)

    script1 = 'ANPR/Automatic_Number_Plate_Detection_Recognition_YOLOv8/ultralytics/yolo/v8/detect/predict7.py'
    # script2 = 'Speed\csv_write_final.py'
    
    base_folder = 'D:/code/Cropper/cars/'  # Replace with the path to your base folder
    model_path = 'ANPR/Automatic_Number_Plate_Detection_Recognition_YOLOv8/ultralytics/yolo/v8/detect/best.pt'
    
    # Get all subfolders in the base folder
    # subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir()]
    

    processed_folders = set()  # To keep track of processed folders
    main_folder = "D:/code/Cropper/cars/"

    # for folder in subfolders:

    #     if folder in processed_folders:
    #         print(f"Skipping already processed folder: {folder}")
    #         continue
    
        args1 = [f'model={model_path}', f'source={folder}']
        # args2 = ['--arg2', 'value2']

        t1 = threading.Thread(target=run_script, args=(script1,) + tuple(args1))
        # t2 = threading.Thread(target=run_script, args=(script2,))

        t1.start()
        # t2.start()

        t1.join()
        # t2.join()

        # print(f"Finished processing folder: {folder}")

    print("All folders processed.")