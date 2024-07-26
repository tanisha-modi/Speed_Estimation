import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

main_folder = "D:/code/Cropper/cars/"
processed_folders = set()

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

if __name__ == "__main__":
    monitor_folder(main_folder)