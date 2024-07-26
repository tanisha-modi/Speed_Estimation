import os
import cv2
import csv
import easyocr
from collections import Counter
import re
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Initialize EasyOCR
reader = easyocr.Reader(['en'], gpu=True)

# Function to validate license plate number format (customize as per your country's format)
def is_valid_license_plate(plate):
    pattern = r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$'
    return re.match(pattern, plate) is not None

# Function to perform OCR on an image
def ocr_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = reader.readtext(gray)
    text = ""
    for res in result:
        if len(res[1]) > 6 and res[2] > 0.2:
            text = res[1]
            break
    return text.upper().replace(" ", "")

# Function to process a single violation folder
def process_violation_folder(folder_path):
    images = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    plates = []
    
    for image_file in images:
        img_path = os.path.join(folder_path, image_file)
        img = cv2.imread(img_path)
        plate = ocr_image(img)
        if plate:
            plates.append(plate)
    
    if not plates:
        return None
    
    most_common_plate = Counter(plates).most_common(1)[0][0]
    
    if is_valid_license_plate(most_common_plate):
        return most_common_plate
    else:
        return None

# Function to append result to CSV
def append_to_csv(csv_file, violation_id, plate_number):
    with open(csv_file, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([violation_id, plate_number])

# Class to handle file system events
class ViolationHandler(FileSystemEventHandler):
    def __init__(self, csv_file, violations_folder):
        self.csv_file = csv_file
        self.processed_folders = set()
        self.violations_folder = violations_folder

    def on_moved(self, event):
        if event.is_directory and os.path.dirname(event.dest_path) == os.path.abspath(self.violations_folder):
            folder_path = event.dest_path
            violation_id = os.path.basename(folder_path)
            
            if violation_id not in self.processed_folders:
                self.processed_folders.add(violation_id)
                logger.info(f"Processing new violation folder: {violation_id}")
                
                # Wait a short time to ensure all files are transferred
                time.sleep(2)
                
                plate_number = process_violation_folder(folder_path)
                if plate_number:
                    append_to_csv(self.csv_file, violation_id, plate_number)
                    logger.info(f"Violation {violation_id} processed. Plate number: {plate_number}")
                else:
                    logger.warning(f"No valid license plate found for violation ID: {violation_id}")

# Main function to start monitoring
def monitor_violations(violations_folder, output_csv):
    print("predict run")
    if not os.path.exists(output_csv):
        with open(output_csv, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Violation ID', 'License Plate'])

    event_handler = ViolationHandler(output_csv, violations_folder)
    observer = Observer()
    observer.schedule(event_handler, path=os.path.dirname(violations_folder), recursive=False)
    observer.start()

    try:
        logger.info(f"Started monitoring {violations_folder}")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# Main function to initialize the script
def main():
    violations_folder = "D:/code/Cropper/speed/violations4"
    output_csv = "violations_license_plates.csv"
    monitor_violations(violations_folder, output_csv)

if __name__ == "__main__":
    main()
