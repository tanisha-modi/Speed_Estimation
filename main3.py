import threading
import subprocess
import os
import csv

def run_script(script_path, *args):
    command = ['python', script_path] + list(args)
    subprocess.run(command)

def create_csv(file_name, header):
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # Write the header
        # writer.writerows(data)   # Write the initial data

if __name__ == '__main__':


    file_name = 'D:/code/Cropper/Speed/final/violation_report.csv'
    header = ['image_folder', 'number_plates']

    create_csv(file_name, header)

    script1 = 'ANPR/Automatic_Number_Plate_Detection_Recognition_YOLOv8/ultralytics/yolo/v8/detect/predict7.py'
    # script2 = 'Speed\csv_write_final.py'
    
    base_folder = 'D:/code/Cropper/cars/'  # Replace with the path to your base folder
    model_path = 'ANPR/Automatic_Number_Plate_Detection_Recognition_YOLOv8/ultralytics/yolo/v8/detect/best.pt'
    
    # Get all subfolders in the base folder
    subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir()]


    processed_folders = set()  # To keep track of processed folders

    for folder in subfolders:

        if folder in processed_folders:
            print(f"Skipping already processed folder: {folder}")
            continue

        args1 = [f'model={model_path}', f'source={folder}']
        # args2 = ['--arg2', 'value2']

        t1 = threading.Thread(target=run_script, args=(script1,) + tuple(args1))
        # t2 = threading.Thread(target=run_script, args=(script2,))

        t1.start()
        # t2.start()

        t1.join()
        # t2.join()

        processed_folders.add(folder)
        print(f"Finished processing folder: {folder}")

    print("All folders processed.")


# import threading
# import subprocess
# import os

# def run_script(script_path, *args):
#     command = ['python', script_path] + list(args)
#     print(f"Running command: {' '.join(command)}")  # Debug print
#     subprocess.run(command)

# if __name__ == '__main__':
#     script1 = 'ANPR/Automatic_Number_Plate_Detection_Recognition_YOLOv8/ultralytics/yolo/v8/detect/predict.py'
#     # script2 = 'Speed\csv_write_final.py'
    
#     base_folder = 'D:/code/Cropper/Speed/violations4'
#     model_path = 'ANPR/Automatic_Number_Plate_Detection_Recognition_YOLOv8/ultralytics/yolo/v8/detect/best.pt'
    
#     # Get all immediate subfolders in the base folder
#     subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir()]
    
#     print(f"Found {len(subfolders)} subfolders:")  # Debug print
#     for folder in subfolders:
#         print(f"  - {folder}")  # Debug print
    
#     processed_folders = set()  # To keep track of processed folders
    
#     for folder in subfolders:
#         if folder in processed_folders:
#             print(f"Skipping already processed folder: {folder}")
#             continue
        
#         args1 = [f'model={model_path}', f'source={folder}']
        
#         print(f"Processing folder: {folder}")  # Debug print
        
#         t1 = threading.Thread(target=run_script, args=(script1,) + tuple(args1))
#         t1.start()
#         t1.join()
        
#         processed_folders.add(folder)
#         print(f"Finished processing folder: {folder}")

#     print("All folders processed.")
#     print(f"Total folders processed: {len(processed_folders)}")  # Debug print


# import threading
# import subprocess
# import os

# log_file = 'processed_folders.log'

# def run_script(script_path, *args):
#     command = ['python', script_path] + list(args)
#     subprocess.run(command)

# def log_processed_folder(folder):
#     with open(log_file, 'a') as f:
#         f.write(folder + '\n')

# def is_folder_processed(folder):
#     if not os.path.exists(log_file):
#         return False
#     with open(log_file, 'r') as f:
#         processed_folders = f.read().splitlines()
#     return folder in processed_folders

# if __name__ == '__main__':
#     script1 = 'ANPR/Automatic_Number_Plate_Detection_Recognition_YOLOv8/ultralytics/yolo/v8/detect/predict.py'
#     # script2 = 'Speed/csv_write_final.py'
    
#     base_folder = 'D:/code/Cropper/Speed/violations4'  # Replace with the path to your base folder
#     model_path = 'ANPR/Automatic_Number_Plate_Detection_Recognition_YOLOv8/ultralytics/yolo/v8/detect/best.pt'
    
#     # Get all subfolders in the base folder
#     subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir()]
    
#     for folder in subfolders:
#         if is_folder_processed(folder):
#             print(f"Skipping already processed folder: {folder}")
#             continue
        
#         args1 = [f'model={model_path}', f'source={folder}']
#         # args2 = ['--arg2', 'value2']

#         t1 = threading.Thread(target=run_script, args=(script1,) + tuple(args1))
#         # t2 = threading.Thread(target=run_script, args=(script2,))

#         t1.start()
#         # t2.start()

#         t1.join()
#         # t2.join()

#         log_processed_folder(folder)
#         print(f"Finished processing folder: {folder}")

#     print("All folders processed.")



# import threading
# import subprocess
# import os

# def run_script(script_path, *args):
#     command = ['python', script_path] + list(args)
#     print("command : - ", command)
#     subprocess.run(command)

# def process_folder(script_path, model_path, folder):
#     args = [f'model={model_path}', f'source={folder}']
#     t = threading.Thread(target=run_script, args=(script_path,) + tuple(args))
#     t.start()
#     t.join()
#     print(f"Finished processing folder: {folder}")

# if __name__ == '__main__':
#     script1 = 'ANPR/Automatic_Number_Plate_Detection_Recognition_YOLOv8/ultralytics/yolo/v8/detect/predict.py'
#     base_folder = 'D:/code/Cropper/Speed/violations4/'  # Replace with the path to your base folder
#     model_path = 'ANPR/Automatic_Number_Plate_Detection_Recognition_YOLOv8/ultralytics/yolo/v8/detect/best.pt'
    
#     # Get all subfolders in the base folder
#     subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir()]

#     print(f"Found {len(subfolders)} subfolders:")
#     for folder in subfolders:
#         print(f"  - {folder}")

#     for folder in subfolders:
#         print(f"Processing folder: {folder}")
#         process_folder(script1, model_path, folder)

#     print("All folders processed.")
