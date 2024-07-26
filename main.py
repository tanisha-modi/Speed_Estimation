import threading
import subprocess

def run_script(script_path, *args):
    command = ['python', script_path] + list(args)
    subprocess.run(command)

if __name__ == '__main__':
    script1 = 'ANPR/Automatic_Number_Plate_Detection_Recognition_YOLOv8/ultralytics/yolo/v8/detect/predict7.py'  # Replace with the full path to webcam_0.py
    # script2 = 'Speed\csv_write_final.py'  # Replace with the full path to webcam_1.py
    
    # Example arguments for webcam_0.py and webcam_1.py
    args1 = ['model=ANPR/Automatic_Number_Plate_Detection_Recognition_YOLOv8/ultralytics/yolo/v8/detect/best.pt', 'source=car2_trim.mp4']
    # args2 = ['--arg2', 'value2']

    t1 = threading.Thread(target=run_script, args=(script1,) + tuple(args1))
    # t2 = threading.Thread(target=run_script, args=(script2,))

    t1.start()
    # t2.start()

    t1.join()
    # t2.join()
