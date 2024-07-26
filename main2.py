import subprocess

def run_script_in_conda_env(env_name, script_path):
    # Build the command to run the script in the specified conda environment
    command = f"conda activate {env_name} && python {script_path}"
    print(command)
    # Execute the command and capture the output with specified encoding
    result = subprocess.run(command, shell=True, capture_output=True, text=True, encoding='utf-8')
    return result.stdout, result.stderr

if __name__ == "__main__":
    # Names of the conda environments and paths to the scripts
    yolo_v1_env_name = 'nodes'
    yolo_v1_script_path = 'Speed/csv_write_final.py'
    
    yolo_v2_env_name = 'ANPR'
    yolo_v2_script_path = 'ANPR/Automatic_Number_Plate_Detection_Recognition_YOLOv8/ultralytics/yolo/v8/detect/predict3.py'
    
    # Run the scripts in their respective environments
    results_v1, errors_v1 = run_script_in_conda_env(yolo_v1_env_name, yolo_v1_script_path)
    results_v2, errors_v2 = run_script_in_conda_env(yolo_v2_env_name, yolo_v2_script_path)
    
    # Print the results
    print("Results from YOLO v1:")
    print("Hi")
    print(results_v1)
    if errors_v1:
        print("Errors from YOLO v1:")
        print(errors_v1)
    
    print("Results from YOLO v2:")
    print(results_v2)
    if errors_v2:
        print("Errors from YOLO v2:")
        print(errors_v2)
