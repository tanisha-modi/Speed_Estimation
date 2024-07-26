import subprocess
import threading

def run_script_1():
    # Activate conda environment and run script_1.py
    subprocess.run(["conda", "run", "--name", "ANPR", "python", r"D:\code\Cropper\main5.py"], check=True)

def run_script_2():
    # Activate conda environment and run script_2.py
    subprocess.run(["conda", "run", "--name", "nodes", "python", r"Speed\fixed_position.py"], check=True)

# Create threads for each script
thread1 = threading.Thread(target=run_script_1)
thread2 = threading.Thread(target=run_script_2)

# Start the threads
thread2.start()
thread1.start()

# Wait for both threads to complete
thread2.join()
thread1.join()