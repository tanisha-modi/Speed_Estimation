# Save this as main.py
from ultralytics.yolo.v8.detect.new import run_anpr

source = "D:/code/Cropper/Speed/violations4/vehicle_34"
ocr_results = run_anpr(source)

print("OCR Results:", ocr_results)