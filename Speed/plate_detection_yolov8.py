from ultralytics import YOLO

# Load a model
model = YOLO("D:/code/Cropper/Speed/plate/best_plate.pt")  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(["D:/code/Cropper/Speed/car_plate.jpg", "D:/code/Cropper/Speed/car_plate.jpg"])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk
    result.save_crop("D:/code/Cropper/Speed/Number_plate")  # save to disk