import cv2
from ultralytics import YOLO, solutions

print("OpenCV version:", cv2.__version__)

# Load the YOLO model
model = YOLO("yolov8n.pt")
names = model.model.names

# Open the video file
cap = cv2.VideoCapture("D:/code/Cropper/video.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter("D:/code/Cropper/speed_estimation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Define line points
line_pts = [(0, 360), (1280, 360)]

# Initialize speed estimation object
speed_obj = solutions.SpeedEstimator(
    reg_pts=line_pts,
    names=names, 
    view_img=True,
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Perform object tracking
    tracks = model.track(im0, persist=True, show=False)

    # Estimate speed
    im0 = speed_obj.estimate_speed(im0, tracks)

    # Write the processed frame to the video writer
    video_writer.write(im0)

    # Display the processed frame
    cv2.imshow("Processed Video", im0)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
video_writer.release()
cv2.destroyAllWindows()
