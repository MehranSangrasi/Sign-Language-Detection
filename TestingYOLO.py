import time
import cv2
from ultralytics import YOLO

# Path to your trained YOLOv8 model
model_path = "best.pt"

# Initialize YOLOv8 model
model = YOLO(model_path)

# Initialize video capture
cap = cv2.VideoCapture(0)  # 0 for webcam, index for other cameras

# Define colors for bounding boxes and labels
colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

while True:
    # Capture frame
    ret, frame = cap.read()
    if not ret:
        break

    # Predict results on the frame
    results = model(frame)

 # Loop through detected objects
    for result in results[0].boxes:
        # Extract bounding box coordinates and class
        x1, y1, x2, y2 = result.xyxy[0]

        cls = result.cls[0]
        conf = result.conf[0]

        # Get color based on class index
        color = colors[int(cls) % len(colors)]

        # Draw bounding box and label
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        text = f"{model.names[int(cls)]}: {conf:.2f}"
        cv2.putText(frame, text,  (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


    # Display the frame with results
    cv2.imshow("Real-time Detection", frame)


    # Handle keyboard input
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
