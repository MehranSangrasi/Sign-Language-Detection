import time
import cv2
import threading
from ultralytics import YOLO

# Path to your trained YOLOv8 model
model_path = "best.pt"

# Initialize YOLOv8 model
model = YOLO(model_path)

# Declare last_detection_time as a global variable
global last_detection_time

# Initialize video capture
cap = cv2.VideoCapture(0)  # 0 for webcam, index for other cameras

# Define colors for bounding boxes and labels
colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

# Initialize variables for delay and accumulating words/sentences
delay_threshold = 5  # Set the threshold delay in seconds
last_detection_time = time.time()
detected_words = []


# Function for asynchronous inference
def async_inference():
    global last_detection_time
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame before processing
        resized_frame = cv2.resize(frame, (640, 640))
        results = model(resized_frame)

        # Process the results as needed
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
            cv2.putText(frame, text, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Accumulate detected words
            detected_words.append(model.names[int(cls)])

            # Check for delay threshold and print complete words/sentences on the frame
            if time.time() - last_detection_time > delay_threshold:
                complete_sentence = ' '.join(detected_words)
                cv2.putText(frame, complete_sentence, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                detected_words.clear()  # Reset detected words
                last_detection_time = time.time()

        # Display the frame with results
        cv2.imshow("Real-time Detection", frame)

        # Handle keyboard input
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Start the asynchronous inference thread
inference_thread = threading.Thread(target=async_inference)
inference_thread.start()

# Wait for the inference thread to finish
inference_thread.join()

# Release resources
cap.release()
cv2.destroyAllWindows()
