import cv2
from roboflow import Roboflow

# Load the video
cap = cv2.VideoCapture(0)

# Initialize the Roboflow model
rf = Roboflow(api_key="ds2uxDxQx48AI3lrbTqU")
project = rf.workspace().project("american-sign-language-letters")
model = project.version(6).model

# Process each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the frame
    predictions = model.predict(frame, confidence=40, overlap=30).json()

    # Label the predictions in the frame
    for prediction in predictions["predictions"]:
        print(prediction)
        label = prediction['class']
        x = int(prediction['x'])
        y = int(prediction['y'])
        w = int(prediction['width'])
        h = int(prediction['height'])

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with predictions
    cv2.imshow("Sign Language Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()
