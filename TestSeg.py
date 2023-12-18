import cv2
import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model('ASL_model.h5')

# Define desired image size
image_size = (224, 224)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Define a function to pre-process the frame
def preprocess_frame(frame):
    # Rescale image
    frame = frame / 255.0

    # Resize to target size
    frame = cv2.resize(frame, image_size)

    # Apply any additional preprocessing
    # (e.g., horizontal flip, normalization)

    # Expand dimensions for prediction
    frame = np.expand_dims(frame, axis=0)

    return frame

while True:
    # Capture frame
    ret, frame = cap.read()

    # Preprocess frame
    preprocessed_frame = preprocess_frame(frame)

    # Predict on the preprocessed frame using your h5 model
    prediction = model.predict(preprocessed_frame)[0]
    
    class_names = {
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
        10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I',
        19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R',
        28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'
    }
    
    prediction = class_names[np.argmax(prediction)]
    
    print(prediction)
    
    # find the index of the label with largest corresponding probability
    

    # Implement your logic to interpret and use the prediction based on your model and task

    # Optionally show results on frame
    cv2.imshow("Frame", frame)

    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
