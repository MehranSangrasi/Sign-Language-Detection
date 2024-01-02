import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Define the custom metric class
class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true, tf.where(y_pred > 0.5, 1, 0), sample_weight)

# Register the custom metric class
custom_objects = {'MyMeanIOU': MyMeanIOU}

# Load the segmentation model
model = load_model('best.h5', custom_objects=custom_objects)

# Open a connection to the camera (you may need to change the index)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame")
        break

    # Resize the frame to match the input size of the model
    input_size = (model.input_shape[1], model.input_shape[2])
    resized_frame = cv2.resize(frame, input_size)

    # Preprocess the frame for the model
    input_data = np.expand_dims(resized_frame, axis=0) / 255.0

    # Perform segmentation
    segmentation_map = model.predict(input_data)[0, :, :, 0]

    # Apply a threshold to create a binary mask
    mask = (segmentation_map > 0.5).astype(np.uint8)
    
    # Resize the mask to match the original frame
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

    # Apply the mask to the original frame
    segmented_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Display the original frame and segmented frame
    cv2.imshow('Original', frame)
    cv2.imshow('Segmented', segmented_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
