import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf

class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight = None):
        return super().update_state(y_true, tf.where(y_pred > 0.5, 1, 0), sample_weight)

# Load the hand segmentation model
segmentation_model = load_model("Segmentation/best.h5")

# Load the sign language detection model
detection_model = load_model("Testing Camera/unet.h5s")


# Load the image
image = cv2.imread("path/to/your/image.jpg")

# Preprocess the image for the segmentation model (if needed)
# ...

# Segment the hand
mask = segmentation_model.predict(image)[0]  # Assuming single-image batch

# Find contours of the hand in the mask
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Assuming the largest contour is the hand
hand_contour = max(contours, key=cv2.contourArea)

# Get bounding box coordinates of the hand contour
x, y, w, h = cv2.boundingRect(hand_contour)

# Crop the hand region from the original image
cropped_hand = image[y:y+h, x:x+w]

# Preprocess the cropped hand image for the detection model (if needed)
# ...

# Predict the sign language gesture
prediction = detection_model.predict(cropped_hand)[0]

# Interpret the prediction (e.g., find the class with the highest probability)
# ...

# Display or use the prediction results
print(prediction)  # Replace with your desired output
