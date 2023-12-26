import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the model
model = load_model("Testing Camera\\unet.h5")

# Load and preprocess the image
image = cv2.imread("captured_photo.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))
image = image / 255.0
image = np.expand_dims(image, axis=0)

# Apply the model to the image
output = model.predict(image)

# Display the original and output side by side
fig, axs = plt.subplots(1, 2)
axs[0].imshow(image[0])
axs[0].set_title('Original')
axs[0].axis('off')
axs[1].imshow(output[0])
axs[1].set_title('Output')
axs[1].axis('off')
plt.show()
