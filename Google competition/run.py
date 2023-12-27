import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

def decode_predictions(predictions):
    # Assuming predictions are indices, convert them to characters
    # Update this function according to your model's output format
    # Example: return ''.join([index_to_char[idx] for idx in predictions])
    return ''.join([str(chr(65 + int(idx))) for idx in predictions])  # Example logic


# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        # Initialize an array to hold landmarks data
        landmarks = []

        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

        # Pad the landmarks if necessary
        if len(landmarks) < 156:
            landmarks.extend([0] * (156 - len(landmarks)))

        # Convert to numpy array and reshape for the model
        input_data = np.array(landmarks, dtype=np.float32).reshape(1, -1)

        # Ensure the landmarks shape matches the model's expected input
        if input_data.shape[1] != 156:
            print(f"Error: Expected 156 elements, but got {input_data.shape[1]}")
            continue

        # Set input tensor and run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_phrase = decode_predictions(output_data[0])

        cv2.putText(frame, predicted_phrase, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('ASL Fingerspelling Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
