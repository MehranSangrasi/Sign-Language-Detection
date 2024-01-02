import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import json

# Load inference arguments from JSON file
with open('inference_args.json', 'r') as file:
    inference_args = json.load(file)
selected_columns = inference_args['selected_columns']

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Function to extract and order landmarks
def extract_and_order_landmarks(hand_results):
    landmarks = np.zeros((21 * 3 * 2,))  # 21 landmarks, 3 coordinates, 2 hands

    # Process right hand landmarks
    if hand_results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            for idx, lm in enumerate(hand_landmarks.landmark):
                if hand_idx == 0:  # Assuming first hand is right
                    base_idx = idx * 3
                else:  # Assuming second hand is left
                    base_idx = 21 * 3 + idx * 3
                landmarks[base_idx] = lm.x
                landmarks[base_idx + 1] = lm.y
                landmarks[base_idx + 2] = lm.z

    # Reorder according to selected_columns (currently only handles hand landmarks)
    ordered_landmarks = []
    for col in selected_columns:
        if "right_hand" in col or "left_hand" in col:
            idx = int(col.split('_')[-1])  # Extract landmark index
            if "x_" in col:
                ordered_landmarks.append(landmarks[idx * 3])
            elif "y_" in col:
                ordered_landmarks.append(landmarks[idx * 3 + 1])
            elif "z_" in col:
                ordered_landmarks.append(landmarks[idx * 3 + 2])

    return ordered_landmarks

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(frame_rgb)

    landmarks = extract_and_order_landmarks(hand_results)
    input_data = np.array(landmarks, dtype=np.float32).reshape(1, -1)

    if input_data.shape[1] != len(selected_columns):
        print(f"Error: Expected {len(selected_columns)} elements, but got {input_data.shape[1]}")
        continue

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
