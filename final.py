import cv2
import inference
import supervision as sv
import threading
import time

annotator = sv.BoxAnnotator()

class_names_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J', 10:'K',
                    11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16:'Q', 17:'R', 18:'S', 19:'T', 20:'U',
                    21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

current_word = ""
letter_durations = {}  # Track duration for each letter
threshold_time = 10  # Set the threshold delay in seconds
no_detection_threshold = 3  # Set the threshold for word confidence
last_detection_time = time.time()  # Track the last detection timestamp

def render(predictions, image):

    detections = sv.Detections.from_roboflow(predictions)

    global current_word, letter_durations, word_duration, last_detection_time

    for detection in detections:
        letter = [class_names_dict[detections.class_id[i]] for i in range(len(detections))][0]
        duration = letter_durations.get(letter, 0) + 1  # Update duration
        letter_durations[letter] = duration

        if duration > threshold_time:
            current_word += letter
            letter_durations.clear()
            print("Saved letter:", letter)
            word_duration = 0
           

        if letter == " ":  # Or any other non-letter character
            if current_word != "":
                print("Word:", current_word)
                current_word = ""
                letter_durations.clear()  # Reset for the next word

        last_detection_time = time.time()  # Update last detection time

    text_size, _ = cv2.getTextSize(f"{current_word}", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)  # Get text size
    text_x = (image.shape[1] - text_size[0]) // 2  # Calculate horizontal center
    text_y = image.shape[0] - 15  # Position near bottom

    cv2.putText(image, f"{current_word}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    image = annotator.annotate(
        scene=image,
        detections=detections,
        labels=[f"{class_names_dict[detections.class_id[i]]} {detections.confidence[i]:0.2f}" for i in range(len(detections))]
    )

    cv2.imshow("Prediction", image)
    key = cv2.waitKey(1)
    if key == ord('q'):  # Close the program when 'q' is pressed
        cv2.destroyAllWindows()
        exit()

def check_no_detections():
    global current_word, word_duration, last_detection_time
    while True:
        time.sleep(0.5)  # Check every 0.5 seconds
        if time.time() - last_detection_time > no_detection_threshold and current_word != "":
            with lock:  # Acquire lock for thread safety
                print("Word:", current_word)
                current_word = ""
                word_duration = 0
                last_detection_time = time.time()

lock = threading.Lock()  # Create a lock for synchronization
no_detection_thread = threading.Thread(target=check_no_detections)
no_detection_thread.start()

inference.Stream(
    source="webcam",
    model="american-sign-language-letters/6",
    api_key="ds2uxDxQx48AI3lrbTqU",
    output_channel_order="BGR",
    use_main_thread=True,
    on_prediction=render,
    confidence=0.7,
    max_candidates=10,
    max_detections=5,
)


