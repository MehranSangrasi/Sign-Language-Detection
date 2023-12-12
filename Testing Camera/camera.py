import cv2
import numpy as np

def capture_photo():
    # Open a connection to the webcam (usually 0 or 1, depending on your setup)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Couldn't open the webcam.")
        return

    while True:
        # Capture the current frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Couldn't capture a frame.")
            break

        # Convert the frame to the HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the range of color for your hands (you may need to adjust these values)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # Create a binary mask where your hands are white and the rest is black
        mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)

        # Bitwise AND the original frame with the mask to keep only the hands
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Display the result
        cv2.imshow("Webcam", result)

        # Check for the spacebar key press
        key = cv2.waitKey(1)
        if key == ord(' '):  # ' ' corresponds to the spacebar
            # Save the captured frame with only your hands visible
            cv2.imwrite("captured_hands.jpg", result)
            print("Hands captured successfully!")
            break

    # Release the webcam and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_photo()
