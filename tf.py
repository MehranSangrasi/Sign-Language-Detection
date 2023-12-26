import cv2
import inference
import supervision as sv

annotator = sv.BoxAnnotator()

class_names_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J', 10:'K',
                    11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16:'Q', 17:'R', 18:'S', 19:'T', 20:'U',
                    21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

def render(predictions, image):

    detections = sv.Detections.from_roboflow(predictions)

    # print(detections)  # Output will now show class names and confidences


    image = annotator.annotate(
        scene=image,
        detections=detections,
        labels= [f"{class_names_dict[detections.class_id[i]]} {detections.confidence[i]:0.2f}" for i in range(len(detections))]

    )

    cv2.imshow("Prediction", image)
    key = cv2.waitKey(1)
    if key == ord('q'):  # Close the program when 'q' is pressed
        cv2.destroyAllWindows()
        exit()

inference.Stream(
    source="webcam",
    model="american-sign-language-letters/6",
    api_key="ds2uxDxQx48AI3lrbTqU",
    output_channel_order="BGR",
    use_main_thread=True,
    on_prediction=render,
    confidence=0.5,
    max_candidates=20,
    max_detections=10,
)
