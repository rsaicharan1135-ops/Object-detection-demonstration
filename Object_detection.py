
import cv2
import pandas as pd
import argparse
from ultralytics import YOLO

# -----------------------------------------
# 1. LOAD PRETRAINED YOLO MODEL
# -----------------------------------------

model = YOLO("yolov8n.pt")

# -----------------------------------------
# 2. IOU CALCULATION
# -----------------------------------------

def calculate_iou(box1, box2):
    """
    Each box format: [x1, y1, x2, y2]
    """

    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])

    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    intersection = max(0, x_right - x_left) * \
                   max(0, y_bottom - y_top)

    area1 = max(0, box1[2] - box1[0]) * \
            max(0, box1[3] - box1[1])

    area2 = max(0, box2[2] - box2[0]) * \
            max(0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


# -----------------------------------------
# 3. OBJECT DETECTION FUNCTION
# -----------------------------------------

def detect_objects(frame, confidence=0.40, nms_iou=0.45):

    results = model.predict(
        source=frame,
        conf=confidence,
        iou=nms_iou,
        verbose=False
    )

    result = results[0]
    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            score = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            detections.append({
                "Object": class_name,
                "Confidence": round(score, 4),
                "X1": x1,
                "Y1": y1,
                "X2": x2,
                "Y2": y2
            })

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Display label and confidence
            label = f"{class_name}: {score:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    return frame, detections


# -----------------------------------------
# 4. IMAGE DETECTION
# -----------------------------------------

def process_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not load image.")
        return

    output, detections = detect_objects(image)

    cv2.imwrite("detected_output.jpg", output)

    cv2.imshow("Object Detection", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save detection report
    df = pd.DataFrame(detections)
    df.to_csv("detection_report.csv", index=False)

    print("\nDETECTION RESULTS")
    print(df)

    print("\nOutput saved:")
    print("1. detected_output.jpg")
    print("2. detection_report.csv")


# -----------------------------------------
# 5. LIVE WEBCAM DETECTION
# -----------------------------------------

def process_webcam():

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Webcam not detected.")
        return

    all_detections = []
    frame_number = 0

    print("Webcam started. Press Q to stop.")

    while True:

        success, frame = camera.read()

        if not success:
            print("Unable to read camera.")
            break

        frame_number += 1

        output, detections = detect_objects(frame)

        for item in detections:
            item["Frame"] = frame_number
            all_detections.append(item)

        cv2.imshow("Live Object Detection", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    # Save webcam detection report
    df = pd.DataFrame(all_detections)
    df.to_csv("webcam_detection_report.csv", index=False)

    print("Webcam detection completed.")
    print("Report saved as webcam_detection_report.csv")


# -----------------------------------------
# 6. MAIN PROGRAM
# -----------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Object Detection Demonstrator"
    )

    parser.add_argument(
        "--source",
        default="0",
        help="Image path or 0 for webcam"
    )

    args = parser.parse_args()

    if args.source == "0":
        process_webcam()
    else:
        process_image(args.source)
