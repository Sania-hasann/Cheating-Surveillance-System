# import cv2
# import torch
# from ultralytics import YOLO

# # Load trained YOLO model
# model = YOLO(r"C:\Users\LENOVO\Documents\FYP\Cheating Surveillance\models\best.pt")  
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)

# # Get class names
# class_names = model.names  
# print("Class Names:", class_names)

# # Find index for "mobile phone" (adjust if necessary)
# mobile_class_index = 0  # Change if "mobile phone" is not index 0

# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)

# confidence_threshold = 0.6  # Adjust for best results

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = model(frame, verbose=False)

#     for result in results:
#         for box in result.boxes:
#             conf = box.conf[0].item()
#             cls = int(box.cls[0].item())

#             if conf < confidence_threshold or cls != mobile_class_index:
#                 continue  # Skip low-confidence or non-mobile detections

#             x1, y1, x2, y2 = map(int, box.xyxy[0])  
#             label = f"Mobile ({conf:.2f})"

#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
#             cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#     cv2.imshow("Mobile Phone Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

#final working with ss
# import cv2
# import torch
# import time
# import os
# from ultralytics import YOLO

# # Load trained YOLO model
# model = YOLO(r"C:\Users\LENOVO\Documents\FYP\Cheating Surveillance\models\best.pt")  
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)

# # Create log folder if it doesn't exist
# log_folder = "log"
# os.makedirs(log_folder, exist_ok=True)

# # Get class names
# class_names = model.names  
# print("Class Names:", class_names)

# # Find index for "mobile phone" (adjust if necessary)
# mobile_class_index = 0  # Change if "mobile phone" is not index 0

# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)

# confidence_threshold = 0.6  # Adjust for best results

# # Timer for mobile phone detection
# mobile_detected_start = None
# detection_duration_threshold = 3  # Seconds

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = model(frame, verbose=False)
#     mobile_detected = False  # Track if mobile is detected

#     for result in results:
#         for box in result.boxes:
#             conf = box.conf[0].item()
#             cls = int(box.cls[0].item())

#             if conf < confidence_threshold or cls != mobile_class_index:
#                 continue  # Skip low-confidence or non-mobile detections

#             # Get bounding box coordinates
#             x1, y1, x2, y2 = map(int, box.xyxy[0])  
#             label = f"Mobile ({conf:.2f})"

#             # Draw bounding box and label
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
#             cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#             mobile_detected = True  # Mark detection as true

#     # Check if mobile is detected for more than 3 seconds
#     if mobile_detected:
#         if mobile_detected_start is None:
#             mobile_detected_start = time.time()  # Start timer
#         elif time.time() - mobile_detected_start >= detection_duration_threshold:
#             # Save screenshot
#             timestamp = time.strftime("%Y%m%d_%H%M%S")
#             screenshot_path = os.path.join(log_folder, f"mobile_detected_{timestamp}.jpg")
#             cv2.imwrite(screenshot_path, frame)
#             print(f"Screenshot saved: {screenshot_path}")
#             mobile_detected_start = None  # Reset timer after saving
#     else:
#         mobile_detected_start = None  # Reset if no mobile detected

#     cv2.imshow("Mobile Phone Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
import cv2
import torch
from ultralytics import YOLO

# Load trained YOLO model
model = YOLO(r"C:\Users\LENOVO\Documents\FYP\Cheating Surveillance\models\best.pt")  
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def process_mobile_detection(frame):
    results = model(frame, verbose=False)
    mobile_detected = False

    for result in results:
        for box in result.boxes:
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())

            if conf < 0.8 or cls != 0:  # Mobile class index is 0
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            label = f"Mobile ({conf:.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            mobile_detected = True
    
    return frame, mobile_detected