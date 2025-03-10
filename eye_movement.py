# # correct up, center, left, right
# import cv2
# import dlib
# import numpy as np

# # Load dlib’s face detector and 68 landmarks model
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("models\shape_predictor_68_face_landmarks.dat")

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# def detect_pupil(eye_region):
#     """
#     Detects the pupil inside the given eye region using thresholding and contours.
#     Returns the pupil center coordinates and bounding box.
#     """
#     gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
#     blurred_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)
#     _, threshold_eye = cv2.threshold(blurred_eye, 50, 255, cv2.THRESH_BINARY_INV)

#     # Find contours (pupil should be the largest dark region)
#     contours, _ = cv2.findContours(threshold_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     if contours:
#         pupil_contour = max(contours, key=cv2.contourArea)
#         px, py, pw, ph = cv2.boundingRect(pupil_contour)
#         pupil_x = px + pw // 2
#         pupil_y = py + ph // 2
#         return (pupil_x, pupil_y), (px, py, pw, ph)

#     return None, None

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces in the frame
#     faces = detector(gray)

#     for face in faces:
#         # Get facial landmarks
#         landmarks = predictor(gray, face)

#         # Get bounding box for left and right eyes
#         left_eye_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
#         right_eye_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])

#         left_eye_rect = cv2.boundingRect(left_eye_points)
#         right_eye_rect = cv2.boundingRect(right_eye_points)

#         # Crop the left and right eye regions
#         left_eye = frame[left_eye_rect[1]:left_eye_rect[1] + left_eye_rect[3], left_eye_rect[0]:left_eye_rect[0] + left_eye_rect[2]]
#         right_eye = frame[right_eye_rect[1]:right_eye_rect[1] + right_eye_rect[3], right_eye_rect[0]:right_eye_rect[0] + right_eye_rect[2]]

#         # Detect pupil positions
#         left_pupil, left_bbox = detect_pupil(left_eye)
#         right_pupil, right_bbox = detect_pupil(right_eye)

#         # Draw bounding boxes around eyes
#         cv2.rectangle(frame, (left_eye_rect[0], left_eye_rect[1]), 
#                       (left_eye_rect[0] + left_eye_rect[2], left_eye_rect[1] + left_eye_rect[3]), (0, 255, 0), 2)

#         cv2.rectangle(frame, (right_eye_rect[0], right_eye_rect[1]), 
#                       (right_eye_rect[0] + right_eye_rect[2], right_eye_rect[1] + right_eye_rect[3]), (0, 255, 0), 2)

#         # Draw pupil positions
#         if left_pupil and left_bbox:
#             px, py, pw, ph = left_bbox
#             cv2.circle(frame, (left_eye_rect[0] + left_pupil[0], left_eye_rect[1] + left_pupil[1]), 5, (0, 0, 255), -1)

#         if right_pupil and right_bbox:
#             px, py, pw, ph = right_bbox
#             cv2.circle(frame, (right_eye_rect[0] + right_pupil[0], right_eye_rect[1] + right_pupil[1]), 5, (0, 0, 255), -1)

#         # Gaze Direction Detection
#         if left_pupil and right_pupil:
#             lx, ly = left_pupil
#             rx, ry = right_pupil

#             eye_width = left_eye_rect[2]
#             eye_height = left_eye_rect[3]  # Get eye height

#             # Normalize pupil position using eye height
#             norm_ly = ly / eye_height
#             norm_ry = ry / eye_height

#             # Detect left & right gaze
#             if lx < eye_width // 3 and rx < eye_width // 3:
#                 gaze_direction = "Looking Left (Cheating)"
#             elif lx > 2 * eye_width // 3 and rx > 2 * eye_width // 3:
#                 gaze_direction = "Looking Right (Cheating)"

#             # Detect up & down gaze using relative position
#             elif norm_ly < 0.3 and norm_ry < 0.3:  # Upper third of the eye
#                 gaze_direction = "Looking Up (Cheating)"
#             elif norm_ly > 0.5 and norm_ry > 0.5:  # Lower third of the eye
#                 gaze_direction = "Looking Down (Cheating)"
#             else:
#                 gaze_direction = "Looking Center"

#             cv2.putText(frame, gaze_direction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     cv2.imshow("Gaze Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

#single fle working with ss
# import cv2
# import dlib
# import numpy as np
# import os
# import time

# # Load dlib’s face detector and 68 landmarks model
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# # Create log folder if it doesn't exist
# if not os.path.exists("log"):
#     os.makedirs("log")

# # Tracking gaze direction timing
# gaze_start_time = None
# previous_gaze = ""
# threshold_time = 3  # 3 seconds

# def detect_pupil(eye_region):
#     gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
#     blurred_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)
#     _, threshold_eye = cv2.threshold(blurred_eye, 50, 255, cv2.THRESH_BINARY_INV)
#     contours, _ = cv2.findContours(threshold_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     if contours:
#         pupil_contour = max(contours, key=cv2.contourArea)
#         px, py, pw, ph = cv2.boundingRect(pupil_contour)
#         return (px + pw // 2, py + ph // 2), (px, py, pw, ph)
#     return None, None

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = detector(gray)
    
#     for face in faces:
#         landmarks = predictor(gray, face)
        
#         left_eye_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
#         right_eye_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])
        
#         left_eye_rect = cv2.boundingRect(left_eye_points)
#         right_eye_rect = cv2.boundingRect(right_eye_points)
        
#         left_eye = frame[left_eye_rect[1]:left_eye_rect[1] + left_eye_rect[3], left_eye_rect[0]:left_eye_rect[0] + left_eye_rect[2]]
#         right_eye = frame[right_eye_rect[1]:right_eye_rect[1] + right_eye_rect[3], right_eye_rect[0]:right_eye_rect[0] + right_eye_rect[2]]
        
#         left_pupil, left_bbox = detect_pupil(left_eye)
#         right_pupil, right_bbox = detect_pupil(right_eye)
        
#         # Draw bounding boxes and pupils
#         cv2.rectangle(frame, (left_eye_rect[0], left_eye_rect[1]), 
#                       (left_eye_rect[0] + left_eye_rect[2], left_eye_rect[1] + left_eye_rect[3]), (0, 255, 0), 2)
#         cv2.rectangle(frame, (right_eye_rect[0], right_eye_rect[1]), 
#                       (right_eye_rect[0] + right_eye_rect[2], right_eye_rect[1] + right_eye_rect[3]), (0, 255, 0), 2)
        
#         if left_pupil and left_bbox:
#             cv2.circle(frame, (left_eye_rect[0] + left_pupil[0], left_eye_rect[1] + left_pupil[1]), 5, (0, 0, 255), -1)
#         if right_pupil and right_bbox:
#             cv2.circle(frame, (right_eye_rect[0] + right_pupil[0], right_eye_rect[1] + right_pupil[1]), 5, (0, 0, 255), -1)
        
#         # Gaze Detection
#         if left_pupil and right_pupil:
#             lx, ly = left_pupil
#             rx, ry = right_pupil
            
#             eye_width = left_eye_rect[2]
#             eye_height = left_eye_rect[3]
#             norm_ly, norm_ry = ly / eye_height, ry / eye_height
            
#             if lx < eye_width // 3 and rx < eye_width // 3:
#                 gaze_direction = "Looking Left"
#             elif lx > 2 * eye_width // 3 and rx > 2 * eye_width // 3:
#                 gaze_direction = "Looking Right"
#             elif norm_ly < 0.3 and norm_ry < 0.3:
#                 gaze_direction = "Looking Up"
#             elif norm_ly > 0.5 and norm_ry > 0.5:
#                 gaze_direction = "Looking Down"
#             else:
#                 gaze_direction = "Looking Center"
            
#             cv2.putText(frame, gaze_direction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
#             # Screenshot logic
#             if gaze_direction in ["Looking Left", "Looking Right", "Looking Up", "Looking Down"]:
#                 if previous_gaze == gaze_direction:
#                     if gaze_start_time and time.time() - gaze_start_time > threshold_time:
#                         timestamp = time.strftime("%Y%m%d-%H%M%S")
#                         cv2.imwrite(f"log/{gaze_direction}_{timestamp}.png", frame)
#                         print(f"Screenshot saved: {gaze_direction}_{timestamp}.png")
#                         gaze_start_time = None  # Reset timer after saving screenshot
#                 else:
#                     gaze_start_time = time.time()
#                     previous_gaze = gaze_direction
#             else:
#                 gaze_start_time = None
#                 previous_gaze = ""
    
#     cv2.imshow("Gaze Detection", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
    
# cap.release()
# cv2.destroyAllWindows()
import cv2
import dlib
import numpy as np

# Load dlib’s face detector and 68 landmarks model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

def detect_pupil(eye_region):
    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    blurred_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)
    _, threshold_eye = cv2.threshold(blurred_eye, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        pupil_contour = max(contours, key=cv2.contourArea)
        px, py, pw, ph = cv2.boundingRect(pupil_contour)
        return (px + pw // 2, py + ph // 2), (px, py, pw, ph)
    return None, None

def process_eye_movement(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    gaze_direction = "Looking Center"

    for face in faces:
        landmarks = predictor(gray, face)
        
        # Extract left and right eye landmarks
        left_eye_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
        right_eye_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])
        
        # Get bounding rectangles for the eyes
        left_eye_rect = cv2.boundingRect(left_eye_points)
        right_eye_rect = cv2.boundingRect(right_eye_points)
        
        # Extract eye regions
        left_eye = frame[left_eye_rect[1]:left_eye_rect[1] + left_eye_rect[3], left_eye_rect[0]:left_eye_rect[0] + left_eye_rect[2]]
        right_eye = frame[right_eye_rect[1]:right_eye_rect[1] + right_eye_rect[3], right_eye_rect[0]:right_eye_rect[0] + right_eye_rect[2]]
        
        # Detect pupils
        left_pupil, left_bbox = detect_pupil(left_eye)
        right_pupil, right_bbox = detect_pupil(right_eye)
        
        # Draw bounding boxes and pupils
        cv2.rectangle(frame, (left_eye_rect[0], left_eye_rect[1]), 
                      (left_eye_rect[0] + left_eye_rect[2], left_eye_rect[1] + left_eye_rect[3]), (0, 255, 0), 2)
        cv2.rectangle(frame, (right_eye_rect[0], right_eye_rect[1]), 
                      (right_eye_rect[0] + right_eye_rect[2], right_eye_rect[1] + right_eye_rect[3]), (0, 255, 0), 2)
        
        if left_pupil and left_bbox:
            cv2.circle(frame, (left_eye_rect[0] + left_pupil[0], left_eye_rect[1] + left_pupil[1]), 5, (0, 0, 255), -1)
        if right_pupil and right_bbox:
            cv2.circle(frame, (right_eye_rect[0] + right_pupil[0], right_eye_rect[1] + right_pupil[1]), 5, (0, 0, 255), -1)
        
        # Gaze Detection
        if left_pupil and right_pupil:
            lx, ly = left_pupil
            rx, ry = right_pupil
            
            eye_width = left_eye_rect[2]
            eye_height = left_eye_rect[3]
            norm_ly, norm_ry = ly / eye_height, ry / eye_height
            
            if lx < eye_width // 3 and rx < eye_width // 3:
                gaze_direction = "Looking Left"
            elif lx > 2 * eye_width // 3 and rx > 2 * eye_width // 3:
                gaze_direction = "Looking Right"
            elif norm_ly < 0.3 and norm_ry < 0.3:
                gaze_direction = "Looking Up"
            elif norm_ly > 0.5 and norm_ry > 0.5:
                gaze_direction = "Looking Down"
            else:
                gaze_direction = "Looking Center"
    
    return frame, gaze_direction