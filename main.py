import cv2
import time
import os
from eye_movement import process_eye_movement
from head_pose import process_head_pose
from mobile_detection import process_mobile_detection

# Initialize webcam
cap = cv2.VideoCapture(0)

# Create a log directory for screenshots
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)

# Calibration for head pose
calibrated_angles = None
start_time = time.time()

# Timers for each functionality
head_misalignment_start_time = None
eye_misalignment_start_time = None
mobile_detection_start_time = None

# Previous states
previous_head_state = "Looking at Screen"
previous_eye_state = "Looking at Screen"
previous_mobile_state = False

# Initialize head_direction with a default value
head_direction = "Looking at Screen"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process eye movement
    frame, gaze_direction = process_eye_movement(frame)
    cv2.putText(frame, f"Gaze Direction: {gaze_direction}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Process head pose
    if time.time() - start_time <= 5:  # Calibration time
        cv2.putText(frame, "Calibrating... Keep your head straight", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if calibrated_angles is None:
            _, calibrated_angles = process_head_pose(frame, None)
    else:
        frame, head_direction = process_head_pose(frame, calibrated_angles)
        cv2.putText(frame, f"Head Direction: {head_direction}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Process mobile detection
    frame, mobile_detected = process_mobile_detection(frame)
    cv2.putText(frame, f"Mobile Detected: {mobile_detected}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Check for head misalignment
    if head_direction != "Looking at Screen":
        if head_misalignment_start_time is None:
            head_misalignment_start_time = time.time()
        elif time.time() - head_misalignment_start_time >= 3:
            filename = os.path.join(log_dir, f"head_{head_direction}_{int(time.time())}.png")
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
            head_misalignment_start_time = None  # Reset timer
    else:
        head_misalignment_start_time = None  # Reset timer

    # Check for eye misalignment
    if gaze_direction != "Looking at Screen":
        if eye_misalignment_start_time is None:
            eye_misalignment_start_time = time.time()
        elif time.time() - eye_misalignment_start_time >= 3:
            filename = os.path.join(log_dir, f"eye_{gaze_direction}_{int(time.time())}.png")
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
            eye_misalignment_start_time = None  # Reset timer
    else:
        eye_misalignment_start_time = None  # Reset timer

    # Check for mobile detection
    if mobile_detected:
        if mobile_detection_start_time is None:
            mobile_detection_start_time = time.time()
        elif time.time() - mobile_detection_start_time >= 3:
            filename = os.path.join(log_dir, f"mobile_detected_{int(time.time())}.png")
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
            mobile_detection_start_time = None  # Reset timer
    else:
        mobile_detection_start_time = None  # Reset timer

    # Display the combined output
    cv2.imshow("Combined Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()