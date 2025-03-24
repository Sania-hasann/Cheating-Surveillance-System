import cv2
import dlib
import numpy as np
import math
from collections import deque
import time

# Load face detector & landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")

# 3D Model Points (Mapped to Facial Landmarks)
model_points = np.array([
    (0.0, 0.0, 0.0),        # Nose tip
    (0.0, -50.0, -10.0),    # Chin
    (-30.0, 40.0, -10.0),   # Left eye
    (30.0, 40.0, -10.0),    # Right eye
    (-25.0, -30.0, -10.0),  # Left mouth corner
    (25.0, -30.0, -10.0)    # Right mouth corner
], dtype=np.float64)

# Camera Calibration (Assuming 640x480)
focal_length = 640
center = (320, 240)
camera_matrix = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]
], dtype=np.float64)

dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

# Define thresholds for "Looking at Screen"
CALIBRATION_TIME = 5  # Time to set neutral position

# Smoothing filter for stable head pose estimation
ANGLE_HISTORY_SIZE = 10
yaw_history = deque(maxlen=ANGLE_HISTORY_SIZE)
pitch_history = deque(maxlen=ANGLE_HISTORY_SIZE)
roll_history = deque(maxlen=ANGLE_HISTORY_SIZE)

# Global variables for state management
previous_state = "Looking at Screen"
calibrated_angles = None

def get_head_pose_angles(image_points):
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    sy = math.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        pitch = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = 0

    return np.degrees(pitch), np.degrees(yaw), np.degrees(roll)

def smooth_angle(angle_history, new_angle):
    angle_history.append(new_angle)
    return np.mean(angle_history)

def process_head_pose(frame, calibrated_angles=None):
    global previous_state

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    head_direction = "Looking at Screen"

    for face in faces:
        landmarks = predictor(gray, face)
        image_points = np.array([
            (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
            (landmarks.part(8).x, landmarks.part(8).y),    # Chin
            (landmarks.part(36).x, landmarks.part(36).y),  # Left eye
            (landmarks.part(45).x, landmarks.part(45).y),  # Right eye
            (landmarks.part(48).x, landmarks.part(48).y),  # Left mouth corner
            (landmarks.part(54).x, landmarks.part(54).y)   # Right mouth corner
        ], dtype=np.float64)

        angles = get_head_pose_angles(image_points)
        if angles is None:
            continue

        pitch = smooth_angle(pitch_history, angles[0])
        yaw = smooth_angle(yaw_history, angles[1])
        roll = smooth_angle(roll_history, angles[2])

        # *Fix: Ensure calibrated_angles is valid before unpacking*
        if calibrated_angles is None:
            return frame, (pitch, yaw, roll)  # Store calibrated angles

        if not isinstance(calibrated_angles, (tuple, list)) or len(calibrated_angles) != 3:
            print("⚠ Error: calibrated_angles is invalid:", calibrated_angles)
            return frame, "Calibration Error"

        pitch_offset, yaw_offset, roll_offset = calibrated_angles  # ✅ Now safe to unpack

        # Define head movement thresholds
        PITCH_THRESHOLD = 8
        YAW_THRESHOLD = 12
        ROLL_THRESHOLD = 5

        # Determine head direction
        if abs(yaw - yaw_offset) <= YAW_THRESHOLD and abs(pitch - pitch_offset) <= PITCH_THRESHOLD and abs(roll - roll_offset) <= ROLL_THRESHOLD:
            current_state = "Looking at Screen"
        elif yaw < yaw_offset - 15:
            current_state = "Looking Left"
        elif yaw > yaw_offset + 15:
            current_state = "Looking Right"
        elif pitch > pitch_offset + 10:
            current_state = "Looking Up"
        elif pitch < pitch_offset - 10:
            current_state = "Looking Down"
        elif abs(roll - roll_offset) > 7:
            current_state = "Tilted"
        else:
            current_state = previous_state

        previous_state = current_state
        head_direction = current_state

    return frame, head_direction