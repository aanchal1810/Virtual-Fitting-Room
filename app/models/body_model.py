import cv2
import mediapipe as mp
import numpy as np
mp_solutions = mp.solutions

class BodyMeasurementExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

    def _get_landmarks(self, image_path):
        """Extract pose landmarks using MediaPipe"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image not found or invalid path")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if not results.pose_landmarks:
            raise ValueError("No human detected in the image")

        landmarks = results.pose_landmarks.landmark
        h, w, _ = image.shape

        keypoints = {
            name: (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
            for idx, name in enumerate([
                'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer',
                'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
                'left_pinky', 'right_pinky', 'left_index', 'right_index', 'left_thumb', 'right_thumb',
                'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
                'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index'
            ])
        }

        return keypoints, (h, w)

    def _distance(self, p1, p2):
        """Euclidean distance between two points"""
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def extract_measurements(self, image_path, user_height_cm):
        keypoints, (img_h, img_w) = self._get_landmarks(image_path)

        # Head to ankle height in pixels
        pixel_height = self._distance(keypoints['nose'], keypoints['left_ankle'])
        cm_per_pixel = user_height_cm / pixel_height

        # Shoulder width
        shoulder_width_px = self._distance(keypoints['left_shoulder'], keypoints['right_shoulder'])
        shoulder_width_cm = shoulder_width_px * cm_per_pixel

        # Hip width
        hip_width_px = self._distance(keypoints['left_hip'], keypoints['right_hip'])
        hip_width_cm = hip_width_px * cm_per_pixel

        # Estimate bust & waist (approximation)
        bust_circumference = shoulder_width_cm * 1.3  # heuristic
        waist_circumference = hip_width_cm * 0.9      # heuristic
        hip_circumference = hip_width_cm * 1.2        # heuristic

        measurements = {
            "height_cm": round(user_height_cm, 2),
            "shoulder_width_cm": round(shoulder_width_cm, 2),
            "hip_width_cm": round(hip_width_cm, 2),
            "bust_circumference_cm": round(bust_circumference, 2),
            "waist_circumference_cm": round(waist_circumference, 2),
            "hip_circumference_cm": round(hip_circumference, 2),
        }

        return measurements