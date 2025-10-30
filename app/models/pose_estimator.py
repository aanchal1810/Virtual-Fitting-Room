import mediapipe as mp
import cv2
import numpy as np


class PoseEstimator:
    def __init__(self, debug=False):
        self.debug = debug
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, model_complexity=2)
        self.drawer = mp.solutions.drawing_utils

    def process_image(self, image_path):
        """Detects landmarks and computes joint angles."""
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "invalid image", "annotated_image": None}

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if not results.pose_landmarks:
            return {"error": "no person detected", "annotated_image": None}

        landmarks = self._extract_landmarks(results.pose_landmarks, image.shape)
        angles = self._compute_joint_angles(landmarks)

        annotated = image.copy()
        self.drawer.draw_landmarks(
            annotated,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            self.drawer.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            self.drawer.DrawingSpec(color=(0, 0, 255), thickness=2),
        )

        return {
            "annotated_image": annotated,
            "landmarks": landmarks,
            "angles": angles,
        }

    def _extract_landmarks(self, landmarks, shape):
        """Extract (x,y,z) coordinates scaled to image size."""
        h, w, _ = shape
        coords = {}
        for i, lm in enumerate(landmarks.landmark):
            coords[self.mp_pose.PoseLandmark(i).name] = {
                "x": lm.x * w,
                "y": lm.y * h,
                "z": lm.z * w,  # scaled for consistency
                "visibility": lm.visibility,
            }
        return coords

    def _compute_joint_angles(self, lm):
        """Compute basic limb and torso angles."""
        def get_point(name):
            return np.array([lm[name]["x"], lm[name]["y"], lm[name]["z"]])

        def angle(a, b, c):
            ba, bc = a - b, c - b
            cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

        angles = {}

        try:
            angles["left_elbow"] = angle(
                get_point("LEFT_SHOULDER"), get_point("LEFT_ELBOW"), get_point("LEFT_WRIST")
            )
            angles["right_elbow"] = angle(
                get_point("RIGHT_SHOULDER"), get_point("RIGHT_ELBOW"), get_point("RIGHT_WRIST")
            )
            angles["left_knee"] = angle(
                get_point("LEFT_HIP"), get_point("LEFT_KNEE"), get_point("LEFT_ANKLE")
            )
            angles["right_knee"] = angle(
                get_point("RIGHT_HIP"), get_point("RIGHT_KNEE"), get_point("RIGHT_ANKLE")
            )
            angles["left_shoulder"] = angle(
                get_point("LEFT_ELBOW"), get_point("LEFT_SHOULDER"), get_point("LEFT_HIP")
            )
            angles["right_shoulder"] = angle(
                get_point("RIGHT_ELBOW"), get_point("RIGHT_SHOULDER"), get_point("RIGHT_HIP")
            )

            # torso tilt (between shoulders–hips line and vertical)
            shoulders = (get_point("LEFT_SHOULDER") + get_point("RIGHT_SHOULDER")) / 2
            hips = (get_point("LEFT_HIP") + get_point("RIGHT_HIP")) / 2
            torso_vector = hips - shoulders
            vertical = np.array([0, 1, 0])
            tilt_angle = np.degrees(
                np.arccos(np.dot(torso_vector, vertical) / np.linalg.norm(torso_vector))
            )
            angles["torso_tilt"] = float(tilt_angle)
        except KeyError:
            pass

        return angles
