# body_measurement_extractor_v3.py
import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Dict

mp_pose = mp.solutions.pose
mp_selfie = mp.solutions.selfie_segmentation

# default depth ratios (front width -> depth) if no side image is available
DEFAULT_DEPTH_RATIOS = {"bust": 0.55, "waist": 0.47, "hip": 0.62}

class BodyMeasurementExtractor:
    def __init__(self,
                 pose_complexity: int = 1,
                 min_detection_confidence: float = 0.6,
                 seg_model_selection: int = 1,
                 vertical_window_px: int = 11):
        """
        vertical_window_px: number of rows (odd) to average width around the target row (e.g., 11 => +/-5 rows)
        """
        self.pose = mp_pose.Pose(static_image_mode=True,
                                 model_complexity=pose_complexity,
                                 min_detection_confidence=min_detection_confidence)
        self.segmenter = mp_selfie.SelfieSegmentation(model_selection=seg_model_selection)
        self.window = vertical_window_px if vertical_window_px % 2 == 1 else vertical_window_px + 1

    @staticmethod
    def _to_px(lm, w, h):
        return int(lm.x * w), int(lm.y * h)

    @staticmethod
    def _euclidean(a, b):
        return float(np.linalg.norm(np.array(a) - np.array(b)))

    @staticmethod
    def _mask_row_widths(mask: np.ndarray, row_center: int, half_window: int):
        """Return list of widths (px) for each row in window; rows with no foreground return 0."""
        h, _ = mask.shape
        widths = []
        for r in range(row_center - half_window, row_center + half_window + 1):
            if r < 0 or r >= h:
                widths.append(0)
                continue
            row = mask[r, :]
            fg = np.where(row > 0)[0]
            if fg.size == 0:
                widths.append(0)
            else:
                widths.append(int(fg[-1] - fg[0] + 1))
        return widths

    def _get_pose_and_mask(self, img_rgb):
        res_pose = self.pose.process(img_rgb)
        res_mask = self.segmenter.process(img_rgb)
        mask = None
        if res_mask and res_mask.segmentation_mask is not None:
            seg = res_mask.segmentation_mask
            mask = (seg > 0.5).astype(np.uint8) * 255
            # morphological close to fill small holes
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return res_pose, mask

    def extract_measurements(self,
                             front_image_path: str,
                             user_height_cm: Optional[float] = None,
                             side_image_path: Optional[str] = None,
                             known_measurement: Optional[Dict[str, float]] = None,
                             debug_out_path: Optional[str] = None
                             ) -> Dict:
        """
        known_measurement: optional dict e.g. {"waist_cm": 74} to calibrate systematic bias.
        Returns a dictionary of measurements and diagnostics.
        """
        img = cv2.imread(front_image_path)
        if img is None:
            raise ValueError("Front image not found or unreadable")

        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res_pose, mask = self._get_pose_and_mask(img_rgb)

        if not res_pose or not res_pose.pose_landmarks:
            raise ValueError("No human detected in the front image")

        lm = res_pose.pose_landmarks.landmark

        # Map indices
        idx = mp_pose.PoseLandmark
        def px(n): return self._to_px(lm[n.value], w, h)

        # Get multiple candidate head landmarks and compute head_top estimate
        face_points = []
        for p in [idx.NOSE, idx.LEFT_EYE, idx.RIGHT_EYE, idx.LEFT_EAR, idx.RIGHT_EAR]:
            l = lm[p.value]
            if getattr(l, "visibility", 0.0) > 0.05:
                face_points.append(self._to_px(l, w, h))
        if len(face_points) == 0:
            # fallback: use nose
            head_top_y = self._to_px(lm[idx.NOSE.value], w, h)[1] - int(0.06 * h)
        else:
            min_y = min([pt[1] for pt in face_points])
            head_top_y = max(0, int(min_y - 0.06 * h))  # subtract 6% of image height as margin

        # foot/ankle: choose lowest visible among ankles/heels/foot_index
        foot_candidates = []
        for p in [idx.LEFT_ANKLE, idx.RIGHT_ANKLE, idx.LEFT_HEEL, idx.RIGHT_HEEL, idx.LEFT_FOOT_INDEX, idx.RIGHT_FOOT_INDEX]:
            l = lm[p.value]
            if getattr(l, "visibility", 0.0) > 0.03:
                foot_candidates.append(self._to_px(l, w, h))
        if len(foot_candidates) == 0:
            # fallback: use hip y + some factor
            foot_y = int(h * 0.95)
        else:
            foot_y = max([pt[1] for pt in foot_candidates])

        pixel_height = max(10, abs(foot_y - head_top_y))
        if user_height_cm:
            cm_per_px = float(user_height_cm) / float(pixel_height)
            estimated_height_cm = float(user_height_cm)
            height_source = "user_input"
        else:
            # fallback assumption (can be changed)
            estimated_height_cm = 167.0
            cm_per_px = estimated_height_cm / float(pixel_height)
            height_source = "assumed_167cm"

        # Shoulders & hips midpoints
        left_sh = px(idx.LEFT_SHOULDER)
        right_sh = px(idx.RIGHT_SHOULDER)
        shoulder_mid = ((left_sh[0] + right_sh[0]) // 2, (left_sh[1] + right_sh[1]) // 2)

        left_hip = px(idx.LEFT_HIP)
        right_hip = px(idx.RIGHT_HIP)
        hip_mid = ((left_hip[0] + right_hip[0]) // 2, (left_hip[1] + right_hip[1]) // 2)

        # define probe rows relative positions (robust multipliers)
        shoulder_y = shoulder_mid[1]
        bust_y = int(shoulder_y + 0.12 * (hip_mid[1] - shoulder_y))
        waist_y = int(shoulder_y + 0.45 * (hip_mid[1] - shoulder_y))
        hip_y = hip_mid[1]

        half_win = (self.window - 1) // 2

        # measure widths from mask using vertical window median
        def measure_widths_at(y):
            if mask is not None:
                widths = self._mask_row_widths(mask, y, half_win)
                # take median of non-zero widths; if all zeros, return 0
                nonzero = [x for x in widths if x > 0]
                if len(nonzero) > 0:
                    return int(np.median(nonzero)), widths
                # try to expand window slightly if nothing found
                # (try a bigger window up to 3x)
                for extra in (half_win*2, half_win*4):
                    widths2 = self._mask_row_widths(mask, y, min(extra, 31))
                    nonzero2 = [x for x in widths2 if x > 0]
                    if len(nonzero2) > 0:
                        return int(np.median(nonzero2)), widths2
                return 0, widths
            else:
                # fallback: use horizontal distance between leftmost and rightmost among shoulders/hips
                xs = [left_sh[0], right_sh[0], left_hip[0], right_hip[0]]
                return int(max(xs) - min(xs)), []

        bust_w_px, _ = measure_widths_at(bust_y)
        waist_w_px, _ = measure_widths_at(waist_y)
        hip_w_px, _ = measure_widths_at(hip_y)

        # convert to cm
        bust_w_cm = bust_w_px * cm_per_px
        waist_w_cm = waist_w_px * cm_per_px
        hip_w_cm = hip_w_px * cm_per_px

        # measure shoulder width via landmark distances
        shoulder_w_px = self._euclidean(left_sh, right_sh)
        shoulder_w_cm = shoulder_w_px * cm_per_px

        # depth: if side image provided, try to get direct depth measures; otherwise use ratios
        depth_available = False
        bust_depth_cm = waist_depth_cm = hip_depth_cm = None
        if side_image_path:
            side_img = cv2.imread(side_image_path)
            if side_img is not None:
                sh, sw, _ = side_img.shape
                side_rgb = cv2.cvtColor(side_img, cv2.COLOR_BGR2RGB)
                _, side_mask = self._get_pose_and_mask(side_rgb)
                if side_mask is not None:
                    # map front y to side image y by proportion
                    bust_row_s = int(bust_y / h * sh)
                    waist_row_s = int(waist_y / h * sh)
                    hip_row_s = int(hip_y / h * sh)
                    b_px, _ = self._mask_row_widths(side_mask, bust_row_s, half_win)
                    wa_px, _ = self._mask_row_widths(side_mask, waist_row_s, half_win)
                    hi_px, _ = self._mask_row_widths(side_mask, hip_row_s, half_win)
                    if b_px > 0: bust_depth_cm = b_px * (estimated_height_cm / float(sh))
                    if wa_px > 0: waist_depth_cm = wa_px * (estimated_height_cm / float(sh))
                    if hi_px > 0: hip_depth_cm = hi_px * (estimated_height_cm / float(sh))
                    depth_available = True

        if not depth_available:
            bust_depth_cm = bust_w_cm * DEFAULT_DEPTH_RATIOS["bust"]
            waist_depth_cm = waist_w_cm * DEFAULT_DEPTH_RATIOS["waist"]
            hip_depth_cm = hip_w_cm * DEFAULT_DEPTH_RATIOS["hip"]

        # compute circumferences using ellipse approx
        def ellipse_circ(w_cm, d_cm):
            a = w_cm / 2.0
            b = d_cm / 2.0
            return 2.0 * np.pi * np.sqrt((a*a + b*b) / 2.0)

        bust_circ = ellipse_circ(bust_w_cm, bust_depth_cm) if bust_w_cm > 0 else 0.0
        waist_circ = ellipse_circ(waist_w_cm, waist_depth_cm) if waist_w_cm > 0 else 0.0
        hip_circ = ellipse_circ(hip_w_cm, hip_depth_cm) if hip_w_cm > 0 else 0.0

        # If user provides a known_measurement (e.g., {"waist_cm": 74}), compute a global correction factor
        correction_factor = 1.0
        if known_measurement:
            # pick the first provided known measurement key among bust/waist/hip
            for k in ("waist_cm", "bust_cm", "hip_cm"):
                if k in known_measurement:
                    provided = float(known_measurement[k])
                    predicted = {"waist_cm": waist_circ, "bust_cm": bust_circ, "hip_cm": hip_circ}[k]
                    if predicted > 0:
                        correction_factor = provided / predicted
                    break

        # apply correction
        bust_circ_corr = bust_circ * correction_factor
        waist_circ_corr = waist_circ * correction_factor
        hip_circ_corr = hip_circ * correction_factor
        shoulder_w_cm_corr = shoulder_w_cm * correction_factor

        results = {
            "height_cm": round(estimated_height_cm, 2),
            "height_source": height_source,
            "pixel_height": int(pixel_height),
            "cm_per_pixel": float(cm_per_px),
            "shoulder_width_cm_raw": round(shoulder_w_cm, 2),
            "shoulder_width_cm": round(shoulder_w_cm_corr, 2),
            "bust_width_cm": round(bust_w_cm, 2),
            "waist_width_cm": round(waist_w_cm, 2),
            "hip_width_cm": round(hip_w_cm, 2),
            "bust_circumference_cm_raw": round(bust_circ, 2),
            "waist_circumference_cm_raw": round(waist_circ, 2),
            "hip_circumference_cm_raw": round(hip_circ, 2),
            "bust_circumference_cm": round(bust_circ_corr, 2),
            "waist_circumference_cm": round(waist_circ_corr, 2),
            "hip_circumference_cm": round(hip_circ_corr, 2),
            "depth_estimated": not depth_available,
            "depths_cm": {
                "bust_depth_cm": round(bust_depth_cm, 2),
                "waist_depth_cm": round(waist_depth_cm, 2),
                "hip_depth_cm": round(hip_depth_cm, 2)
            },
            "correction_factor": round(correction_factor, 4),
            "landmark_visibility": {p.name: float(getattr(lm[p.value], "visibility", 0.0)) for p in mp_pose.PoseLandmark}
        }

        # debug visualization if requested
        if debug_out_path:
            vis = img.copy()
            if mask is not None:
                cmask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                vis = cv2.addWeighted(vis, 0.78, cmask, 0.22, 0)
            # lines
            for y, label in [(bust_y, "bust"), (waist_y, "waist"), (hip_y, "hip")]:
                cv2.line(vis, (0, y), (w-1, y), (0, 255, 0), 1)
                cv2.putText(vis, label, (10, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            # landmarks for reference
            for k in ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"]:
                p = mp_pose.PoseLandmark[k]
                x, y = self._to_px(lm[p.value], w, h)
                cv2.circle(vis, (x, y), 4, (255, 128, 0), -1)
            # annotate numeric
            cv2.putText(vis, f"height_src:{height_source} h_cm:{results['height_cm']}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(vis, f"waist_cm:{results['waist_circumference_cm']}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.imwrite(debug_out_path, vis)
            

        return results

# quick usage example (run as script)
# if __name__ == "__main__":
#     extractor = BodyMeasurementExtractorV3()
#     front = "sample.jpeg"
#     # optionally ask user to input one true measurement to calibrate
#     known = {"waist_cm": 74}  # replace with real waist if available, else None
#     res = extractor.extract_measurements(front, user_height_cm=165.0, known_measurement=known, debug_out_path="debug_v3.jpg")
#     for k,v in res.items():
#         print(k, ":", v)
#     print("Debug image written to debug_v3.jpg")
