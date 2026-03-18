import cv2
import numpy as np


class SpeedEstimator:
    """
    Uses Farneback Optical Flow to compute per-vehicle motion vectors
    and estimate relative speed between ego vehicle and detected vehicles.
    """

    def __init__(self, fps=30, px_per_meter=20.0):
        self.prev_gray = None
        self.fps = fps
        self.px_per_meter = px_per_meter          # Calibration: pixels per metre at mid-frame
        self.flow = None
        self.vehicle_speeds = {}                  # label+lane → speed history
        self._smooth_window = 5
        self._speed_history = {}

    def update(self, frame):
        """Compute dense optical flow for the current frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            self.flow = np.zeros((*gray.shape, 2), dtype=np.float32)
            return

        self.flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        self.prev_gray = gray

    def get_vehicle_flow(self, bbox):
        """Average flow vector inside a bounding box."""
        if self.flow is None:
            return 0.0, 0.0
        x1, y1, x2, y2 = bbox
        h, w = self.flow.shape[:2]
        x1c = max(0, x1); y1c = max(0, y1)
        x2c = min(w, x2); y2c = min(h, y2)
        if x2c <= x1c or y2c <= y1c:
            return 0.0, 0.0
        roi = self.flow[y1c:y2c, x1c:x2c]
        mean_flow = np.mean(roi, axis=(0, 1))
        return float(mean_flow[0]), float(mean_flow[1])   # fx, fy

    def estimate_relative_speed(self, detections):
        """
        For each detection compute relative speed in km/h.
        Negative  → target moving away  (safe)
        Positive  → target approaching  (warning)
        Near zero → same speed

        Also computes the background (ego) flow to subtract.
        """
        if self.flow is None:
            for d in detections:
                d["rel_speed_kmh"] = 0.0
                d["flow_fy"] = 0.0
            return detections

        # Ego flow: median of entire frame (camera motion)
        ego_fy = float(np.median(self.flow[:, :, 1]))

        for d in detections:
            fx, fy = self.get_vehicle_flow(d["bbox"])

            # Relative vertical flow (positive fy in image = moving towards camera)
            rel_fy = fy - ego_fy

            # Convert pixel displacement/frame → m/s → km/h
            # rel_fy pixels/frame ÷ px_per_meter × fps = m/s
            rel_speed_ms = (rel_fy / self.px_per_meter) * self.fps
            rel_speed_kmh = rel_speed_ms * 3.6

            # Smooth using rolling window
            key = f"{d['label']}_{d['lane']}"
            if key not in self._speed_history:
                self._speed_history[key] = []
            self._speed_history[key].append(rel_speed_kmh)
            if len(self._speed_history[key]) > self._smooth_window:
                self._speed_history[key].pop(0)
            smooth_speed = float(np.mean(self._speed_history[key]))

            d["rel_speed_kmh"] = round(smooth_speed, 1)
            d["flow_fy"] = round(rel_fy, 2)

        return detections

    def draw_flow_overlay(self, frame, step=16):
        """Draw optical flow arrows on frame (debug view)."""
        if self.flow is None:
            return frame
        h, w = frame.shape[:2]
        overlay = frame.copy()
        for y in range(0, h, step):
            for x in range(0, w, step):
                fx, fy = self.flow[y, x]
                mag = np.sqrt(fx**2 + fy**2)
                if mag < 1.0:
                    continue
                end_x = int(x + fx * 2)
                end_y = int(y + fy * 2)
                cv2.arrowedLine(overlay, (x, y), (end_x, end_y),
                                (0, 255, 0), 1, tipLength=0.3)
        return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
