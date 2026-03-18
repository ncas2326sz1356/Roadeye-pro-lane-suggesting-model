import cv2
import numpy as np

from modules.lane_detector   import LaneDetector
from modules.vehicle_detector import VehicleDetector
from modules.speed_estimator  import SpeedEstimator
from modules.decision_engine  import DecisionEngine
from modules.hud              import draw_hud


class RoadEyePipeline:
    """
    Master pipeline: ties all modules together.
    Call process_frame() on each video frame.
    """

    def __init__(self,
                 yolo_model:    str   = "yolov8n.pt",
                 confidence:    float = 0.40,
                 fps:           int   = 30,
                 px_per_meter:  float = 20.0,
                 ego_lane:      str   = "center",
                 show_flow:     bool  = False):

        self.lane_detector   = LaneDetector()
        self.vehicle_detector = VehicleDetector(yolo_model, confidence)
        self.speed_estimator  = SpeedEstimator(fps, px_per_meter)
        self.decision_engine  = DecisionEngine()
        self.ego_lane         = ego_lane
        self.show_flow        = show_flow

        # Stats
        self.frame_count     = 0
        self.lane_count      = 2
        self.last_decision   = None
        self.last_detections = []

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Full pipeline for one frame.
        Returns the annotated output frame.
        """
        self.frame_count += 1

        # ── 1. Lane detection ─────────────────────────────────────────
        lane_frame, lane_zones, lane_count = self.lane_detector.detect(frame)
        self.lane_count = lane_count

        # ── 2. Vehicle detection ──────────────────────────────────────
        detections, det_frame = self.vehicle_detector.detect(lane_frame, lane_zones)

        # ── 3. Optical Flow + relative speed ─────────────────────────
        self.speed_estimator.update(frame)
        self.speed_estimator.frame_height = frame.shape[0]
        detections = self.speed_estimator.estimate_relative_speed(detections)

        if self.show_flow:
            det_frame = self.speed_estimator.draw_flow_overlay(det_frame)

        # ── 4. Decision ───────────────────────────────────────────────
        # Choose ego lane dynamically
        available = list(lane_zones.keys())
        if "center" in available:
            ego = "center"
        elif len(available) == 2:
            ego = "left"          # dashcam: ego is left lane of 2
        else:
            ego = available[0] if available else "left"

        self.decision_engine.frame_height = frame.shape[0]
        decision = self.decision_engine.decide(detections, lane_zones, ego)

        self.last_decision   = decision
        self.last_detections = detections

        # ── 5. HUD overlay ────────────────────────────────────────────
        output = draw_hud(det_frame, decision, lane_zones,
                          detections, lane_count, self.show_flow)

        return output

    def get_stats(self) -> dict:
        """Return current frame statistics for the Streamlit sidebar."""
        stats = {
            "frame":      self.frame_count,
            "lane_count": self.lane_count,
            "vehicles":   len(self.last_detections),
            "slow_down":  False,
            "urgent":     False,
            "move_left":  False,
            "move_right": False,
            "message":    "Initialising...",
            "by_lane":    {},
        }
        if self.last_decision:
            d = self.last_decision
            stats.update({
                "slow_down":  d.slow_down,
                "urgent":     d.slow_down_urgent,
                "move_left":  d.move_left,
                "move_right": d.move_right,
                "message":    d.message,
            })
        for det in self.last_detections:
            lane = det.get("lane", "unknown")
            if lane not in stats["by_lane"]:
                stats["by_lane"][lane] = []
            stats["by_lane"][lane].append({
                "label": det["label"],
                "speed": det.get("rel_speed_kmh", 0.0),
            })
        return stats
