import cv2
import numpy as np
from ultralytics import YOLO


# Vehicle class IDs in COCO dataset
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Color per class
CLASS_COLORS = {
    "car":        (0, 200, 255),
    "motorcycle": (255, 100, 0),
    "bus":        (0, 255, 100),
    "truck":      (255, 0, 200),
}


class VehicleDetector:
    """
    Detects vehicles using YOLOv8 and assigns each detection
    to a lane zone.
    """

    def __init__(self, model_path="yolov8n.pt", confidence=0.40):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame, lane_zones: dict):
        """
        Run YOLO detection and assign vehicles to lanes.

        Returns:
            detections: list of dicts with keys:
                id, label, bbox (x1,y1,x2,y2), center, lane, confidence
            annotated_frame
        """
        results = self.model(frame, verbose=False)[0]
        detections = []
        annotated = frame.copy()

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue
            conf = float(box.conf[0])
            if conf < self.confidence:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = VEHICLE_CLASSES[cls_id]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Assign to lane
            lane = self._assign_lane(cx, cy, lane_zones)

            color = CLASS_COLORS.get(label, (200, 200, 200))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.circle(annotated, (cx, cy), 4, color, -1)
            cv2.putText(annotated, f"{label} {conf:.0%}",
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            detections.append({
                "label":      label,
                "bbox":       (x1, y1, x2, y2),
                "center":     (cx, cy),
                "lane":       lane,
                "confidence": conf,
            })

        return detections, annotated

    def _assign_lane(self, cx, cy, lane_zones):
        """Check which lane zone the vehicle center falls into."""
        for lane_name, zone in lane_zones.items():
            if (zone["x_min"] <= cx <= zone["x_max"] and
                    zone["y_min"] <= cy <= zone["y_max"]):
                return lane_name
        return "unknown"
