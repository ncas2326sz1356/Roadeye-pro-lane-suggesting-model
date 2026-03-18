import cv2
import numpy as np


class LaneDetector:
    """
    Detects lane lines from dashcam video and segments frame into
    2-lane (left/right) or 3-lane (left/center/right) zones.
    """

    def __init__(self):
        self.lane_count = 2
        self.lane_lines = []
        self.left_boundary = None
        self.right_boundary = None
        self.center_boundaries = []

    def preprocess(self, frame):
        """Convert to grayscale, blur, edge detect."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        return edges

    def region_of_interest(self, edges, frame_shape):
        """Focus on the lower 60% of the frame — the road area."""
        h, w = frame_shape[:2]
        mask = np.zeros_like(edges)
        # Trapezoid ROI
        polygon = np.array([[
            (int(w * 0.0), h),
            (int(w * 0.45), int(h * 0.40)),
            (int(w * 0.55), int(h * 0.40)),
            (int(w * 1.0), h),
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        return cv2.bitwise_and(edges, mask)

    def detect_lines(self, roi):
        """Hough transform to get raw line segments."""
        lines = cv2.HoughLinesP(
            roi,
            rho=1,
            theta=np.pi / 180,
            threshold=40,
            minLineLength=60,
            maxLineGap=150
        )
        return lines

    def classify_lines(self, lines, frame_width):
        """Classify lines into left, center, right based on slope & position."""
        left_lines = []
        right_lines = []
        center_lines = []

        if lines is None:
            return left_lines, center_lines, right_lines

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            x_mid = (x1 + x2) / 2

            # Filter near-horizontal lines
            if abs(slope) < 0.3:
                continue

            if slope < 0 and x_mid < frame_width * 0.5:
                left_lines.append(line[0])
            elif slope > 0 and x_mid > frame_width * 0.5:
                right_lines.append(line[0])
            else:
                # Near-vertical lines in center area = possible center divider
                if abs(slope) > 1.5:
                    center_lines.append(line[0])

        return left_lines, center_lines, right_lines

    def average_line(self, lines, frame_height):
        """Average a group of lines into one clean line."""
        if not lines:
            return None
        x_coords, y_coords = [], []
        for x1, y1, x2, y2 in lines:
            x_coords += [x1, x2]
            y_coords += [y1, y2]
        poly = np.polyfit(x_coords, y_coords, 1)
        poly_fn = np.poly1d(poly)

        y_bottom = frame_height
        y_top = int(frame_height * 0.40)
        x_bottom = int((y_bottom - poly[1]) / poly[0]) if poly[0] != 0 else x_coords[0]
        x_top = int((y_top - poly[1]) / poly[0]) if poly[0] != 0 else x_coords[0]
        return (x_bottom, y_bottom, x_top, y_top)

    def detect(self, frame):
        """
        Main detection method.
        Returns:
            annotated_frame, lane_zones dict, detected lane_count
        """
        h, w = frame.shape[:2]
        edges = self.preprocess(frame)
        roi = self.region_of_interest(edges, frame.shape)
        lines = self.detect_lines(roi)
        left_lines, center_lines, right_lines = self.classify_lines(lines, w)

        left_avg = self.average_line(left_lines, h)
        right_avg = self.average_line(right_lines, h)
        center_avg = self.average_line(center_lines, h)

        # Determine lane count based on detected lines
        detected_lane_count = 2
        lane_zones = {}
        annotated = frame.copy()

        if center_avg and left_avg and right_avg:
            detected_lane_count = 3
        elif center_avg and (left_avg or right_avg):
            detected_lane_count = 3
        else:
            detected_lane_count = 2

        self.lane_count = detected_lane_count

        # Draw lines & define zones
        if detected_lane_count == 2:
            # Single center divider logic
            if left_avg and right_avg:
                center_x_bottom = (left_avg[0] + right_avg[0]) // 2
                center_x_top = (left_avg[2] + right_avg[2]) // 2
                divider = (center_x_bottom, h, center_x_top, int(h * 0.40))
            elif left_avg:
                # Use left line as divider
                divider = left_avg
            elif right_avg:
                divider = right_avg
            else:
                # Fallback: center of frame
                divider = (w // 2, h, w // 2, int(h * 0.40))

            # Draw single white divider
            cv2.line(annotated,
                     (divider[0], divider[1]),
                     (divider[2], divider[3]),
                     (255, 255, 255), 3)

            # Zone definitions (x-range at mid height)
            mid_y = int(h * 0.65)
            div_x_at_mid = int(np.interp(mid_y,
                                          [divider[3], divider[1]],
                                          [divider[2], divider[0]]))

            lane_zones = {
                "left":  {"x_min": 0,         "x_max": div_x_at_mid, "y_min": int(h*0.40), "y_max": h},
                "right": {"x_min": div_x_at_mid, "x_max": w,         "y_min": int(h*0.40), "y_max": h},
            }
            # Draw zone labels
            cv2.putText(annotated, "LEFT", (div_x_at_mid // 2 - 30, int(h * 0.75)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(annotated, "RIGHT", (div_x_at_mid + (w - div_x_at_mid) // 2 - 30, int(h * 0.75)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        else:  # 3 lanes
            # Two dividers
            if left_avg and right_avg:
                div1 = left_avg
                div2 = right_avg
            elif left_avg and center_avg:
                div1 = left_avg
                div2 = center_avg
            elif right_avg and center_avg:
                div1 = center_avg
                div2 = right_avg
            else:
                div1 = (w // 3, h, w // 3, int(h * 0.40))
                div2 = (2 * w // 3, h, 2 * w // 3, int(h * 0.40))

            for div in [div1, div2]:
                cv2.line(annotated, (div[0], div[1]), (div[2], div[3]),
                         (255, 255, 255), 3)

            mid_y = int(h * 0.65)
            d1x = int(np.interp(mid_y, [div1[3], div1[1]], [div1[2], div1[0]]))
            d2x = int(np.interp(mid_y, [div2[3], div2[1]], [div2[2], div2[0]]))
            if d1x > d2x:
                d1x, d2x = d2x, d1x

            lane_zones = {
                "left":   {"x_min": 0,    "x_max": d1x, "y_min": int(h*0.40), "y_max": h},
                "center": {"x_min": d1x,  "x_max": d2x, "y_min": int(h*0.40), "y_max": h},
                "right":  {"x_min": d2x,  "x_max": w,   "y_min": int(h*0.40), "y_max": h},
            }
            mid_labels = {
                "left":   d1x // 2 - 20,
                "center": d1x + (d2x - d1x) // 2 - 35,
                "right":  d2x + (w - d2x) // 2 - 30,
            }
            for name, x in mid_labels.items():
                cv2.putText(annotated, name.upper(), (x, int(h * 0.75)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return annotated, lane_zones, detected_lane_count
