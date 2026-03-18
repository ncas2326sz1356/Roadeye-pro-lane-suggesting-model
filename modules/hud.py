import cv2
import numpy as np
from modules.decision_engine import Decision


# Lane status colors
LANE_SAFE    = (0, 200, 80)
LANE_BLOCKED = (0, 60, 220)
LANE_NONE    = (80, 80, 80)


def draw_hud(frame: np.ndarray,
             decision: Decision,
             lane_zones: dict,
             detections: list,
             lane_count: int,
             show_flow: bool = False) -> np.ndarray:
    """
    Draw the full Heads-Up Display overlay onto the frame.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # ── 1. Lane zone tint ────────────────────────────────────────────
    available = list(lane_zones.keys())
    ego_lane  = "center" if "center" in available else (
                "left" if len(available) == 1 else "left")

    for ln, zone in lane_zones.items():
        x1, x2 = zone["x_min"], zone["x_max"]
        y1, y2 = zone["y_min"], zone["y_max"]

        if ln == ego_lane:
            color = (30, 30, 30)
        elif (ln == "left"  and not decision.left_blocked) or \
             (ln == "right" and not decision.right_blocked) or \
             (ln == "center" and ln != ego_lane and True):
            color = LANE_SAFE
        else:
            color = LANE_BLOCKED

        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)
        cv2.fillPoly(overlay, [pts], color)

    frame = cv2.addWeighted(frame, 0.75, overlay, 0.25, 0)

    # ── 2. Lane status badges (top of screen) ────────────────────────
    badge_y   = 20
    badge_h   = 36
    badge_gap = 8

    lane_order = ["left", "center", "right"] if lane_count == 3 else ["left", "right"]
    total_badge_w = len([l for l in lane_order if l in available]) * (120 + badge_gap)
    start_x = (w - total_badge_w) // 2

    bx = start_x
    for ln in lane_order:
        if ln not in available:
            continue
        if ln == ego_lane:
            bg = (60, 60, 60)
            text = f"YOU ({ln.upper()})"
        elif ln == "left":
            bg = LANE_SAFE if not decision.left_blocked else LANE_BLOCKED
            status = "CLEAR" if not decision.left_blocked else "BLOCKED"
            text = f"LEFT: {status}"
        elif ln == "right":
            bg = LANE_SAFE if not decision.right_blocked else LANE_BLOCKED
            status = "CLEAR" if not decision.right_blocked else "BLOCKED"
            text = f"RIGHT: {status}"
        else:
            bg = (60, 60, 60)
            text = "CENTER"

        cv2.rectangle(frame, (bx, badge_y), (bx + 120, badge_y + badge_h), bg, -1)
        cv2.rectangle(frame, (bx, badge_y), (bx + 120, badge_y + badge_h), (255,255,255), 1)
        cv2.putText(frame, text, (bx + 6, badge_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        bx += 120 + badge_gap

    # ── 3. Per-vehicle speed tags ─────────────────────────────────────
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        spd = d.get("rel_speed_kmh", 0.0)

        if spd > 5:
            spd_color = (0, 0, 255)
            spd_label = f"▲{abs(spd):.0f}km/h"
        elif spd < -5:
            spd_color = (0, 200, 100)
            spd_label = f"▼{abs(spd):.0f}km/h"
        else:
            spd_color = (200, 200, 200)
            spd_label = f"~{abs(spd):.0f}km/h"

        cv2.putText(frame, spd_label,
                    (x1, y2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, spd_color, 2)

    # ── 4. Main decision banner (bottom) ─────────────────────────────
    banner_h  = 52
    banner_y  = h - banner_h - 6
    msg_color = decision.color

    cv2.rectangle(frame,
                  (10, banner_y),
                  (w - 10, banner_y + banner_h),
                  (20, 20, 20), -1)
    cv2.rectangle(frame,
                  (10, banner_y),
                  (w - 10, banner_y + banner_h),
                  msg_color, 2)

    # Word-wrap if message is long
    lines = _wrap_text(decision.message, max_chars=72)
    for i, line in enumerate(lines[:2]):
        cv2.putText(frame, line,
                    (20, banner_y + 20 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, msg_color, 2)

    # ── 5. Lane count indicator ───────────────────────────────────────
    cv2.putText(frame,
                f"Lanes detected: {lane_count}",
                (10, h - banner_h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    return frame


def _wrap_text(text: str, max_chars: int = 72):
    """Split long text into lines."""
    words = text.split()
    lines, current = [], ""
    for w in words:
        if len(current) + len(w) + 1 <= max_chars:
            current = (current + " " + w).strip()
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines
