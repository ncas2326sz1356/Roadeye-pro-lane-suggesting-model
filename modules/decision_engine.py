from dataclasses import dataclass, field
from typing import List, Dict, Optional


# Thresholds
APPROACH_SPEED_WARN   = 5.0    # km/h relative — vehicle ahead is approaching
APPROACH_SPEED_DANGER = 15.0   # km/h — urgent slow down
SAFE_LANE_THRESHOLD   = 3.0    # max relative approach speed to consider lane safe
FRONT_ZONE_RATIO      = 0.35   # top 35% of frame height = "ahead" zone


@dataclass
class Decision:
    slow_down:        bool = False
    slow_down_urgent: bool = False
    move_left:        bool = False
    move_right:       bool = False
    left_blocked:     bool = False
    right_blocked:    bool = False
    center_blocked:   bool = False
    message:          str  = ""
    color:            tuple = (0, 255, 0)   # Green = safe
    vehicles_ahead:   List  = field(default_factory=list)


class DecisionEngine:
    """
    Analyses per-lane vehicle data and relative speeds to produce
    actionable driving suggestions.
    """

    def __init__(self, frame_height: int = 480):
        self.frame_height = frame_height

    def decide(self,
               detections: list,
               lane_zones: dict,
               ego_lane: str = "center") -> Decision:
        """
        Parameters
        ----------
        detections  : list of vehicle dicts with lane & rel_speed_kmh
        lane_zones  : dict of active lane zones
        ego_lane    : which lane the ego vehicle is in (always 'center'
                      for a dashcam mounted centrally, else 'left'/'right')

        Returns
        -------
        Decision dataclass
        """
        dec = Decision()
        available_lanes = list(lane_zones.keys())

        # ── Categorise vehicles by lane ──────────────────────────────
        by_lane: Dict[str, list] = {ln: [] for ln in available_lanes}
        for d in detections:
            if d["lane"] in by_lane:
                by_lane[d["lane"]].append(d)

        # ── Check ego lane: vehicles directly ahead ───────────────────
        ego_vehicles = by_lane.get(ego_lane, [])
        ahead_vehicles = self._filter_ahead(ego_vehicles)

        if ahead_vehicles:
            worst = max(ahead_vehicles, key=lambda v: v["rel_speed_kmh"])
            spd = worst["rel_speed_kmh"]

            if spd >= APPROACH_SPEED_DANGER:
                dec.slow_down = True
                dec.slow_down_urgent = True
                dec.vehicles_ahead = ahead_vehicles
            elif spd >= APPROACH_SPEED_WARN:
                dec.slow_down = True
                dec.vehicles_ahead = ahead_vehicles

        # ── Check adjacent lanes ──────────────────────────────────────
        left_lane  = self._left_of(ego_lane, available_lanes)
        right_lane = self._right_of(ego_lane, available_lanes)

        left_safe  = self._is_lane_safe(by_lane, left_lane)
        right_safe = self._is_lane_safe(by_lane, right_lane)

        dec.left_blocked  = not left_safe  if left_lane  else True
        dec.right_blocked = not right_safe if right_lane else True
        dec.move_left     = left_safe      if left_lane  else False
        dec.move_right    = right_safe     if right_lane else False

        # ── Compose message & color ───────────────────────────────────
        dec.message, dec.color = self._compose(dec, left_lane, right_lane)
        return dec

    # ── Helpers ───────────────────────────────────────────────────────

    def _filter_ahead(self, vehicles):
        """Keep only vehicles in upper portion of frame (ahead of ego)."""
        ahead = []
        for v in vehicles:
            _, y1, _, y2 = v["bbox"]
            cy = (y1 + y2) / 2
            if cy < self.frame_height * 0.72:   # upper 72% = ahead
                ahead.append(v)
        return ahead

    def _is_lane_safe(self, by_lane, lane_name):
        if lane_name is None:
            return False
        vehicles = by_lane.get(lane_name, [])
        if not vehicles:
            return True
        for v in vehicles:
            if v["rel_speed_kmh"] >= SAFE_LANE_THRESHOLD:
                return False
        return True

    def _left_of(self, ego, available):
        order = ["left", "center", "right"]
        # Also handle 2-lane (left/right)
        if "center" not in available:
            order = ["left", "right"]
        try:
            idx = order.index(ego)
            return order[idx - 1] if idx > 0 else None
        except ValueError:
            return None

    def _right_of(self, ego, available):
        order = ["left", "center", "right"]
        if "center" not in available:
            order = ["left", "right"]
        try:
            idx = order.index(ego)
            return order[idx + 1] if idx < len(order) - 1 else None
        except ValueError:
            return None

    def _compose(self, dec: Decision, left_lane, right_lane):
        messages = []
        color = (0, 220, 0)   # green default

        if dec.slow_down_urgent:
            messages.append("⛔ BRAKE NOW — Vehicle closing fast!")
            color = (0, 0, 255)
        elif dec.slow_down:
            messages.append("⚠️  Vehicle ahead slowing — Reduce speed")
            color = (0, 165, 255)

        if dec.move_left and dec.move_right:
            messages.append("✅ Both lanes clear — you may change")
            color = (0, 220, 0) if not dec.slow_down else color
        elif dec.move_left:
            messages.append("⬅️  Safe to move LEFT")
            color = (0, 220, 0) if not dec.slow_down else color
        elif dec.move_right:
            messages.append("➡️  Safe to move RIGHT")
            color = (0, 220, 0) if not dec.slow_down else color
        else:
            if left_lane or right_lane:
                messages.append("🚫 Adjacent lanes blocked — Stay in lane")
                if not dec.slow_down:
                    color = (0, 165, 255)

        if not messages:
            messages.append("✅ Road clear — Drive safely")

        return "  |  ".join(messages), color
