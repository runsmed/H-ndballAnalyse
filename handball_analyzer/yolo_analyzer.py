"""Lokal, gratis håndballanalyse med YOLO (objektgjenkjenning) - ingen API-kall.

YOLO ser kun objekter (spillere, ball) og posisjonene deres - den forstår
ikke regler eller semantikk. Hendelsene under er derfor HEURISTIKKER avledet
fra ball-/spillerbevegelse (f.eks. "ballen beveger seg raskt mot en
målsone"), ikke bekreftede håndballhendelser. Bruk denne analysen til å
finne interessante tidsseksjoner gratis, og send eventuelt kun disse videre
til Claude (se analyze.py --sections-from-report) for faktisk klassifisering
av mål, skudd, frikast, utvisning osv.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ultralytics er ikke installert. Kjør: pip install ultralytics"
    ) from exc

from .events import MatchEvent
from .sections import TimeRange, merge_ranges, pad_ranges

PERSON_CLASS_ID = 0
BALL_CLASS_ID = 32  # "sports ball" i COCO

DEFAULT_YOLO_MODEL = "yolov8n.pt"

# Heuristikk-terskler (justerbare via HeuristicConfig)
DEFAULT_GOAL_ZONE_FRACTION = 0.15
DEFAULT_SHOT_SPEED_THRESHOLD = 0.5  # bildebredder/sekund
DEFAULT_MAX_BALL_GAP_SECONDS = 1.5
DEFAULT_FAST_BREAK_MAX_SECONDS = 4.0
DEFAULT_FAST_BREAK_MIN_DISTANCE_FRACTION = 0.6
DEFAULT_HIGH_DENSITY_PLAYER_COUNT = 6
DEFAULT_MIN_EVENT_GAP_SECONDS = 3.0


@dataclass
class FrameDetections:
    timestamp: float
    player_boxes: List[Tuple[float, float, float, float]] = field(default_factory=list)
    ball_position: Optional[Tuple[float, float]] = None
    ball_confidence: Optional[float] = None


class HandballYoloAnalyzer:
    """Kjører YOLO-deteksjon/-sporing på enkeltframes (lokalt, gratis)."""

    def __init__(self, model_path: str = DEFAULT_YOLO_MODEL, confidence: float = 0.3):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def analyze_frame(self, timestamp: float, frame: np.ndarray) -> FrameDetections:
        results = self.model.track(
            frame,
            persist=True,
            verbose=False,
            conf=self.confidence,
            classes=[PERSON_CLASS_ID, BALL_CLASS_ID],
        )[0]

        player_boxes: List[Tuple[float, float, float, float]] = []
        ball_position = None
        ball_confidence = None

        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                conf = float(box.conf[0])
                if cls_id == PERSON_CLASS_ID:
                    player_boxes.append((x1, y1, x2, y2))
                elif cls_id == BALL_CLASS_ID:
                    if ball_confidence is None or conf > ball_confidence:
                        ball_position = ((x1 + x2) / 2, (y1 + y2) / 2)
                        ball_confidence = conf

        return FrameDetections(timestamp, player_boxes, ball_position, ball_confidence)


@dataclass
class HeuristicConfig:
    goal_zone_fraction: float = DEFAULT_GOAL_ZONE_FRACTION
    shot_speed_threshold: float = DEFAULT_SHOT_SPEED_THRESHOLD
    max_ball_gap_seconds: float = DEFAULT_MAX_BALL_GAP_SECONDS
    fast_break_max_seconds: float = DEFAULT_FAST_BREAK_MAX_SECONDS
    fast_break_min_distance_fraction: float = DEFAULT_FAST_BREAK_MIN_DISTANCE_FRACTION
    high_density_player_count: int = DEFAULT_HIGH_DENSITY_PLAYER_COUNT
    min_event_gap_seconds: float = DEFAULT_MIN_EVENT_GAP_SECONDS


def _throttled_append(last_emitted: dict, event_type: str, min_gap: float, timestamp: float) -> bool:
    """Returner True (og oppdater last_emitted) hvis det er lenge nok siden forrige
    hendelse av samme type til at en ny hendelse bør legges til."""
    last = last_emitted.get(event_type)
    if last is not None and timestamp - last < min_gap:
        return False
    last_emitted[event_type] = timestamp
    return True


def derive_heuristics(
    detections: List[FrameDetections],
    frame_width: int,
    config: Optional[HeuristicConfig] = None,
) -> Tuple[List[MatchEvent], dict]:
    """Avled heuristiske hendelser og statistikk fra en tidsserie av deteksjoner."""
    config = config or HeuristicConfig()
    events: List[MatchEvent] = []
    last_emitted: dict = {}

    detections_sorted = sorted(detections, key=lambda d: d.timestamp)
    ball_points = [
        (d.timestamp, d.ball_position[0], d.ball_position[1])
        for d in detections_sorted
        if d.ball_position is not None
    ]

    goal_left = config.goal_zone_fraction * frame_width
    goal_right = (1 - config.goal_zone_fraction) * frame_width

    # --- Mulig skudd: rask ballbevegelse med ball i/mot en målsone ---
    for (t1, x1, y1), (t2, x2, y2) in zip(ball_points, ball_points[1:]):
        dt = t2 - t1
        if dt <= 0 or dt > config.max_ball_gap_seconds:
            continue
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        speed = (distance / frame_width) / dt
        near_goal = x2 <= goal_left or x2 >= goal_right
        if speed >= config.shot_speed_threshold and near_goal:
            if _throttled_append(last_emitted, "heuristic_possible_shot", config.min_event_gap_seconds, t2):
                confidence = min(1.0, speed / (config.shot_speed_threshold * 2))
                events.append(MatchEvent(
                    timestamp=t2,
                    event_type="heuristic_possible_shot",
                    description=(
                        f"Rask ballbevegelse ({speed:.2f} bildebredder/sek) mot målsone - "
                        "heuristikk, ikke semantisk bekreftet"
                    ),
                    team=None,
                    confidence=round(confidence, 2),
                ))

    # --- Mulig kontring: ballen krysser midtlinjen raskt over en kort periode ---
    half = frame_width / 2
    for i, (t1, x1, _) in enumerate(ball_points):
        for t2, x2, _ in ball_points[i + 1:]:
            dt = t2 - t1
            if dt > config.fast_break_max_seconds:
                break
            crossed_half = (x1 < half) != (x2 < half)
            distance_fraction = abs(x2 - x1) / frame_width
            if crossed_half and distance_fraction >= config.fast_break_min_distance_fraction:
                if _throttled_append(last_emitted, "heuristic_fast_break", config.min_event_gap_seconds, t2):
                    events.append(MatchEvent(
                        timestamp=t2,
                        event_type="heuristic_fast_break",
                        description=(
                            f"Ballen flyttet seg raskt over banehalvdel på {dt:.1f}s - "
                            "heuristikk, ikke semantisk bekreftet"
                        ),
                        team=None,
                        confidence=round(min(1.0, distance_fraction), 2),
                    ))
                break

    # --- Høy spillertetthet i en målsone ---
    for d in detections_sorted:
        if not d.player_boxes:
            continue
        count_in_zone = sum(
            1 for (x1, _, x2, _) in d.player_boxes
            if ((x1 + x2) / 2) <= goal_left or ((x1 + x2) / 2) >= goal_right
        )
        if count_in_zone >= config.high_density_player_count:
            if _throttled_append(last_emitted, "heuristic_high_density", config.min_event_gap_seconds, d.timestamp):
                events.append(MatchEvent(
                    timestamp=d.timestamp,
                    event_type="heuristic_high_density",
                    description=(
                        f"{count_in_zone} spillere registrert i målsone samtidig - "
                        "heuristikk, ikke semantisk bekreftet"
                    ),
                    team=None,
                    confidence=round(min(1.0, count_in_zone / (config.high_density_player_count * 2)), 2),
                ))

    total_frames = len(detections_sorted)
    frames_with_ball = len(ball_points)
    avg_players = (
        sum(len(d.player_boxes) for d in detections_sorted) / total_frames
        if total_frames else 0.0
    )

    stats = {
        "total_frames_analyzed": total_frames,
        "frames_with_ball_detected": frames_with_ball,
        "ball_detection_rate": round(frames_with_ball / total_frames, 3) if total_frames else 0.0,
        "avg_players_detected_per_frame": round(avg_players, 2),
    }

    return sorted(events, key=lambda e: e.timestamp), stats


def suggested_sections_from_events(
    events: List[MatchEvent], pad_seconds: float, min_gap_seconds: float, max_time: float
) -> List[TimeRange]:
    """Bygg en liste med (start, slutt)-intervaller rundt heuristiske hendelser,
    egnet til å sende videre til Claude for faktisk klassifisering."""
    if not events:
        return []
    raw_ranges = [(e.timestamp, e.timestamp) for e in events]
    padded = pad_ranges(raw_ranges, pad_seconds, max_time)
    return merge_ranges(padded, min_gap_seconds)
