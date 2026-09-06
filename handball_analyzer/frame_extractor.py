"""Ekstraher frames fra en videofil med jevne tidsintervaller."""
from __future__ import annotations

from typing import Iterator, Tuple

import cv2


def extract_frames(
    video_path: str,
    interval_seconds: float,
    max_dimension: int = 768,
) -> Iterator[Tuple[float, bytes]]:
    """Yield (tidsstempel_sekunder, jpeg_bytes) for frames samplet med gitt intervall."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Kunne ikke åpne videofil: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(int(round(fps * interval_seconds)), 1)
    frame_index = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % step == 0:
                timestamp = frame_index / fps
                frame = _resize(frame, max_dimension)
                ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    yield timestamp, buffer.tobytes()
            frame_index += 1
    finally:
        cap.release()


def get_video_duration(video_path: str) -> float:
    """Returner varighet i sekunder for videofilen."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Kunne ikke åpne videofil: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count / fps if fps else 0.0


def get_video_info(video_path: str) -> tuple[float, int, int]:
    """Returner (varighet_sekunder, bredde_px, høyde_px) uten å dekode frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Kunne ikke åpne videofil: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = frame_count / fps if fps else 0.0
    return duration, width, height


def _resize(frame, max_dimension: int):
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= max_dimension:
        return frame
    scale = max_dimension / longest
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
