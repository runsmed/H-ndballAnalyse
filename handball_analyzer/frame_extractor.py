"""Ekstraher frames fra en videofil med jevne tidsintervaller."""
from __future__ import annotations

from typing import Iterator, Optional, Sequence, Tuple

import cv2
import numpy as np


def extract_frames(
    video_path: str,
    interval_seconds: float,
    max_dimension: int = 768,
    time_ranges: Optional[Sequence[Tuple[float, float]]] = None,
) -> Iterator[Tuple[float, np.ndarray]]:
    """Yield (tidsstempel_sekunder, frame) samplet med gitt intervall.

    `frame` er et nedskalert BGR numpy-array (ikke enkodet) - bruk
    `encode_jpeg()` for å konvertere til JPEG-bytes ved behov (f.eks. før
    sending til Claude). Hvis `time_ranges` er satt, hoppes det direkte til
    disse intervallene i stedet for å dekode hele videoen sekvensielt -
    nyttig når man kun vil analysere noen få seksjoner av en lang video.
    """
    if time_ranges:
        yield from _extract_frames_in_ranges(
            video_path, interval_seconds, max_dimension, time_ranges
        )
        return

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
                yield timestamp, _resize(frame, max_dimension)
            frame_index += 1
    finally:
        cap.release()


def _extract_frames_in_ranges(
    video_path: str,
    interval_seconds: float,
    max_dimension: int,
    time_ranges: Sequence[Tuple[float, float]],
) -> Iterator[Tuple[float, np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Kunne ikke åpne videofil: {video_path}")

    try:
        for start, end in time_ranges:
            timestamp = max(start, 0.0)
            while timestamp <= end:
                cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
                ret, frame = cap.read()
                if not ret:
                    break
                yield timestamp, _resize(frame, max_dimension)
                timestamp += interval_seconds
    finally:
        cap.release()


def encode_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    """Enkod et frame-array til JPEG-bytes."""
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Kunne ikke enkode frame til JPEG")
    return buffer.tobytes()


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


def _resize(frame: np.ndarray, max_dimension: int) -> np.ndarray:
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= max_dimension:
        return frame
    scale = max_dimension / longest
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
