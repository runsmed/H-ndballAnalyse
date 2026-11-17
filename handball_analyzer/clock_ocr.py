"""Les kampklokka automatisk fra videoens klokkegrafikk (OCR).

De fleste håndball-sendinger har en fast klokkegrafikk (typisk midt øverst
i bildet) som viser forløpt tid i omgangen. Hvis vi kan lese den, slipper
brukeren å manuelt finne --period1-start/--period2-start for
build_training_clips.py - vi kan utlede dem automatisk.

Metode: sample klokka med jevne mellomrom gjennom hele videoen. For hver
avlesning beregner vi "anker-kandidat" = video_tidspunkt - klokke_sekunder.
Siden klokka går i sanntid innenfor én omgang (ingen klipp/kutt), er dette
anker-tallet KONSTANT for alle riktige avlesninger i samme omgang - feil
avlesninger (OCR-støy) havner langt unna og filtreres bort automatisk ved
klynging. Vi trenger derfor ikke finne nøyaktig hvor klokka nullstilles -
bare nok riktige stikkprøver spredt utover hver omgang.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract

CLOCK_RE = re.compile(r"^(\d{1,2}):(\d{2})$")

# Standard gjetning: midt øverst, ca. 20% av bredden, øverste 12% av høyden.
# Finjuster med --crop etter å ha sjekket en forhåndsvisning.
DEFAULT_CROP = (0.40, 0.00, 0.60, 0.12)


def save_crop_preview(video_path: str, crop: Tuple[float, float, float, float], timestamp: float, out_path: str) -> None:
    """Lagre et videobilde med beskjæringsboksen tegnet på, for visuell kontroll."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Kunne ikke åpne videofil: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ret, frame = cap.read()
        if not ret:
            raise IOError(f"Kunne ikke lese frame ved {timestamp}s")
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = _crop_to_pixels(crop, width, height)
        preview = frame.copy()
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite(out_path, preview)

        crop_img = frame[y1:y2, x1:x2]
        crop_out_path = out_path.rsplit(".", 1)[0] + "_crop.png"
        cv2.imwrite(crop_out_path, cv2.resize(crop_img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC))
    finally:
        cap.release()


def _crop_to_pixels(crop: Tuple[float, float, float, float], width: int, height: int) -> Tuple[int, int, int, int]:
    x1f, y1f, x2f, y2f = crop
    return int(x1f * width), int(y1f * height), int(x2f * width), int(y2f * height)


def read_clock_from_frame(frame: np.ndarray, crop: Tuple[float, float, float, float]) -> Optional[float]:
    """Les klokkeverdi (sekunder) fra ett videobilde, eller None hvis ikke lesbart."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = _crop_to_pixels(crop, width, height)
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return None

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    for variant in (gray, cv2.bitwise_not(gray)):
        _, thresh = cv2.threshold(variant, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(
            thresh, config="--psm 7 -c tessedit_char_whitelist=0123456789:"
        ).strip()
        match = CLOCK_RE.match(text)
        if match:
            minutes, seconds = int(match.group(1)), int(match.group(2))
            if minutes <= 45 and seconds <= 59:
                return minutes * 60 + seconds
    return None


@dataclass
class ClockSample:
    video_timestamp: float
    clock_seconds: float


def sample_clock(
    video_path: str,
    crop: Tuple[float, float, float, float],
    sample_interval: float = 5.0,
) -> List[ClockSample]:
    """Les klokka med jevne mellomrom gjennom hele videoen."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Kunne ikke åpne videofil: {video_path}")

    duration_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration = duration_frames / fps if fps else 0.0

    samples: List[ClockSample] = []
    try:
        t = 0.0
        while t < duration:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ret, frame = cap.read()
            if ret:
                clock_seconds = read_clock_from_frame(frame, crop)
                if clock_seconds is not None:
                    samples.append(ClockSample(t, clock_seconds))
            t += sample_interval
    finally:
        cap.release()

    return samples


def detect_period_anchors(
    samples: List[ClockSample],
    cluster_tolerance: float = 3.0,
    min_samples_per_period: int = 3,
) -> Dict[int, float]:
    """Grupper anker-kandidater (video_tid - klokke_tid) i klynger -> én klynge per omgang."""
    candidates = sorted(s.video_timestamp - s.clock_seconds for s in samples)
    if not candidates:
        return {}

    clusters: List[List[float]] = [[candidates[0]]]
    for value in candidates[1:]:
        if value - clusters[-1][-1] <= cluster_tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])

    clusters = [c for c in clusters if len(c) >= min_samples_per_period]
    clusters.sort(key=lambda c: c[0])

    anchors = {}
    for period_index, cluster in enumerate(clusters, start=1):
        sorted_cluster = sorted(cluster)
        anchors[period_index] = sorted_cluster[len(sorted_cluster) // 2]  # median
    return anchors
