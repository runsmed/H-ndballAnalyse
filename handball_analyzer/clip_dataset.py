"""Bygg et merket treningsklipp-datasett fra en videofil + parsede kamphendelser.

Hvert klipp er et kort videosegment rundt en hendelses tidspunkt, lagret i
en mappe navngitt etter hendelsestype (f.eks. clips/goal/, clips/turnover/),
sammen med en manifest.json som lister alle klipp og metadata. Dette er
treningsdataen en lokal action-spotting-modell (f.eks. I3D) kan trenes på.

Spillerbytter og periodemarkører er allerede filtrert bort av
match_log_parser.py og inngår ikke her.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, List

import cv2

from .frame_extractor import get_video_info
from .match_log_parser import ParsedLogEntry


def build_clip_dataset(
    video_path: str,
    events: List[ParsedLogEntry],
    period_anchors: Dict[int, float],
    output_dir: str,
    clip_before_seconds: float = 3.0,
    clip_after_seconds: float = 1.0,
) -> dict:
    """Klipp ut ett videosegment per hendelse og skriv en manifest.json.

    `period_anchors` må inneholde video-tidspunktet (sekunder fra start av
    VIDEOFILEN) der klokka for hver omgang viser 00:00, f.eks.
    {1: 125.0, 2: 1960.0}.
    """
    duration, width, height = get_video_info(video_path)
    fps = _get_fps(video_path)

    os.makedirs(output_dir, exist_ok=True)

    manifest_clips = []
    out_of_range = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Kunne ikke åpne videofil: {video_path}")

    try:
        for index, event in enumerate(events):
            if event.half not in period_anchors:
                out_of_range.append({"index": index, "reason": f"ingen anker for omgang {event.half}", "event": asdict(event)})
                continue

            absolute_timestamp = period_anchors[event.half] + event.clock_seconds
            start = max(0.0, absolute_timestamp - clip_before_seconds)
            end = min(duration, absolute_timestamp + clip_after_seconds)

            if start >= duration or end <= start:
                out_of_range.append({"index": index, "reason": "utenfor videoens lengde", "event": asdict(event)})
                continue

            type_dir = os.path.join(output_dir, event.event_type)
            os.makedirs(type_dir, exist_ok=True)
            clip_filename = f"{index:04d}_{event.half}h_{int(absolute_timestamp)}s.mp4"
            clip_path = os.path.join(type_dir, clip_filename)

            _extract_clip(cap, start, end, fps, width, height, clip_path)

            manifest_clips.append({
                "clip_path": os.path.relpath(clip_path, output_dir),
                "event_type": event.event_type,
                "shot_type": event.shot_type,
                "half": event.half,
                "clock_seconds": event.clock_seconds,
                "absolute_video_timestamp": round(absolute_timestamp, 2),
                "clip_start": round(start, 2),
                "clip_end": round(end, 2),
                "description": event.description,
                "player_name": event.player_name,
                "assist_player_name": event.assist_player_name,
                "caused_by_player_name": event.caused_by_player_name,
                "score": event.score,
            })
    finally:
        cap.release()

    manifest = {
        "video": video_path,
        "period_anchors": period_anchors,
        "clip_before_seconds": clip_before_seconds,
        "clip_after_seconds": clip_after_seconds,
        "total_clips": len(manifest_clips),
        "skipped_out_of_range": len(out_of_range),
        "clips": manifest_clips,
    }

    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    if out_of_range:
        with open(os.path.join(output_dir, "skipped.json"), "w", encoding="utf-8") as f:
            json.dump(out_of_range, f, ensure_ascii=False, indent=2)

    return manifest


def _get_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return fps


def _extract_clip(
    cap: cv2.VideoCapture,
    start: float,
    end: float,
    fps: float,
    width: int,
    height: int,
    out_path: str,
) -> None:
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        current = start
        while current <= end:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            current += 1.0 / fps
    finally:
        writer.release()


def print_dataset_summary(manifest: dict) -> None:
    from collections import Counter

    print(f"\n=== Treningsklipp-datasett bygget fra {manifest['video']} ===")
    print(f"Totalt {manifest['total_clips']} klipp lagret, {manifest['skipped_out_of_range']} hoppet over (utenfor video)\n")

    counts = Counter(c["event_type"] for c in manifest["clips"])
    print("Klipp per hendelsestype:")
    for event_type, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {event_type}: {count}")
