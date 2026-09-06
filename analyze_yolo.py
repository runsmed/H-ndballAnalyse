#!/usr/bin/env python3
"""CLI for lokal, GRATIS håndballanalyse med YOLO - ingen Claude API-kall.

YOLO kjører lokalt og koster ingenting, men forstår kun objekter (ball,
spillere) og posisjonene deres - ikke regler eller semantikk. Denne CLI-en
gir deg derfor heuristiske forslag til interessante tidsseksjoner
(mulig skudd, mulig kontring, høy spillertetthet), som du deretter kan
velge å sende videre til Claude for faktisk klassifisering av mål, skudd,
frikast, utvisning osv.

Bruk:
    python analyze_yolo.py --video kamp.mp4 --output yolo_rapport.json
    python analyze.py --video kamp.mp4 --sections-from-report yolo_rapport.json
"""
from __future__ import annotations

import argparse
import os
import sys

from handball_analyzer.frame_extractor import extract_frames, get_video_info
from handball_analyzer.report import build_yolo_report, print_yolo_summary, save_report
from handball_analyzer.yolo_analyzer import (
    DEFAULT_YOLO_MODEL,
    HandballYoloAnalyzer,
    HeuristicConfig,
    derive_heuristics,
    suggested_sections_from_events,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lokal, gratis YOLO-basert forhåndsanalyse av håndballkamp-video.",
    )
    parser.add_argument("--video", required=True, help="Sti til videofil (mp4, mov, ...)")
    parser.add_argument(
        "--interval", type=float, default=0.5,
        help="Sekunder mellom hvert frame som analyseres (default: 0.5 - YOLO er gratis, "
             "så finere sampling koster kun tid)",
    )
    parser.add_argument(
        "--output", default="yolo_rapport.json",
        help="Filsti for JSON-rapport (default: yolo_rapport.json)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_YOLO_MODEL,
        help=f"YOLO-modellvekter (default: {DEFAULT_YOLO_MODEL}, lastes ned automatisk)",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.3,
        help="Minimum deteksjonssikkerhet for YOLO (default: 0.3)",
    )
    parser.add_argument(
        "--max-dimension", type=int, default=768,
        help="Maks bredde/høyde på frames før YOLO-analyse (default: 768px)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Maks antall frames å analysere (nyttig for testing)",
    )
    parser.add_argument(
        "--goal-zone-fraction", type=float, default=0.15,
        help="Andel av bildebredden som regnes som målsone i hver ende (default: 0.15)",
    )
    parser.add_argument(
        "--pad-seconds", type=float, default=5.0,
        help="Sekunder lagt til før/etter hver foreslått seksjon (default: 5)",
    )
    parser.add_argument(
        "--min-gap-seconds", type=float, default=10.0,
        help="Slå sammen foreslåtte seksjoner som er nærmere hverandre enn dette (default: 10)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Ikke skriv sammendrag til konsoll (kun lagre JSON-rapport)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not os.path.isfile(args.video):
        print(f"Fant ikke videofil: {args.video}", file=sys.stderr)
        return 1

    try:
        duration, _orig_width, _orig_height = get_video_info(args.video)
        frames = list(extract_frames(args.video, args.interval, args.max_dimension))
    except IOError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.max_frames:
        frames = frames[: args.max_frames]

    if not frames:
        print("Fant ingen frames å analysere.", file=sys.stderr)
        return 1

    frame_height, frame_width = frames[0][1].shape[:2]

    print(
        f"Analyserer {len(frames)} frames lokalt med YOLO "
        f"(intervall: {args.interval}s, modell: {args.model}) - ingen API-kostnad..."
    )

    analyzer = HandballYoloAnalyzer(model_path=args.model, confidence=args.confidence)
    detections = []
    for i, (timestamp, frame) in enumerate(frames, start=1):
        detections.append(analyzer.analyze_frame(timestamp, frame))
        if i % 50 == 0 or i == len(frames):
            print(f"  {i}/{len(frames)} frames analysert...")

    config = HeuristicConfig(goal_zone_fraction=args.goal_zone_fraction)
    events, stats = derive_heuristics(detections, frame_width, config)
    suggested_sections = suggested_sections_from_events(
        events, args.pad_seconds, args.min_gap_seconds, max_time=duration
    )

    report = build_yolo_report(events, stats, suggested_sections, args.video, duration)
    save_report(report, args.output)
    print(f"\nYOLO-rapport lagret til {args.output}")

    if not args.quiet:
        print_yolo_summary(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
