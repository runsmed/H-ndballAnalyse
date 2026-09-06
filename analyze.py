#!/usr/bin/env python3
"""CLI for AI-basert analyse av håndballkamp-video ved hjelp av Claude.

Bruk:
    python analyze.py --video kamp.mp4 --interval 1 --output rapport.json
"""
from __future__ import annotations

import argparse
import os
import sys

from handball_analyzer.events import format_timestamp
from handball_analyzer.frame_extractor import extract_frames, get_video_duration
from handball_analyzer.report import build_report, print_summary, save_report
from handball_analyzer.vision_analyzer import DEFAULT_MODEL, ClaudeVisionAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyser video av håndballkamp med Claude AI.",
    )
    parser.add_argument("--video", required=True, help="Sti til videofil (mp4, mov, ...)")
    parser.add_argument(
        "--interval", type=float, default=1.0,
        help="Sekunder mellom hvert frame som analyseres (default: 1)",
    )
    parser.add_argument(
        "--output", default="rapport.json",
        help="Filsti for JSON-rapport (default: rapport.json)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=5,
        help="Antall frames sendt per API-kall (default: 5)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Claude-modell som brukes (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Maks antall frames å analysere (nyttig for testing/kostnadskontroll)",
    )
    parser.add_argument(
        "--max-dimension", type=int, default=768,
        help="Maks bredde/høyde på frames før sending til API (default: 768px)",
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

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Miljøvariabelen ANTHROPIC_API_KEY er ikke satt.", file=sys.stderr)
        return 1

    try:
        duration = get_video_duration(args.video)
        frames = list(extract_frames(args.video, args.interval, args.max_dimension))
    except IOError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.max_frames:
        frames = frames[: args.max_frames]

    if not frames:
        print("Fant ingen frames å analysere.", file=sys.stderr)
        return 1

    print(
        f"Analyserer {len(frames)} frames fra {args.video} "
        f"(intervall: {args.interval}s, batch-størrelse: {args.batch_size}, "
        f"modell: {args.model})..."
    )

    analyzer = ClaudeVisionAnalyzer(api_key=api_key, model=args.model)
    all_events = []
    total_batches = (len(frames) + args.batch_size - 1) // args.batch_size

    for batch_index, batch_start in enumerate(range(0, len(frames), args.batch_size), start=1):
        batch = frames[batch_start: batch_start + args.batch_size]
        time_range = f"{format_timestamp(batch[0][0])}-{format_timestamp(batch[-1][0])}"
        print(f"  Batch {batch_index}/{total_batches} ({time_range})...")
        try:
            events = analyzer.analyze_batch(batch)
        except RuntimeError as exc:
            print(f"  Advarsel: batch feilet, hopper over: {exc}", file=sys.stderr)
            continue
        all_events.extend(events)

    report = build_report(all_events, args.video, duration)
    save_report(report, args.output)
    print(f"\nRapport lagret til {args.output}")

    if not args.quiet:
        print_summary(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
