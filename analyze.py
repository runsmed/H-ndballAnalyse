#!/usr/bin/env python3
"""CLI for AI-basert analyse av håndballkamp-video ved hjelp av Claude.

Bruk:
    python analyze.py --video kamp.mp4 --interval 1 --output rapport.json

For å holde kostnaden nede kan du analysere kun utvalgte tidsseksjoner i
stedet for hele kampen, enten manuelt:
    python analyze.py --video kamp.mp4 --sections "12:30-13:00,45:10-45:40"

eller basert på forslag fra den gratis, lokale YOLO-analysen
(se analyze_yolo.py):
    python analyze_yolo.py --video kamp.mp4 --output yolo_rapport.json
    python analyze.py --video kamp.mp4 --sections-from-report yolo_rapport.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from handball_analyzer.cost_estimator import (
    estimate_frame_dimensions,
    estimate_run,
    format_estimate,
)
from handball_analyzer.events import format_timestamp
from handball_analyzer.frame_extractor import extract_frames, get_video_info
from handball_analyzer.reference_examples import (
    build_reference_prompt_text,
    load_reference_examples,
)
from handball_analyzer.report import build_report, print_summary, save_report
from handball_analyzer.sections import format_ranges, parse_time_ranges, total_duration
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
        "--sections", default=None,
        help="Kun analyser disse tidsintervallene, f.eks. '12:30-13:00,45:10-45:40' "
             "(default: hele videoen)",
    )
    parser.add_argument(
        "--sections-from-report", default=None, metavar="YOLO_RAPPORT.JSON",
        help="Bruk 'suggested_sections' fra en YOLO-rapport (analyze_yolo.py) i stedet "
             "for å angi --sections manuelt",
    )
    parser.add_argument(
        "--reference", action="append", default=[], metavar="RETTET_RAPPORT.JSON",
        help="Sti til en tidligere, manuelt rettet kamprapport som brukes som "
             "few-shot-eksempel i prompten (kan gjentas for flere eksempler)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Ikke skriv sammendrag til konsoll (kun lagre JSON-rapport)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Vis kostnadsanslag og avslutt uten å sende noe til Claude API",
    )
    parser.add_argument(
        "-y", "--yes", action="store_true",
        help="Ikke spør om bekreftelse før API-kall (for bruk i skript/CI)",
    )
    return parser.parse_args()


def _resolve_time_ranges(args: argparse.Namespace):
    if args.sections and args.sections_from_report:
        raise ValueError("Bruk enten --sections eller --sections-from-report, ikke begge.")

    if args.sections:
        return parse_time_ranges(args.sections)

    if args.sections_from_report:
        with open(args.sections_from_report, "r", encoding="utf-8") as f:
            yolo_report = json.load(f)
        sections = yolo_report.get("suggested_sections", [])
        if not sections:
            raise ValueError(
                f"Fant ingen 'suggested_sections' i {args.sections_from_report}."
            )
        return [(s["start"], s["end"]) for s in sections]

    return None


def main() -> int:
    args = parse_args()

    if not os.path.isfile(args.video):
        print(f"Fant ikke videofil: {args.video}", file=sys.stderr)
        return 1

    try:
        time_ranges = _resolve_time_ranges(args)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"Feil ved lesing av seksjoner: {exc}", file=sys.stderr)
        return 1

    try:
        reference_examples = load_reference_examples(args.reference)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"Feil ved lesing av referanseeksempel: {exc}", file=sys.stderr)
        return 1
    reference_text = build_reference_prompt_text(reference_examples)
    if reference_examples:
        print(f"Bruker {len(reference_examples)} referanseeksempel(er): {', '.join(args.reference)}")

    try:
        duration, orig_width, orig_height = get_video_info(args.video)
    except IOError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if time_ranges:
        print(f"Analyserer kun {len(time_ranges)} seksjon(er): {format_ranges(time_ranges)}")
        analyzed_seconds = total_duration(time_ranges)
    else:
        analyzed_seconds = duration

    # Anslå antall frames uten å dekode hele videoen, slik at vi kan vise et
    # kostnadsanslag før noe faktisk sendes til API-et.
    estimated_num_frames = (
        int(analyzed_seconds // args.interval) + 1 if analyzed_seconds > 0 else 0
    )
    if args.max_frames:
        estimated_num_frames = min(estimated_num_frames, args.max_frames)

    resized_width, resized_height = estimate_frame_dimensions(
        orig_width, orig_height, args.max_dimension
    )
    estimate = estimate_run(
        num_frames=estimated_num_frames,
        batch_size=args.batch_size,
        frame_width=resized_width,
        frame_height=resized_height,
        model=args.model,
    )
    print(format_estimate(estimate))

    if args.dry_run:
        print("\n--dry-run: avslutter uten å kalle Claude API.")
        return 0

    if not args.yes:
        answer = input("\nFortsette og sende disse frames til Claude API? [y/N] ").strip().lower()
        if answer not in ("y", "yes", "j", "ja"):
            print("Avbrutt av bruker.")
            return 0

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Miljøvariabelen ANTHROPIC_API_KEY er ikke satt.", file=sys.stderr)
        return 1

    try:
        frames = list(extract_frames(args.video, args.interval, args.max_dimension, time_ranges))
    except IOError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.max_frames:
        frames = frames[: args.max_frames]

    if not frames:
        print("Fant ingen frames å analysere.", file=sys.stderr)
        return 1

    print(
        f"\nAnalyserer {len(frames)} frames fra {args.video} "
        f"(intervall: {args.interval}s, batch-størrelse: {args.batch_size}, "
        f"modell: {args.model})..."
    )

    analyzer = ClaudeVisionAnalyzer(api_key=api_key, model=args.model, reference_text=reference_text)
    all_events = []
    total_batches = (len(frames) + args.batch_size - 1) // args.batch_size

    for batch_index, batch_start in enumerate(range(0, len(frames), args.batch_size), start=1):
        batch = frames[batch_start: batch_start + args.batch_size]
        batch_time_range = f"{format_timestamp(batch[0][0])}-{format_timestamp(batch[-1][0])}"
        print(f"  Batch {batch_index}/{total_batches} ({batch_time_range})...")
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
