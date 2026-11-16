#!/usr/bin/env python3
"""CLI for å bygge et merket treningsklipp-datasett fra video + offisielt kampreferat.

Dette er det første steget mot å trene en lokal action-spotting-modell
(f.eks. I3D) på dine egne kamper: den tar en videofil og et offisielt
kampreferat (tekstformatet fra live-rapportering), og klipper ut korte,
merkede videosegmenter rundt hvert hendelsestidspunkt.

VIKTIG: Du må oppgi hvilket tidspunkt I VIDEOEN hver omgang starter
(klokka viser 00:00 der) - dette kan ikke gjettes automatisk. Finn dette
ved å spole frem til start-fløyten i videoen og notere tidspunktet.

Bruk:
    python build_training_clips.py --video kamp.mp4 --log kampreferat.txt \
        --period1-start 125 --period2-start 1960 --output-dir clips/
"""
from __future__ import annotations

import argparse
import sys

from handball_analyzer.clip_dataset import build_clip_dataset, print_dataset_summary
from handball_analyzer.match_log_parser import parse_match_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bygg merkede treningsklipp fra video + offisielt kampreferat.",
    )
    parser.add_argument("--video", required=True, help="Sti til videofil")
    parser.add_argument("--log", required=True, help="Sti til tekstfil med kampreferatet")
    parser.add_argument(
        "--period1-start", type=float, required=True,
        help="Video-tidspunkt (sekunder) der 1. omgangs klokke viser 00:00",
    )
    parser.add_argument(
        "--period2-start", type=float, default=None,
        help="Video-tidspunkt (sekunder) der 2. omgangs klokke viser 00:00 "
             "(utelat hvis videoen kun dekker 1. omgang)",
    )
    parser.add_argument(
        "--output-dir", default="clips", help="Mappe for klipp + manifest.json (default: clips)",
    )
    parser.add_argument(
        "--clip-before", type=float, default=3.0,
        help="Sekunder før hendelsestidspunktet som tas med i klippet (default: 3)",
    )
    parser.add_argument(
        "--clip-after", type=float, default=1.0,
        help="Sekunder etter hendelsestidspunktet som tas med i klippet (default: 1)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        with open(args.log, "r", encoding="utf-8") as f:
            log_text = f.read()
    except OSError as exc:
        print(f"Kunne ikke lese kampreferat: {exc}", file=sys.stderr)
        return 1

    result = parse_match_log(log_text)
    print(
        f"Fant {len(result.events)} hendelser, {len(result.period_markers)} periodemarkører, "
        f"hoppet over {len(result.skipped)} (spillerbytter e.l.)"
    )
    if result.unrecognized:
        print(f"ADVARSEL: {len(result.unrecognized)} blokker ble ikke gjenkjent:", file=sys.stderr)
        for block in result.unrecognized[:10]:
            print(f"  {block}", file=sys.stderr)

    period_anchors = {1: args.period1_start}
    if args.period2_start is not None:
        period_anchors[2] = args.period2_start

    half_2_events = [e for e in result.events if e.half == 2]
    if half_2_events and 2 not in period_anchors:
        print(
            f"ADVARSEL: Fant {len(half_2_events)} hendelser i 2. omgang, men "
            f"--period2-start er ikke oppgitt - disse hoppes over.",
            file=sys.stderr,
        )

    try:
        manifest = build_clip_dataset(
            args.video, result.events, period_anchors, args.output_dir,
            clip_before_seconds=args.clip_before, clip_after_seconds=args.clip_after,
        )
    except IOError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print_dataset_summary(manifest)
    print(f"\nManifest lagret til {args.output_dir}/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
