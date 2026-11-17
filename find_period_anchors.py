#!/usr/bin/env python3
"""CLI for å finne periode-ankerpunkter automatisk ved å lese kampklokka.

Erstatter manuell leting etter --period1-start/--period2-start til
build_training_clips.py, ved å lese klokkegrafikken (typisk midt øverst i
bildet) gjennom hele videoen.

Bruk i to steg:

1. Forhåndsvisning - sjekk at beskjæringsboksen faktisk treffer klokka:
   python find_period_anchors.py --video kamp.mp4 --preview-at 120

   Åpne crop_preview.png (hele bildet med rød boks) og
   crop_preview_crop.png (selve utsnittet, forstørret) - juster --crop
   hvis boksen ikke treffer klokka presist.

2. Full kjøring - les klokka gjennom hele videoen og regn ut ankerpunkter:
   python find_period_anchors.py --video kamp.mp4

Output kan brukes direkte som --period1-start/--period2-start til
build_training_clips.py.
"""
from __future__ import annotations

import argparse
import sys

from handball_analyzer.clock_ocr import (
    DEFAULT_CROP,
    detect_period_anchors,
    sample_clock,
    save_crop_preview,
)
from handball_analyzer.events import format_timestamp


def parse_crop(value: str) -> tuple:
    parts = [float(p) for p in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--crop må være 4 tall: x1,y1,x2,y2 (andel av bredde/høyde, 0-1)")
    return tuple(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finn periode-ankerpunkter automatisk ved å lese kampklokka (OCR).",
    )
    parser.add_argument("--video", required=True, help="Sti til videofil")
    parser.add_argument(
        "--crop", type=parse_crop, default=DEFAULT_CROP, metavar="X1,Y1,X2,Y2",
        help=f"Beskjæringsboks rundt klokka, som andel av bildets bredde/høyde "
             f"(default: {','.join(str(v) for v in DEFAULT_CROP)} - midt øverst)",
    )
    parser.add_argument(
        "--preview-at", type=float, default=None, metavar="SEKUNDER",
        help="Lagre kun en forhåndsvisning av beskjæringsboksen ved dette tidspunktet, og avslutt",
    )
    parser.add_argument(
        "--preview-output", default="crop_preview.png",
        help="Filnavn for forhåndsvisning (default: crop_preview.png)",
    )
    parser.add_argument(
        "--sample-interval", type=float, default=5.0,
        help="Sekunder mellom hver klokkeavlesning (default: 5)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.preview_at is not None:
        try:
            save_crop_preview(args.video, args.crop, args.preview_at, args.preview_output)
        except IOError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        crop_path = args.preview_output.rsplit(".", 1)[0] + "_crop.png"
        print(f"Forhåndsvisning lagret: {args.preview_output} (hele bildet) og {crop_path} (beskjæring, forstørret)")
        print("Sjekk at den røde boksen treffer klokka presist. Juster --crop og prøv igjen om nødvendig.")
        return 0

    print(f"Leser klokka hvert {args.sample_interval:.0f}. sekund gjennom hele videoen (kan ta litt tid)...")
    samples = sample_clock(args.video, args.crop, args.sample_interval)
    print(f"Klarte å lese klokka på {len(samples)} tidspunkter.")

    if len(samples) < 6:
        print(
            "For få vellykkede avlesninger til å finne ankerpunkter pålitelig. "
            "Sjekk --crop med --preview-at først.",
            file=sys.stderr,
        )
        return 1

    anchors = detect_period_anchors(samples)
    if not anchors:
        print("Fant ingen tydelige perioder - klokkeavlesningene var for spredte/inkonsistente.", file=sys.stderr)
        return 1

    print(f"\nFant {len(anchors)} omgang(er):")
    for period, anchor in sorted(anchors.items()):
        print(f"  Omgang {period}: starter ved {anchor:.1f}s i videoen ({format_timestamp(anchor)})")

    print("\nBruk disse direkte i build_training_clips.py:")
    flags = " ".join(f"--period{p}-start {a:.1f}" for p, a in sorted(anchors.items()))
    print(f"  {flags}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
