#!/usr/bin/env python3
"""CLI for AI-basert analyse av håndballkamp-video ved hjelp av Google Gemini.

I motsetning til analyze.py (Claude) sender denne HELE videofilen direkte
til Gemini - ingen egen frame-ekstraksjon nødvendig, Gemini gjør det selv.

Bruk:
    python analyze_gemini.py --video kamp.mp4 --output rapport_gemini.json

Krever miljøvariabelen GEMINI_API_KEY (gratis å opprette på
https://aistudio.google.com/apikey). Merk: en betalt Google AI Pro/Ultra-
abonnement for Gemini-appen er IKKE det samme som API-tilgang - sjekk
aistudio.google.com med din konto for å se hva den faktisk gir deg av kvote.

Gemini sitt gratis-nivå har lave hastighetsgrenser (typisk 10-15
forespørsler/minutt) og skifter ofte hvilken modell som er gratis - dette
verktøyet bruker som default modellalias-et "gemini-flash-latest", som
Google selv oppdaterer til å peke på nyeste Flash-modell.
"""
from __future__ import annotations

import argparse
import os
import sys

from handball_analyzer.frame_extractor import get_video_duration
from handball_analyzer.gemini_analyzer import DEFAULT_MODEL, GeminiVideoAnalyzer
from handball_analyzer.report import build_report, print_summary, save_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyser video av håndballkamp med Google Gemini (hel video, ikke enkeltbilder).",
    )
    parser.add_argument("--video", required=True, help="Sti til videofil (mp4, mov, ...)")
    parser.add_argument(
        "--output", default="rapport_gemini.json",
        help="Filsti for JSON-rapport (default: rapport_gemini.json)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Gemini-modell som brukes (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--chunk-seconds", type=float, default=600.0,
        help="Sekunder video analysert per API-kall (default: 600 = 10 min). "
             "Lengre kamper deles automatisk opp i flere kall.",
    )
    parser.add_argument(
        "--fps", type=float, default=1.0,
        help="Bilder per sekund Gemini sampler internt fra videoen (default: 1.0, maks 24.0)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Ikke skriv sammendrag til konsoll (kun lagre JSON-rapport)",
    )
    parser.add_argument(
        "-y", "--yes", action="store_true",
        help="Ikke spør om bekreftelse før API-kall (for bruk i skript/CI)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not os.path.isfile(args.video):
        print(f"Fant ikke videofil: {args.video}", file=sys.stderr)
        return 1

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print(
            "Miljøvariabelen GEMINI_API_KEY (eller GOOGLE_API_KEY) er ikke satt.\n"
            "Hent en gratis nøkkel på https://aistudio.google.com/apikey",
            file=sys.stderr,
        )
        return 1

    try:
        duration = get_video_duration(args.video)
    except IOError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    num_chunks = max(1, int(-(-duration // args.chunk_seconds))) if duration > 0 else 1
    print(
        f"Video: {args.video} ({duration:.0f} sekunder)\n"
        f"Modell: {args.model}\n"
        f"Vil gjøre {num_chunks} API-kall à {args.chunk_seconds:.0f} sekunder video hver.\n"
        "\n"
        "Merk: dette er trolig gratis hvis du er innenfor Geminis gratis-kvote for "
        "denne modellen (sjekk selv på aistudio.google.com), men det er IKKE det "
        "samme som et betalt Gemini-abonnement for chat-appen. Hastighetsgrensen på "
        "gratis-nivået er lav, så dette kan ta noen minutter for lengre kamper."
    )

    if not args.yes:
        answer = input("\nFortsette og sende videoen til Gemini API? [y/N] ").strip().lower()
        if answer not in ("y", "yes", "j", "ja"):
            print("Avbrutt av bruker.")
            return 0

    analyzer = GeminiVideoAnalyzer(api_key=api_key, model=args.model)

    def on_progress(chunk_index: int, total_chunks: int, start: float, end: float) -> None:
        print(f"  Segment {chunk_index}/{total_chunks} ({start:.0f}s-{end:.0f}s)...")

    print("\nLaster opp video til Gemini...")
    try:
        events = analyzer.analyze_video(
            args.video, chunk_seconds=args.chunk_seconds, fps=args.fps, on_progress=on_progress
        )
    except RuntimeError as exc:
        print(f"Analyse feilet: {exc}", file=sys.stderr)
        return 1

    report = build_report(events, args.video, duration)
    save_report(report, args.output)
    print(f"\nRapport lagret til {args.output}")

    if not args.quiet:
        print_summary(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
