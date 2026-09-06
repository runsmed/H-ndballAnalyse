"""Bygg kamprapport (hendelseslogg + statistikk) fra identifiserte hendelser."""
from __future__ import annotations

import json
from collections import Counter
from typing import List

from .events import EVENT_TYPES, TACTIC_EVENT_TYPES, MatchEvent, format_timestamp
from .sections import TimeRange, format_ranges


def build_report(events: List[MatchEvent], video_path: str, duration: float) -> dict:
    events_sorted = sorted(events, key=lambda e: e.timestamp)
    counts = Counter(e.event_type for e in events_sorted)

    return {
        "video": video_path,
        "duration_seconds": round(duration, 1),
        "duration_formatted": format_timestamp(duration),
        "total_events": len(events_sorted),
        "statistics": {
            EVENT_TYPES.get(key, key): count for key, count in counts.items()
        },
        "tactical_observations": _summarize_tactics(events_sorted),
        "event_log": [e.to_dict() for e in events_sorted],
    }


def _summarize_tactics(events: List[MatchEvent]) -> List[str]:
    tactic_events = [e for e in events if e.event_type in TACTIC_EVENT_TYPES]
    if not tactic_events:
        return []

    counts = Counter(e.event_type for e in tactic_events)
    observations = [
        f"{EVENT_TYPES.get(event_type, event_type)} observert {count} gang(er) i løpet av kampen."
        for event_type, count in counts.most_common()
    ]

    first = tactic_events[0]
    observations.append(
        f"Første taktiske mønster ({EVENT_TYPES.get(first.event_type, first.event_type)}) "
        f"observert ved {first.timecode}."
    )
    return observations


def build_yolo_report(
    events: List[MatchEvent],
    stats: dict,
    suggested_sections: List[TimeRange],
    video_path: str,
    duration: float,
) -> dict:
    """Bygg en rapport fra den lokale, gratis YOLO-analysen.

    Hendelsene her er heuristikker avledet fra ball-/spillerbevegelse, ikke
    semantisk bekreftede håndballhendelser. `suggested_sections` er
    tidsintervaller som kan sendes videre til Claude (via
    analyze.py --sections-from-report) for faktisk klassifisering.
    """
    events_sorted = sorted(events, key=lambda e: e.timestamp)
    counts = Counter(e.event_type for e in events_sorted)

    return {
        "video": video_path,
        "duration_seconds": round(duration, 1),
        "duration_formatted": format_timestamp(duration),
        "analysis_type": "yolo_local_heuristic",
        "total_events": len(events_sorted),
        "statistics": {
            EVENT_TYPES.get(key, key): count for key, count in counts.items()
        },
        "detection_stats": stats,
        "event_log": [e.to_dict() for e in events_sorted],
        "suggested_sections": [
            {"start": round(s, 1), "end": round(e, 1)} for s, e in suggested_sections
        ],
    }


def print_yolo_summary(report: dict) -> None:
    print(f"\n=== YOLO-analyse (lokal, gratis): {report['video']} ===")
    print(f"Varighet: {report['duration_formatted']}")
    print(f"Frames analysert: {report['detection_stats']['total_frames_analyzed']}")
    print(f"Ball oppdaget i {report['detection_stats']['ball_detection_rate'] * 100:.0f}% av frames")
    print(f"Gj.snitt spillere per frame: {report['detection_stats']['avg_players_detected_per_frame']}")
    print(f"Totalt antall heuristiske hendelser: {report['total_events']}\n")

    if report["statistics"]:
        print("Statistikk (heuristiske hendelser):")
        for label, count in sorted(report["statistics"].items(), key=lambda x: -x[1]):
            print(f"  {label}: {count}")

    print("\nHendelseslogg (heuristikk - ikke semantisk bekreftet):")
    for event in report["event_log"]:
        print(f"  {event['timecode']} - {event['event_type_label']}: {event['description']}")

    sections = report.get("suggested_sections", [])
    if sections:
        ranges = [(s["start"], s["end"]) for s in sections]
        total = sum(e - s for s, e in ranges)
        print(f"\nForeslåtte seksjoner å sende til Claude ({len(ranges)} stk, "
              f"totalt {format_timestamp(total)} av kampen):")
        print(f"  {format_ranges(ranges)}")
        print(
            f"\nFor å analysere kun disse med Claude:\n"
            f"  python analyze.py --video {report['video']} --sections-from-report <denne_rapporten>.json"
        )
    else:
        print("\nIngen seksjoner ble foreslått (ingen tydelig aktivitet oppdaget).")


def save_report(report: dict, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def print_summary(report: dict) -> None:
    print(f"\n=== Kamprapport: {report['video']} ===")
    print(f"Varighet: {report['duration_formatted']}")
    print(f"Totalt antall hendelser: {report['total_events']}\n")

    print("Statistikk:")
    for label, count in sorted(report["statistics"].items(), key=lambda x: -x[1]):
        print(f"  {label}: {count}")

    if report["tactical_observations"]:
        print("\nTaktiske observasjoner:")
        for obs in report["tactical_observations"]:
            print(f"  - {obs}")

    print("\nHendelseslogg:")
    for event in report["event_log"]:
        team = f" [{event['team']}]" if event.get("team") else ""
        print(
            f"  {event['timecode']} - {event['event_type_label']}{team}: "
            f"{event['description']}"
        )
