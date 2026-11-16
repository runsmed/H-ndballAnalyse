"""Parser for offisielle håndball-kampreferater (tekstformat fra live-rapportering).

Format: blokker som starter med "<omgang>. <mm:ss>" (stigende klokke per
omgang, 00:00 ved start -> 30:00 ved slutt), etterfulgt av 1-3 linjer med
hendelsesdetaljer. Denne modulen gjør om slik tekst til strukturerte
hendelser som kan brukes til å klippe ut merkede treningsklipp fra
tilhørende videofil (se clip_dataset.py).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from .events import SHOT_TYPES

HEADER_RE = re.compile(r"^(\d+)\.\s+(\d{1,2}):(\d{2})$")
SCORE_RE = re.compile(r"^\d+\s*-\s*\d+$")

# Hendelsestyper vi IKKE bruker til treningsklipp (for upresise/hyppige til
# at en synsmodell kan spotte dem pålitelig, eller rene strukturmarkører).
SKIPPED_PREFIXES = (
    "Spiller inn:", "Spiller ut:", "Målvakt inn:", "Målvakt ut:",
    "Ballkontroll",  # alltid etterfulgt av en egen "Teknisk feil"-blokk
)
PERIOD_MARKERS = (
    "Kamp startet", "Kamp avsluttet",
    "omgang startet", "omgang avsluttet",
)


@dataclass
class ParsedLogEntry:
    half: int
    clock_seconds: float
    event_type: str
    description: str
    player_name: Optional[str] = None
    assist_player_name: Optional[str] = None
    caused_by_player_name: Optional[str] = None
    score: Optional[str] = None
    shot_type: Optional[str] = None
    raw_lines: List[str] = field(default_factory=list)


@dataclass
class ParseResult:
    events: List[ParsedLogEntry]
    period_markers: List[ParsedLogEntry]
    skipped: List[List[str]]
    unrecognized: List[List[str]]


def _detect_shot_type(text: str) -> Optional[str]:
    lowered = text.lower()
    if "gjennombrudd" in lowered:
        return "breakthrough"
    if "distanseskudd" in lowered:
        return "long_range"
    if "kontring" in lowered:
        return "fast_break"
    if "nærskudd" in lowered or "kantskudd" in lowered or "nærskudd" in lowered:
        return "close_range"
    return None


def _first_non_meta_line(lines: List[str]) -> Optional[str]:
    for line in lines:
        if line.startswith("Assist:") or line.startswith("Forårsaket:") or SCORE_RE.match(line):
            continue
        return line
    return None


def _classify_block(half: int, clock_seconds: float, lines: List[str]) -> Optional[ParsedLogEntry]:
    if not lines:
        return None
    head = lines[0]
    rest = lines[1:]

    if any(head.startswith(p) for p in SKIPPED_PREFIXES):
        return None
    if any(marker in head for marker in PERIOD_MARKERS):
        return ParsedLogEntry(half, clock_seconds, "period_marker", head, raw_lines=lines)

    if head.startswith("Mål!"):
        desc = head[len("Mål!"):].strip()
        assist = next((l[len("Assist:"):].strip() for l in rest if l.startswith("Assist:")), None)
        score = next((l for l in rest if SCORE_RE.match(l)), None)
        player = _first_non_meta_line(rest)
        return ParsedLogEntry(
            half, clock_seconds, "goal", desc, player_name=player,
            assist_player_name=assist, score=score,
            shot_type=_detect_shot_type(desc), raw_lines=lines,
        )

    for prefix, event_type in (
        ("Redning - ", "shot_on_target"),
        ("Utenfor - ", "shot_wide"),
        ("Stolpe - ", "shot_post"),
        ("Blokkert - ", "shot_blocked"),
    ):
        if head.startswith(prefix):
            desc = head[len(prefix):].strip()
            player = _first_non_meta_line(rest)
            return ParsedLogEntry(
                half, clock_seconds, event_type, desc, player_name=player,
                shot_type=_detect_shot_type(desc), raw_lines=lines,
            )

    if head.startswith("7-meter! Tildelt"):
        caused_by = next((l[len("Forårsaket:"):].strip() for l in rest if l.startswith("Forårsaket:")), None)
        player = _first_non_meta_line(rest)
        return ParsedLogEntry(
            half, clock_seconds, "penalty", "Straffekast tildelt",
            player_name=player, caused_by_player_name=caused_by, raw_lines=lines,
        )

    if head.startswith("2 minutter"):
        player = _first_non_meta_line(rest)
        return ParsedLogEntry(half, clock_seconds, "exclusion", "Utvisning 2 minutter", player_name=player, raw_lines=lines)

    if head.startswith("Advarsel"):
        player = _first_non_meta_line(rest)
        return ParsedLogEntry(half, clock_seconds, "warning", "Advarsel (gult kort)", player_name=player, raw_lines=lines)

    if head.startswith("Diskvalifikasjon") or head.startswith("Rødt kort"):
        player = _first_non_meta_line(rest)
        return ParsedLogEntry(half, clock_seconds, "disqualification", "Diskvalifikasjon (rødt kort)", player_name=player, raw_lines=lines)

    if head.startswith("Teknisk feil"):
        desc = head
        player = _first_non_meta_line(rest)
        return ParsedLogEntry(half, clock_seconds, "turnover", desc, player_name=player, raw_lines=lines)

    if head.startswith("Timeout"):
        return ParsedLogEntry(half, clock_seconds, "timeout", "Timeout", raw_lines=lines)

    if head.startswith("Tomt mål"):
        return ParsedLogEntry(half, clock_seconds, "empty_net", "Tomt mål", raw_lines=lines)

    if head.startswith("Målvakt tilbake"):
        return ParsedLogEntry(half, clock_seconds, "goalkeeper_back", "Målvakt tilbake", raw_lines=lines)

    return None  # ukjent - havner i "unrecognized"


def parse_match_log(text: str) -> ParseResult:
    """Parse et offisielt kampreferat (tekstformat) til strukturerte hendelser."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    blocks: List[tuple] = []  # (half, clock_seconds, [body_lines])
    current_header: Optional[tuple] = None
    current_body: List[str] = []

    def flush():
        if current_header is not None:
            blocks.append((current_header[0], current_header[1], current_body[:]))

    for line in lines:
        match = HEADER_RE.match(line)
        if match:
            flush()
            half = int(match.group(1))
            clock_seconds = int(match.group(2)) * 60 + int(match.group(3))
            current_header = (half, clock_seconds)
            current_body = []
        else:
            current_body.append(line)
    flush()

    events: List[ParsedLogEntry] = []
    period_markers: List[ParsedLogEntry] = []
    skipped: List[List[str]] = []
    unrecognized: List[List[str]] = []

    for half, clock_seconds, body in blocks:
        if not body:
            continue
        if any(body[0].startswith(p) for p in SKIPPED_PREFIXES):
            skipped.append(body)
            continue
        entry = _classify_block(half, clock_seconds, body)
        if entry is None:
            unrecognized.append(body)
        elif entry.event_type == "period_marker":
            period_markers.append(entry)
        else:
            events.append(entry)

    events.sort(key=lambda e: (e.half, e.clock_seconds))
    period_markers.sort(key=lambda e: (e.half, e.clock_seconds))

    return ParseResult(events=events, period_markers=period_markers, skipped=skipped, unrecognized=unrecognized)
